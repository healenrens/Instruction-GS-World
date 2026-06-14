"""GPSToken-JEPA world model trainer (PLAN_GPSTOKEN_JEPA_zh.md, agent.md §95). Single-GPU v1.

Per clip (frame0 -> frameK window):
  place sparse tokens (entropy+GT-mover saliency) -> lift 3D (nearest dense means) -> frozen Qwen feat
  -> DiT predict per-token future xyz (geom, load-bearing) + future feature (JEPA aux)
  -> L = L_geom + w_jepa*L_jepa + w_sigreg*SIGReg + w_ground*(InfoNCE + w_cf*counterfactual).
Rotation is NOT trained — read out by Kabsch at eval. E1: --geom_mode {xyz, flowd}.

  python code/scripts/train_gpstoken_wm.py --data data/mix_v15 --out checkpoints/gpswm_xyz \
      --geom_mode xyz --L 512 --steps 2000
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, SIGReg, place_tokens, sample_grid_feat, project_to_uv  # noqa: E402
from igsw.gpstoken_wm.losses import geom_loss, jepa_loss, relevance_infonce, counterfactual_push  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402

LIBERO_NOUNS = ["alphabet soup", "cream cheese", "butter", "tomato sauce", "ketchup", "milk",
                "orange juice", "chocolate pudding", "bbq sauce", "salad dressing"]


def move_vlm_inputs(inputs, dev, dtype):
    out = {}
    for k, v in inputs.items():
        out[k] = (v.to(dev, dtype=dtype) if (torch.is_tensor(v) and v.is_floating_point())
                  else (v.to(dev) if torch.is_tensor(v) else v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geom_mode", default="xyz", choices=["xyz", "flowd"])
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--fdim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=30.0, help="GT-mover saliency boost for token placement")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_jepa", type=float, default=0.5)
    ap.add_argument("--w_sigreg", type=float, default=0.05)
    ap.add_argument("--w_ground", type=float, default=1.0)
    ap.add_argument("--w_cf", type=float, default=0.0, help="counterfactual weight (extra Qwen fwd; 0=off)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--max_clips", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    model = GPSTokenWM(geom_mode=args.geom_mode, fdim=args.fdim).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    sigreg = SIGReg(num_proj=512).to(dev)
    enc = model.encoder
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    print(f"[gpswm] trainable={model.num_trainable()/1e9:.3f}B geom_mode={args.geom_mode} L={args.L}", flush=True)

    clips = sorted(glob.glob(f"{args.data}/*_train.pt"))
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"[gpswm] {len(clips)} train clips", flush=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    rng = np.random.default_rng(0)

    step, t0 = 0, time.time()
    while step < args.steps:
        for cp in clips:
            if step >= args.steps:
                break
            ok = True
            try:
                c = torch.load(cp, map_location=dev, weights_only=False)
                means = c["means"].to(dev).float()
                uv = c["uv"].to(dev).float()
                traj = c["traj"].to(dev).float()                                # [K+1,N,3]
                N = means.shape[0]
                H, W = int(c["H"]), int(c["W"])
                K = int(c["Kf"])
                Ki = c["K_intr"].to(dev).float()
                vm = c["viewmat"].to(dev).float()
                gt_rgb = c["gt_rgb"]                                            # [K+1,H,W,3] uint8 (cpu)
                is_obj = c["is_obj"].to(dev) if "is_obj" in c else None
                instr = c.get("instruction", "")
                n_keep = N - int(c.get("n_fill", 0))
                disp = (traj[K] - traj[0]).norm(dim=-1)                         # [N]
                sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
                rgb0 = gt_rgb[0].cpu().numpy().astype(np.uint8)
                cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
                M = idx.shape[0]
                if M < 16:
                    raise ValueError("too few tokens")
                tok_xyz0 = means[idx]                                           # [M,3]
                xyz1_gt = traj[K][idx]                                          # [M,3] clean GT
                sig_n = (sig / float(max(H, W))).clamp(0, 1)                    # footprint, normalized
                center = means[:n_keep].mean(0, keepdim=True)
                radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
            except Exception as e:
                print(f"[skip] prep {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
                step += 1
                continue

            vlm0 = move_vlm_inputs(enc.build_inputs(instr, rgb0), dev, torch.bfloat16)
            with amp:
                ctx, ctxm, cond, text_feats = model.encode_cond(vlm0)
                grid0, ghw0 = enc.image_grid_features(vlm0)
                gfeat0 = sample_grid_feat(grid0, ghw0, cen, H, W)               # [M,Hq]
                tok_feat = model.feat_in(gfeat0).float()                       # [M,fdim] fp32 for loss path
                x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctx, ctxm, cond)
                xyz1_pred, feat_pred = model.heads(x, tok_xyz0, Ki, vm)

            # JEPA target: frozen-encode frameK, sample at the token's GT-future uv (stop-grad)
            imgK = gt_rgb[K].cpu().numpy().astype(np.uint8)
            vlmK = move_vlm_inputs(enc.build_inputs(instr, imgK), dev, torch.bfloat16)
            with torch.no_grad(), amp:
                gridK, ghwK = enc.image_grid_features(vlmK)
                fut_uv = project_to_uv(xyz1_gt, Ki, vm)
                gfeatK = sample_grid_feat(gridK, ghwK, fut_uv, H, W)
                tgt_feat = model.feat_in(gfeatK).float()

            # ---- losses ----
            l_geom = geom_loss(xyz1_pred.float(), xyz1_gt.float())
            fp = F.layer_norm(feat_pred.float(), (args.fdim,))
            ft = F.layer_norm(tgt_feat, (args.fdim,)).detach()
            l_jepa = jepa_loss(fp, ft)
            l_sig = sigreg(tok_feat.float())
            l_inst = tok_feat.new_zeros(())
            l_cf = tok_feat.new_zeros(())
            rel = None
            if is_obj is not None and args.w_ground > 0:
                text_emb = text_feats.mean(0)
                rel = model.relevance(tok_feat, text_emb)
                pos = is_obj[idx].bool()
                l_inst = relevance_infonce(rel, pos)
                if args.w_cf > 0 and pos.any() and not pos.all():
                    wj = LIBERO_NOUNS[int(rng.integers(len(LIBERO_NOUNS)))]
                    if wj not in instr:
                        wrong = instr
                        for n in LIBERO_NOUNS:
                            if n in instr:
                                wrong = instr.replace(n, wj)
                                break
                        vlmw = move_vlm_inputs(enc.build_inputs(wrong, rgb0), dev, torch.bfloat16)
                        with amp:
                            _, _, _, tfw = model.encode_cond(vlmw)
                            relw = model.relevance(tok_feat, tfw.mean(0))
                        l_cf = counterfactual_push(relw, pos)

            loss = (l_geom + args.w_jepa * l_jepa + args.w_sigreg * l_sig
                    + args.w_ground * (l_inst + args.w_cf * l_cf))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            else:
                print(f"[skip] non-finite grad @s{step}", flush=True)

            if step % args.log_every == 0:
                with torch.no_grad():
                    err_pred = (xyz1_pred - xyz1_gt).norm(dim=-1).mean()
                    err_static = (tok_xyz0 - xyz1_gt).norm(dim=-1).mean()
                    mv = disp[idx] > 0.01
                    if mv.any():
                        pd = (xyz1_pred - tok_xyz0)[mv]
                        gd = (xyz1_gt - tok_xyz0)[mv]
                        dcos = F.cosine_similarity(pd, gd, dim=-1).mean()
                    else:
                        dcos = torch.zeros((), device=dev)
                    relsel = (is_obj[idx][rel.argmax()].float()
                              if (rel is not None and is_obj[idx].any()) else torch.zeros((), device=dev))
                print(f"s{step} loss{loss.item():.3f} geom{l_geom.item():.4f} jepa{l_jepa.item():.3f} "
                      f"sig{l_sig.item():.3f} inst{float(l_inst):.3f} | M{M} skill{(err_static-err_pred).item()*100:.1f}cm "
                      f"errp{err_pred.item()*100:.1f} dcos{dcos.item():.2f} relSel{float(relsel):.0f} "
                      f"fstd{tok_feat.float().std().item():.2f} {(step+1)/(time.time()-t0):.2f}it/s", flush=True)
            step += 1
            if step % args.save_every == 0 or step == args.steps:
                # save ONLY trainable params (exclude the frozen 2.4B Qwen encoder -> small ckpt)
                sd = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
                torch.save({"model": sd, "args": vars(args), "step": step},
                           f"{args.out}/wm_{step:06d}.pt")
                print(f"[gpswm] saved wm_{step:06d}.pt ({len(sd)} tensors)", flush=True)
    print("[gpswm] DONE", flush=True)


if __name__ == "__main__":
    main()
