"""Eval the GPSToken-JEPA world model (PLAN test): per-token EPE3D, mover direction, mag-ratio,
grounding hit@1, and the EMERGENT-ROTATION readout (E4) — Kabsch per entity on the predicted token
positions (rotation is NEVER trained; this only MEASURES what the per-token translation field produced).
Placement uses GT-mover saliency (oracle placement — isolates the predictor; rel-grid placement = a later
deployment test). Usage: python code/scripts/eval_gpstoken_wm.py --ckpt <wm.pt> --data data/mix_v15 --split heldseed
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gpstoken_wm.losses import kabsch_R  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def rot_deg(Ra, Rb):
    c = ((Ra.T @ Rb).diagonal().sum() - 1) / 2
    return float(torch.arccos(c.clamp(-1, 1)) * 180 / np.pi)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="heldseed")
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--beta", type=float, default=30.0)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    model = GPSTokenWM(geom_mode=cargs.get("geom_mode", "xyz"), fdim=cargs.get("fdim", 128)).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    miss_train = [k for k in missing if not k.startswith("encoder.")]
    print(f"[eval] {args.ckpt} geom_mode={cargs.get('geom_mode')} | non-encoder missing={len(miss_train)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enc = model.encoder

    def mv_in(inp):
        return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inp.items()}

    clips = sorted(glob.glob(f"{args.data}/*_{args.split}.pt"))
    epe, dcos, magr, rote, transe, fivefive, hit, gtrot = [], [], [], [], [], [], [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    for cp in clips:
        c = torch.load(cp, map_location=dev, weights_only=False)
        means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
        traj = c["traj"].to(dev).float(); N = means.shape[0]
        H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
        Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
        is_obj = c["is_obj"].to(dev) if "is_obj" in c else None
        seg = c["seg_per_g"].to(dev).long() if "seg_per_g" in c else None
        instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
        disp = (traj[K] - traj[0]).norm(dim=-1)
        sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
        rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
        cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
        tok_xyz0 = means[idx]; xyz1_gt = traj[K][idx]
        sig_n = (sig / float(max(H, W))).clamp(0, 1)
        center = means[:n_keep].mean(0, keepdim=True)
        radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
        vlm0 = mv_in(enc.build_inputs(instr, rgb0))
        with torch.no_grad(), amp:
            ctx, ctxm, cond, text_feats = model.encode_cond(vlm0)
            grid0, ghw0 = enc.image_grid_features(vlm0)
            tok_feat = model.feat_in(sample_grid_feat(grid0, ghw0, cen, H, W)).float()
            x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctx, ctxm, cond)
            xyz1_pred, _ = model.heads(x, tok_xyz0, Ki, vm)
        xyz1_pred = xyz1_pred.float()
        e = (xyz1_pred - xyz1_gt).norm(dim=-1)
        epe.append(e.median().item() * 100)
        mv = disp[idx] > 0.01
        if mv.sum() >= 3:
            pd, gd = (xyz1_pred - tok_xyz0)[mv], (xyz1_gt - tok_xyz0)[mv]
            dcos.append(F.cosine_similarity(pd, gd, dim=-1).mean().item())
            magr.append((pd.norm(dim=-1).median() / gd.norm(dim=-1).median().clamp_min(1e-6)).item())
        # per-entity emergent-rotation readout (5°5cm) + grounding
        if seg is not None:
            segk = seg[idx]
            for s in torch.unique(segk).tolist():
                if s == 0:
                    continue
                sel = (segk == s) & mv
                if sel.sum() < 6:
                    continue
                Rp = kabsch_R(tok_xyz0[sel], xyz1_pred[sel]); Rg = kabsch_R(tok_xyz0[sel], xyz1_gt[sel])
                tp = (xyz1_pred[sel] - tok_xyz0[sel]).mean(0); tg = (xyz1_gt[sel] - tok_xyz0[sel]).mean(0)
                re = rot_deg(Rp, Rg); te = (tp - tg).norm().item()
                rote.append(re); transe.append(te * 100)
                fivefive.append(1.0 if (re <= 5 and te <= 0.05) else 0.0)
                gtrot.append(rot_deg(torch.eye(3, device=dev), Rg))
        if is_obj is not None and is_obj[idx].any():
            with torch.no_grad(), amp:
                rel = model.relevance(tok_feat, text_feats.mean(0))
            hit.append(float(is_obj[idx][rel.argmax()].item()))

    print(f"[{args.split}] n_clip={len(clips)}", flush=True)
    print(f"  EPE3D median {med(epe):.1f}cm   dcos {med(dcos):+.2f}   mag-ratio {med(magr):.2f}", flush=True)
    print(f"  grounding hit@1 {np.mean(hit)*100 if hit else float('nan'):.0f}%  (n={len(hit)})", flush=True)
    print(f"  [emergent-rot readout] 5°5cm {np.mean(fivefive)*100 if fivefive else float('nan'):.0f}%  "
          f"rot-err median {med(rote):.1f}°  trans-err {med(transe):.1f}cm  GT-rot median {med(gtrot):.1f}°  (n_ent={len(rote)})", flush=True)


if __name__ == "__main__":
    main()
