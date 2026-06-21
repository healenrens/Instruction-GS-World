"""GPSToken-JEPA world model trainer (PLAN_GPSTOKEN_JEPA_zh.md, agent.md §95). DDP (multi-GPU).

Per clip (frame0 -> frameK window): place sparse tokens (entropy+GT-mover saliency) -> lift 3D (nearest
dense means) -> frozen Qwen feat -> DiT predict per-token future xyz (geom, load-bearing) + future feat
(JEPA aux) -> L = L_geom + w_jepa*L_jepa + w_sigreg*SIGReg + w_ground*InfoNCE. Rotation NOT trained
(Kabsch readout at eval). E1: --geom_mode {xyz, flowd}.

  torchrun --nproc_per_node=2 code/scripts/train_gpstoken_wm.py --data data/mix_v15 \
      --out checkpoints/gpswm_xyz --geom_mode xyz --L 512 --steps 1500
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def mv_in(inputs, dev):
    return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inputs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geom_mode", default="xyz", choices=["xyz", "flowd"])
    ap.add_argument("--feat_source", default="qwen", choices=["qwen", "dino"],
                    help="token visual feature source: qwen VLM patches (baseline) or frozen DINOv2 dense (A)")
    ap.add_argument("--dino_imgsize", type=int, default=518, help="DINOv2 input res (mult of 14); 770=55x55 finer grid = lower noise floor")
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--fdim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_mag", type=float, default=0.0, help="mover-magnitude loss (DEAD END — kills direction)")
    ap.add_argument("--w_motion", type=float, default=0.0, help="motion-weighted geom loss (direction-preserving mag fix)")
    ap.add_argument("--mw_cap", type=float, default=10.0, help="cap on the per-token motion up-weight (raise to push magnitude on sparse big movers, e.g. SpaTracker GT)")
    ap.add_argument("--norm_target", type=int, default=0, help="1=scale-decoupled geom (predict scale-invariant FIELD + single global scale); attacks the aleatoric-scale under-prediction")
    ap.add_argument("--w_scale", type=float, default=0.1, help="weight on the single-global-scale loss (low = treat scale as aleatoric, focus on the field)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation: effective batch = accum clips/step (stabilizes 1-clip/step, esp. with norm_target)")
    ap.add_argument("--img_loss", type=int, default=0, help="1=supervise on NORMALIZED 2D image displacement (Δu/W,Δv/H) instead of 3D meters; logs 'mag'=image-flow magR, 'dcos'=image dcos")
    ap.add_argument("--w_depth", type=float, default=0.5, help="weight on the normalized depth-change (Δlog z) term under img_loss (full 3D = image-flow + depth)")
    ap.add_argument("--fuse", type=int, default=0, help="v2 fused arch: grounding=GATE modulating motion; JEPA=read-out into RAW pretrained latent (footprint-pooled, derived from motion), not parallel heads")
    ap.add_argument("--cam_cond", type=int, default=0, help="B: condition the predictor on the camera pose (global emb -> cond + per-token cam-frame pos -> hidden); makes VARYING cameras a generalization asset not poison. Zero-init => warm-startable from the fixed-cam v2")
    ap.add_argument("--jepa_couple", type=int, default=0, help="ablation: 1 = JEPA gradient flows INTO the trunk (multi-task, future-feature task helps motion?); 0 = stop-grad read-out (default, JEPA does not affect motion)")
    ap.add_argument("--traj_pred", type=int, default=0, help="CURVE: geom head outputs PER-FRAME (Δu,Δv,Δlogz) for t=1..Kf -> full 3D trajectory; img_loss supervised per-frame (GT=traj[t]). Endpoint=last waypoint. vs the straight single-displacement baseline")
    ap.add_argument("--init_from", default="", help="warm-start: load model weights from this ckpt (strict=False; new modules e.g. cam heads keep their zero-init) -> keep the fixed-cam v2 head start for B")
    ap.add_argument("--cond_scale", type=int, default=0, help="1=condition the predictor on the (oracle) global motion scale; tests whether SUPPLYING the scale fixes magnitude (path-1 premise)")
    ap.add_argument("--w_jepa", type=float, default=0.5)
    ap.add_argument("--w_sigreg", type=float, default=0.05)
    ap.add_argument("--w_ground", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=750)
    ap.add_argument("--lr_min_frac", type=float, default=1.0,
                    help="<1 = cosine-decay LR to lr*lr_min_frac over steps (stabilizes; 1.0=constant LR legacy)")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = f"cuda:{local}"
    else:
        rank, world, local, dev = 0, 1, 0, "cuda"
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    # peek one clip for Kf so the traj head is sized right
    _probe = sorted(glob.glob(f"{args.data}/*_train.pt"))
    _Kf = int(torch.load(_probe[0], map_location="cpu", weights_only=False)["Kf"]) if _probe else 12
    model = GPSTokenWM(geom_mode=args.geom_mode, fdim=args.fdim, feat_source=args.feat_source,
                       dino_imgsize=args.dino_imgsize, traj_pred=bool(args.traj_pred), Kf=_Kf).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.w_jepa, model.w_sigreg, model.w_ground = args.w_jepa, args.w_sigreg, args.w_ground
    model.norm_target, model.w_scale = bool(args.norm_target), args.w_scale
    model.img_loss = bool(args.img_loss)
    model.w_depth = args.w_depth
    model.fuse = bool(args.fuse)
    model.cam_cond = bool(args.cam_cond)
    model.jepa_couple = bool(args.jepa_couple)
    model.cond_scale = bool(args.cond_scale)
    model.w_mag, model.w_motion = args.w_mag, args.w_motion
    model.mw_cap = args.mw_cap
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev, weights_only=False)["model"]
        miss, _ = model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[gpswm] warm-start {args.init_from}: loaded {len(sd)} tensors, {len(miss)} new (zero-init, e.g. cam_head/cam_tok_head)", flush=True)
    enc = model.encoder
    model._ddp_touch = ddp                                  # all params participate -> find_unused not needed
    mdl = DDP(model, device_ids=[local], find_unused_parameters=False, broadcast_buffers=False) if ddp else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * args.lr_min_frac)
             if args.lr_min_frac < 1.0 else None)
    if is_main:
        print(f"[gpswm] trainable={model.num_trainable()/1e9:.3f}B geom_mode={args.geom_mode} L={args.L} "
              f"world={world}", flush=True)

    clips = sorted(glob.glob(f"{args.data}/*_train.pt"))
    shard = clips[rank::world]                                   # each rank a disjoint clip shard
    if is_main:
        print(f"[gpswm] {len(clips)} train clips ({len(shard)}/rank)", flush=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step, t0, ci = 0, time.time(), 0
    accum = max(1, args.accum)
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        agg, n_ok = None, 0
        for _m in range(accum):                                  # gradient accumulation -> effective batch = accum clips
            cp = shard[ci % len(shard)]; ci += 1
            ok = True
            try:
                c = torch.load(cp, map_location=dev, weights_only=False)
                means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
                traj = c["traj"].to(dev).float(); N = means.shape[0]
                H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
                instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
                disp = (traj[K] - traj[0]).norm(dim=-1)
                sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
                rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
                imgK = c["gt_rgb"][K].cpu().numpy().astype(np.uint8)
                cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
                if idx.shape[0] < 16:
                    raise ValueError("too few tokens")
                center = means[:n_keep].mean(0, keepdim=True)
                batch = {
                    "vlm0": mv_in(enc.build_inputs(instr, rgb0), dev),
                    "vlmK": mv_in(enc.build_inputs(instr, imgK), dev),
                    "cen": cen, "sig_n": (sig / float(max(H, W))).clamp(0, 1),
                    "tok_xyz0": means[idx], "xyz1_gt": traj[K][idx], "disp_tok": disp[idx],
                    "traj_gt": (traj[1:K + 1][:, idx] if args.traj_pred else None),  # [Kf,M,3] frames t=1..Kf
                    "is_obj_tok": (c["is_obj"].to(dev)[idx] if "is_obj" in c else None),
                    "center": center, "radius": (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6),
                    "K_intr": c["K_intr"].to(dev).float(), "viewmat": c["viewmat"].to(dev).float(),
                    "H": H, "W": W, "rgb0_np": rgb0, "rgbK_np": imgK,
                }
            except Exception as e:
                ok = False
                if is_main:
                    print(f"[skip] prep {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
            if ddp:
                flag = torch.tensor([1.0 if ok else 0.0], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                ok = flag.item() > 0.5
            if not ok:
                continue
            with amp:
                loss, logs = mdl(batch)
            (loss / accum).backward()
            n_ok += 1
            agg = ({k: v.detach() for k, v in logs.items()} if agg is None
                   else {k: agg[k] + logs[k].detach() for k in agg})
        if n_ok == 0:
            step += 1
            continue
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        finite = torch.tensor([1.0 if torch.isfinite(gnorm) else 0.0], device=dev)
        if ddp:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() > 0.5:
            opt.step()
        elif is_main:
            print(f"[skip] non-finite grad @s{step}", flush=True)
        if sched is not None:
            sched.step()
        logs = {k: v / n_ok for k, v in agg.items()}

        if is_main and step % args.log_every == 0:
            print(f"s{step} loss{logs['loss'].item():.3f} geom{logs['geom'].item():.4f} "
                  f"mag{logs['mag'].item():.2f} jepa{logs['jepa'].item():.3f} sig{logs['sig'].item():.3f} inst{logs['inst'].item():.3f} "
                  f"| skill{logs['skill'].item()*100:.1f}cm errp{logs['errp'].item()*100:.1f} "
                  f"dcos{logs['dcos'].item():.2f} relSel{logs['relsel'].item():.0f} fstd{logs['fstd'].item():.2f} "
                  f"{(step+1)/(time.time()-t0):.2f}it/s", flush=True)
        step += 1
        if is_main and (step % args.save_every == 0 or step == args.steps):
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            torch.save({"model": sd, "args": vars(args), "step": step}, f"{args.out}/wm_{step:06d}.pt")
            print(f"[gpswm] saved wm_{step:06d}.pt ({len(sd)} tensors)", flush=True)
    if is_main:
        print("[gpswm] DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
