"""Feasibility test for the lifting+rendering substrate.

Pipeline: decode head frame(s) -> Pi3 lift -> GaussianSet -> recover K ->
gsplat render from each source camera -> PSNR vs input. Saves side-by-side PNGs.

This validates: Pi3 inference, the GaussianSet plumbing, intrinsics recovery, and
the gsplat rasterizer (JIT-compiled on first call) — the substrate the dynamics
model will sit on top of.

Usage:
    python scripts/feasibility_lift_render.py --ep 0 --frame 600 --n 1
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data import AgiBotLeRobotTask, list_tasks  # noqa: E402
from igsw.lifting import Pi3Lifter, points_to_gaussians  # noqa: E402
from igsw.gaussians import (  # noqa: E402
    render_gaussianset,
    psnr,
    intrinsics_from_local_points,
    viewmat_from_pose,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--frame", type=int, default=600)
    ap.add_argument("--n", type=int, default=1, help="number of frames (stride 5) to lift jointly")
    ap.add_argument("--cam", default="observation.images.head")
    ap.add_argument("--opacity", type=float, default=0.9)
    ap.add_argument("--scale_factor", type=float, default=1.0)
    ap.add_argument("--conf_thr", type=float, default=0.1)
    ap.add_argument("--edge_rtol", type=float, default=0.03, help="<=0 disables edge removal")
    ap.add_argument("--out", default="outputs/feasibility")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from PIL import Image

    task = args.task or list_tasks()[0]
    t = AgiBotLeRobotTask(task)
    idx = [args.frame + 5 * i for i in range(args.n)]
    print(f"[data] task={os.path.basename(os.path.dirname(task))} ep={args.ep} frames={idx} cam={args.cam}")
    print(f"[data] language = {t.language(args.ep)!r}")
    frames = t.decode_frames(args.ep, args.cam, idx)  # [n,H,W,3] uint8
    print(f"[data] decoded {frames.shape}")

    print("[pi3] loading model + lifting ...")
    lifter = Pi3Lifter(device="cuda")
    tic = time.time()
    res = lifter.lift(frames, conf_thr=args.conf_thr, edge_rtol=args.edge_rtol)
    torch.cuda.synchronize()
    print(f"[pi3] lift done in {time.time()-tic:.2f}s  "
          f"points={tuple(res['points'].shape)} mask_kept={int(res['mask'].sum())}/{res['mask'].numel()}")

    dev = torch.device("cuda")
    pts = res["points"].to(dev)
    imgs = res["images"].to(dev)
    mask = res["mask"].to(dev)
    local = res["local_points"].to(dev)
    poses = res["camera_poses"].to(dev)
    n, _, H, W = imgs.shape

    gs = points_to_gaussians(pts, imgs, mask, opacity_init=args.opacity, scale_factor=args.scale_factor)
    print(f"[gs] {len(gs)} gaussians; scale[min/med/max]="
          f"{gs.scales.min():.4f}/{gs.scales.median():.4f}/{gs.scales.max():.4f}")

    # render every source view from its own camera, compute PSNR
    psnrs, mpsnrs = [], []
    for i in range(n):
        K = intrinsics_from_local_points(local[i])
        viewmat = viewmat_from_pose(poses[i])
        tic = time.time()
        colors, alphas, _ = render_gaussianset(gs, viewmat, K, width=W, height=H)
        torch.cuda.synchronize()
        rend = colors[0].clamp(0, 1)          # [H,W,3]
        alpha = alphas[0, ..., 0]             # [H,W]
        gt = imgs[i].permute(1, 2, 0)         # [H,W,3]
        full = psnr(rend, gt)
        amask = alpha > 0.5
        mp = psnr(rend[amask], gt[amask]) if amask.any() else float("nan")
        psnrs.append(full); mpsnrs.append(mp)
        if i == 0:
            print(f"[render] view0 {time.time()-tic:.3f}s  K=fx{K[0,0]:.1f} fy{K[1,1]:.1f} "
                  f"cx{K[0,2]:.1f} cy{K[1,2]:.1f}  PSNR full={full:.2f} masked={mp:.2f} "
                  f"coverage={float(amask.float().mean()):.2f}")
        side = torch.cat([gt, rend], dim=1).cpu().numpy()
        Image.fromarray((side * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"ep{args.ep}_f{idx[i]}_view{i}_gt_vs_render.png"))

    gs.save_ply(os.path.join(args.out, f"ep{args.ep}_f{args.frame}.ply"))
    print(f"[result] mean PSNR full={np.mean(psnrs):.2f}  masked={np.nanmean(mpsnrs):.2f}")
    print(f"[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
