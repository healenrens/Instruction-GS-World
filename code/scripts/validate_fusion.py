"""Validate the WHOLE-VIDEO temporal fusion vs the single-frame G0, on the SAME episode + window.
Sweeps the dedupe voxel to find the size that PRESERVES native resolution while ADDING completeness.
Evidence: point count, SAME canonical-view PSNR across the analytic trajectory (must not hurt the
front), NOVEL-view coverage (rotate the camera: single-frame shell has holes the fused set fills).
Usage: python code/scripts/validate_fusion.py [ENV] [SEED] [FUSE_STRIDE]"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import imageio.v2 as iio  # noqa: E402
from scripts.maniskill_gt import generate_episode, build_clip, validate  # noqa: E402
from igsw.gaussians import render_gaussianset  # noqa: E402

dev = "cuda"
ENV = sys.argv[1] if len(sys.argv) > 1 else "PickCube-v1"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs("outputs/fusion_val", exist_ok=True)

rec, instruction, success = generate_episode(ENV, SEED, 512, 512)
print(f"episode frames={len(rec.frames)} instr={instruction[:40]!r}", flush=True)
common = dict(start_frac=0.4, window_steps=80)
clip_sf = build_clip(rec, 16, dev, fuse_stride=0, **common)
H, W = clip_sf["H"], clip_sf["W"]
K = clip_sf["K_intr"]; vm0 = clip_sf["viewmat"]


def render(g, vm):
    c, a, _ = render_gaussianset(g, vm[None], K[None], W, H)
    return c[0].clamp(0, 1).cpu().numpy(), a[0].clamp(0, 1).cpu().numpy()


def novel_viewmat(vm, P, deg):
    c = P.mean(0); th = math.radians(deg)
    Ry = torch.tensor([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]],
                      device=vm.device, dtype=vm.dtype)
    c2w = torch.linalg.inv(vm)
    newt = c + Ry @ (c2w[:3, 3] - c)
    c2w_new = torch.eye(4, device=vm.device, dtype=vm.dtype)
    c2w_new[:3, :3] = Ry @ c2w[:3, :3]; c2w_new[:3, 3] = newt
    return torch.linalg.inv(c2w_new)


full_sf, dyn_sf, _ = validate(clip_sf, dev, out_dir=None)
vm25 = novel_viewmat(vm0, clip_sf["g0"].means, 25.0)
_, a_sf25 = render(clip_sf["g0"], vm25)
cov_sf = float((a_sf25 > 0.5).mean())
print(f"\n[single-frame] N={len(clip_sf['g0'])}  sameview_full={np.mean(full_sf[1:]):.2f}  "
      f"sameview_dyn={np.nanmean(dyn_sf[1:]):.2f}  novel25_cov={cov_sf:.3f}", flush=True)
rgb_sf25, _ = render(clip_sf["g0"], vm25)

for voxel in (0.001, 0.0015, 0.002, 0.003):
    clip_fu = build_clip(rec, 16, dev, fuse_stride=STRIDE, fuse_voxel=voxel, **common)
    full_fu, dyn_fu, _ = validate(clip_fu, dev, out_dir=None)
    rgb_fu25, a_fu25 = render(clip_fu["g0"], vm25)
    cov_fu = float((a_fu25 > 0.5).mean())
    print(f"[fused v={voxel*1000:.1f}mm] N={len(clip_fu['g0'])} (x{len(clip_fu['g0'])/len(clip_sf['g0']):.2f})  "
          f"sameview_full={np.mean(full_fu[1:]):.2f}  sameview_dyn={np.nanmean(dyn_fu[1:]):.2f}  "
          f"novel25_cov={cov_fu:.3f} ({'+' if cov_fu >= cov_sf else ''}{100*(cov_fu-cov_sf)/cov_sf:.0f}%)", flush=True)
    iio.imwrite(f"outputs/fusion_val/novel25_single_vs_fused_v{int(voxel*1000*10)}.png",
                (np.concatenate([rgb_sf25, rgb_fu25], axis=1) * 255).astype(np.uint8))
print("\nwrote outputs/fusion_val/novel25_single_vs_fused_v*.png (left=single, right=fused)", flush=True)
