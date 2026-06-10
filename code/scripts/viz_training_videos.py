"""TRAINING-DATA quality videos. For several clips, render the clean whole-video-FUSED GT rollout in a
3-panel video:  [ natural appearance | SEGMENTATION color | MOVER highlight ].
  - natural   = the fused real-RGB Gaussians moved by the EXACT analytic trajectory (geometry quality)
  - seg color = one distinct color per entity (the per-Gaussian SEMANTIC label we already have)
  - mover     = RED if this Gaussian actually moves in the GT (disp>1cm), GRAY if static
This lets the user judge data quality AND confirms the GT is semantically correct (table=static/gray,
cube+arm=moving/red) — i.e. the failure (table sinks, cube frozen) is the MODEL, not the data.
Usage: python code/scripts/viz_training_videos.py"""
import colorsys
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
import imageio.v2 as iio  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset  # noqa: E402

dev = "cuda"
DATA = "data/maniskill_fused"
os.makedirs("outputs/train_videos", exist_ok=True)
clips = (sorted(glob.glob(f"{DATA}/*pickcube*_train.pt"))[:2] +
         sorted(glob.glob(f"{DATA}/*pushcube*_train.pt"))[:1] +
         sorted(glob.glob(f"{DATA}/*stackcube*_heldtask.pt"))[:1])


def seg_colors(seg, uniq):
    C = torch.zeros(len(seg), 3, device=seg.device)
    for i, s in enumerate(uniq):
        rgb = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.75, 0.95)
        C[seg == int(s)] = torch.tensor(rgb, device=seg.device, dtype=torch.float32)
    return C


for p in clips:
    c = torch.load(p, map_location=dev, weights_only=False)
    g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
    traj = c["traj"].to(dev); K = min(16, int(c["Kf"])); H, W = int(c["H"]), int(c["W"])
    Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
    seg = c["seg_per_g"].to(dev); uniq = torch.unique(seg).tolist()
    segC = seg_colors(seg, uniq)
    disp = (traj[K] - traj[0]).norm(dim=-1); mover = disp > 0.01
    moverC = torch.where(mover[:, None], torch.tensor([0.95, 0.15, 0.15], device=dev),
                         torch.tensor([0.45, 0.45, 0.45], device=dev))
    tag = os.path.basename(p).replace(".pt", "")
    w = iio.get_writer(f"outputs/train_videos/{tag}_GTquality.mp4", fps=6)
    for t in range(K + 1):
        out = []
        for col in (g0.colors, segC, moverC):
            g = GaussianSet(traj[t].float(), g0.quats, g0.scales, g0.opacities, col, None)
            im, _, _ = render_gaussianset(g, vm[None], Ki[None], W, H)
            out.append(im[0].clamp(0, 1))
        w.append_data((torch.cat(out, 1) * 255).to(torch.uint8).cpu().numpy())
    w.close()
    nm = c.get("name_by_sid", {})
    print(f"{tag}  N={len(g0)}  movers={int(mover.sum())}/{len(g0)}  entities={[nm.get(int(s), s) for s in uniq][:6]}", flush=True)
print("done -> outputs/train_videos/", flush=True)
