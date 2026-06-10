"""GT RECONSTRUCTION video: [ real sim RGB | our fused-Gaussian reconstruction ] side by side, over the
clip. The right panel = the whole-video-FUSED canonical Gaussians moved by the EXACT analytic trajectory
(traj) and rendered from the SAME camera (isotropic Gaussians => moving means is an exact render). This
is literally 'the learning video -> processed into Gaussians -> rendered back', so you can judge the GT
data fidelity vs the original. No model, no prediction — pure GT.
Usage: python code/scripts/viz_gt_recon_video.py [clip.pt]"""
import os
import sys

import torch

sys.path.insert(0, "code")
import imageio.v2 as iio  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset, psnr  # noqa: E402

dev = "cuda"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/maniskill_fused/pickcube_s1002_train.pt"
c = torch.load(CLIP, map_location=dev, weights_only=False)
g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
traj = c["traj"].to(dev); K = min(16, int(c["Kf"])); H, W = int(c["H"]), int(c["W"])
Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
gt_rgb = c["gt_rgb"].to(dev).float() / 255.0          # [Kf+1,H,W,3] REAL sim RGB (the original video)
os.makedirs("outputs/gt_recon", exist_ok=True)
tag = os.path.basename(CLIP).replace(".pt", "")

w = iio.get_writer(f"outputs/gt_recon/{tag}_GTrecon.mp4", fps=6)
ps = []
for t in range(K + 1):
    gm = GaussianSet(traj[t].float(), g0.quats, g0.scales, g0.opacities, g0.colors, None)
    rec, alpha, _ = render_gaussianset(gm, vm[None], Ki[None], W, H)
    rec = rec[0].clamp(0, 1)
    real = gt_rgb[t].clamp(0, 1)
    # composite the recon over the real frame-0 only where the cloud is transparent (fills the far-bg
    # we cropped at depth_max); the workspace (table/arm/cube) is the actual Gaussian reconstruction.
    a = alpha[0].clamp(0, 1)
    rec_c = rec + (1 - a) * gt_rgb[0]
    ps.append(psnr(rec_c, real))
    row = torch.cat([real, rec_c], 1)                 # [H, 2W, 3] : REAL | RECON
    w.append_data((row * 255).to(torch.uint8).cpu().numpy())
w.close()
print(f"{tag}: N={len(g0)} K={K} recon-PSNR mean={sum(ps)/len(ps):.1f} "
      f"frame0={ps[0]:.1f} -> outputs/gt_recon/{tag}_GTrecon.mp4", flush=True)
