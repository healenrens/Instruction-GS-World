"""Export the TRAINING Gaussians as STANDARD 3DGS .ply (full splat attributes: means, log-scales,
wxyz quats, inverse-sigmoid opacity, SH-DC colors) so they open directly in a Gaussian-splat viewer
(SuperSplat https://supersplat.play.canva.com , antimatter15/splat, etc.) for 3D inspection of the
DATA — not a rendered video. Exports the fused canonical scene (t=0) in NATURAL color and in
SEGMENTATION color (per-entity), plus a couple of analytic-moved timesteps (the motion).
Usage: python code/scripts/export_3dgs_ply.py [clip.pt]"""
import colorsys
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
from igsw.gaussians import GaussianSet  # noqa: E402

C0 = 0.28209479177387814  # SH band-0 constant


def write_3dgs_ply(path, means, scales, quats, opac, colors):
    means = means.detach().cpu().float().numpy()
    scales = scales.detach().cpu().float().numpy()
    if scales.ndim == 1:
        scales = np.repeat(scales[:, None], 3, 1)
    quats = quats.detach().cpu().float().numpy()
    opac = opac.detach().cpu().float().numpy().reshape(-1)
    colors = colors.detach().cpu().float().clamp(0, 1).numpy()
    n = means.shape[0]
    f_dc = (colors - 0.5) / C0
    log_scale = np.log(np.clip(scales, 1e-8, None))
    o = np.clip(opac, 1e-6, 1 - 1e-6)
    opac_logit = np.log(o / (1 - o))
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)   # wxyz; isotropic so order is moot
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n +
           "".join("property float %s\n" % p for p in props) + "end_header\n")
    d = np.zeros((n, 17), dtype=np.float32)
    d[:, 0:3] = means
    d[:, 6:9] = f_dc
    d[:, 9] = opac_logit
    d[:, 10:13] = log_scale
    d[:, 13:17] = q
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(d.tobytes())
    print(f"wrote {path}  ({n} gaussians)", flush=True)


def seg_colors(seg, uniq):
    C = torch.zeros(len(seg), 3)
    for i, s in enumerate(uniq):
        C[seg == int(s)] = torch.tensor(colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.75, 0.95))
    return C


CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/maniskill_fused/pickcube_s1002_train.pt"
c = torch.load(CLIP, map_location="cpu", weights_only=False)
g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
traj = c["traj"]; K = min(16, int(c["Kf"]))
seg = c["seg_per_g"]; uniq = torch.unique(seg).tolist()
os.makedirs("outputs/ply3dgs", exist_ok=True)
tag = os.path.basename(CLIP).replace(".pt", "")
print(f"{tag}: N={len(g0)} entities={uniq} instr='{c['instruction'][:50]}'", flush=True)

# (a) fused canonical scene at t=0 — NATURAL color (inspect reconstruction quality)
write_3dgs_ply(f"outputs/ply3dgs/{tag}_t00_natural.ply", g0.means, g0.scales, g0.quats, g0.opacities, g0.colors)
# (b) same scene in SEGMENTATION color (inspect the per-Gaussian semantic labels in 3D)
write_3dgs_ply(f"outputs/ply3dgs/{tag}_t00_segment.ply", g0.means, g0.scales, g0.quats, g0.opacities,
               seg_colors(seg, uniq))
# (c) analytic-moved natural Gaussians at t=8 and t=K (the GT motion, in 3D)
for t in [8, K]:
    write_3dgs_ply(f"outputs/ply3dgs/{tag}_t{t:02d}_natural.ply", traj[t], g0.scales, g0.quats,
                   g0.opacities, g0.colors)
print("done -> outputs/ply3dgs/", flush=True)
