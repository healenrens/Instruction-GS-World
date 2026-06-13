"""Fast CPU smoke for the §93 GPSToken control-selection path (no model load). Validates
gpstoken_ctrl_idx end-to-end on one clip and reports effective-M + on-mover fraction."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gaussians.gpstoken import gpstoken_ctrl_idx, mover_saliency  # noqa: E402

c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
uv = c["uv"]
tr = c["traj"]
disp = (tr[-1] - tr[0]).norm(dim=-1)
H, W = int(c["H"]), int(c["W"])
N = uv.shape[0]
n_keep = N - int(c.get("n_fill", 0))
rgb0 = c["gt_rgb"][0].numpy()  # uint8 [H,W,3]
print(f"clip {os.path.basename(sys.argv[1])} N={N} n_keep={n_keep} H={H} W={W}")
for beta in [0.0, 30.0]:
    sal = mover_saliency(uv, disp, n_keep, H, W) if beta > 0 else None
    idx = gpstoken_ctrl_idx(rgb0, uv, n_keep, 256, "cpu", sal=sal, beta=beta)
    onmover = (disp[idx] > 0.01).float().mean().item()
    rand_onmover = (disp[:n_keep] > 0.01).float().mean().item()
    print(f"  beta={beta:4.0f}: effM={idx.numel():3d}  on-mover={onmover*100:.1f}%  (random {rand_onmover*100:.1f}%)  conc={onmover/max(rand_onmover,1e-6):.2f}x")
print("SMOKE OK")
