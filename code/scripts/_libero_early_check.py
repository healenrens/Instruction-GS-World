"""Verify the EARLY window is pre-contact: render frame-0 of _e vs _c for a clip + report the object's
GT 2D centroid vs the gripper(id10) centroid distance at frame 0. Usage: _libero_early_check.py <epi>"""
import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio, numpy as np

epi = sys.argv[1] if len(sys.argv) > 1 else "000000"
for tag in ["e", "c"]:
    p = f"data/libero_pi3_v2/epi{epi}_{tag}_train.pt"
    c = torch.load(p, map_location="cpu", weights_only=False)
    iio.imwrite(f"outputs/review_v8b/frame0_{epi}_{tag}.png", c["gt_rgb"][0].numpy())
    d = (c["traj"][int(c["Kf"])] - c["traj"][0]).norm(dim=-1)
    seg = c["seg_per_g"]
    uv = c["uv"].numpy() / 2.0
    obj = uv[(seg == 1).numpy()].mean(0) if (seg == 1).any() else None
    grip = uv[(seg == 10).numpy()].mean(0) if (seg == 10).any() else None
    dist = float(np.linalg.norm(obj - grip)) if (obj is not None and grip is not None) else -1
    inst = c["instruction"]
    print(f"{tag}-window: movers={int((d > 0.01).sum())} max_disp={float(d.max()):.2f}m  "
          f"frame0 obj-gripper dist={dist:.0f}px  instr={inst!r}")
print("frame0 e vs c saved (EARLY 'e' should have a LARGER obj-gripper distance = pre-contact)")
