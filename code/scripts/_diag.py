import sys; sys.path.insert(0, "code")
import torch, numpy as np
c = torch.load("data/libero_video_v3/epi000000_train.pt", map_location="cuda", weights_only=False)
tr = c["traj"].cuda().float(); K = int(c["Kf"]); seg = c["seg_per_g"].cuda()
Ki = c["K_intr"].cuda().float(); W, H = int(c["W"]), int(c["H"])
P = tr[K]
u = Ki[0, 0] * P[:, 0] / P[:, 2] + Ki[0, 2]
v = Ki[1, 1] * P[:, 1] / P[:, 2] + Ki[1, 2]
# the dark fragment region at t12 (right side, arm height): u in [330, 500], v in [40, 250]
m = (u > 330) & (u < 500) & (v > 40) & (v < 250)
disp = (tr[K] - tr[0]).norm(dim=-1)
m_static = m & (disp < 0.01)
print(f"Gaussians rendering in fragment region at t12: {int(m.sum())}, of which STATIC {int(m_static.sum())}")
segs, cnts = torch.unique(seg[m_static], return_counts=True)
for s, n in zip(segs.tolist(), cnts.tolist()):
    sel = m_static & (seg == s)
    zr = P[sel][:, 2]
    print(f"  seg={s}: n={n}  z[{float(zr.min()):.2f},{float(zr.max()):.2f}]  "
          f"colors mean {c['colors'].cuda()[sel].mean(0).tolist()}")
# where were these Gaussians at frame 0 (2D)?
sel = m_static
u0 = Ki[0, 0] * tr[0][sel][:, 0] / tr[0][sel][:, 2] + Ki[0, 2]
v0 = Ki[1, 1] * tr[0][sel][:, 1] / tr[0][sel][:, 2] + Ki[1, 2]
print(f"frame-0 px of these statics: u[{float(u0.min()):.0f},{float(u0.max()):.0f}] v[{float(v0.min()):.0f},{float(v0.max()):.0f}]")
print(f"their z at frame0: [{float(tr[0][sel][:,2].min()):.2f},{float(tr[0][sel][:,2].max()):.2f}]  scene z range [{float(tr[0][:,2].min()):.2f},{float(tr[0][:,2].max()):.2f}]")
