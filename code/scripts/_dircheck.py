import sys; sys.path.insert(0, "code")
import torch
c = torch.load("data/libero_video/_test400.pt", map_location="cpu", weights_only=False)
tr = c["traj"].float(); K = int(c["Kf"]); Ki = c["K_intr"]
f = float(Ki[0, 0]); cx = float(Ki[0, 2]); cy = float(Ki[1, 2])
disp = (tr[K] - tr[0]).norm(dim=-1); mv = disp > 0.01
def proj(p): return torch.stack([f * p[:, 0] / p[:, 2] + cx, f * p[:, 1] / p[:, 2] + cy], 1)
c0 = proj(tr[0][mv]).mean(0); cK = proj(tr[K][mv]).mean(0)
print(f"n_movers={int(mv.sum())}")
print(f"mover 2D centroid (src px): t0={(c0/2).tolist()}  tK={(cK/2).tolist()}")
print(f"shift (src px) dx={float((cK[0]-c0[0])/2):+.1f} dy={float((cK[1]-c0[1])/2):+.1f}   REAL salad: dx=-31 dy=-12 (left+up)")
print(f"3D mover disp: mean={float(disp[mv].mean()):.3f} max={float(disp[mv].max()):.3f} median={float(disp[mv].median()):.3f}")
