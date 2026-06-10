"""Isolate-render a clip's traj[t] by entity subset to attribute visual artifacts.
Usage: _libero_isobjcheck2.py <clip.pt> [t]"""
import sys; sys.path.insert(0, "code")
import torch, numpy as np, imageio.v2 as iio, cv2
from igsw.gaussians import GaussianSet, render_gaussianset

c = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
t = int(sys.argv[2]) if len(sys.argv) > 2 else int(c["Kf"])
tr = c["traj"].cuda().float(); seg = c["seg_per_g"].cuda().long()
N = tr.shape[1]; nf = int(c.get("n_fill", 0))
fill = torch.zeros(N, dtype=torch.bool, device="cuda"); fill[N - nf:] = True
disp = (tr[int(c["Kf"])] - tr[0]).norm(dim=-1)
subs = [("object id1", (seg == 1) & ~fill), ("arm id8+sub", ((seg == 8) | (seg >= 50)) & ~fill),
        ("gripper id10", (seg == 10) & ~fill), ("fill", fill),
        ("static others", (seg != 1) & (seg != 8) & (seg != 10) & (seg < 50) & ~fill),
        ("ALL", torch.ones(N, dtype=torch.bool, device="cuda"))]
sc = c["scales"].cuda() * 4.0; vm = c["viewmat"][None].cuda().float(); Ki = c["K_intr"][None].cuda().float()
W, H = int(c["W"]), int(c["H"])
rows = []
for name, m in subs:
    if int(m.sum()) < 1:
        im = np.zeros((H, W, 3), np.uint8)
    else:
        g = GaussianSet(tr[t][m].contiguous(), c["quats"].cuda()[m].contiguous(), sc[m].contiguous(),
                        c["opacities"].cuda()[m].contiguous(), c["colors"].cuda()[m].contiguous(), None)
        col, _, _ = render_gaussianset(g, vm, Ki, W, H)
        im = (col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    # how many of this subset are "high" (above table) AND moved a lot — candidate fliers
    flier = int((m & (disp > 0.05) & (tr[t][:, 1] < tr[0][:, 1].quantile(0.2))).sum())
    cv2.putText(im, f"{name} n={int(m.sum())} hi-mov={flier}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    rows.append(im)
grid = np.concatenate([np.concatenate(rows[:3], 1), np.concatenate(rows[3:], 1)], 0)
iio.imwrite("outputs/review_pi3/_isolate.png", grid[::2, ::2])
# also: count Gaussians whose t-position is implausibly far from g0 (teleport survivors)
far = (tr[t] - tr[0]).norm(dim=-1)
print(f"t={t}: max disp {float(far.max()):.2f}m  >0.5m: {int((far>0.5).sum())}  >0.7m: {int((far>0.7).sum())}")
for name, m in subs[:5]:
    fm = far[m]
    if int(m.sum()) and int((fm > 0.5).sum()):
        print(f"  {name}: {int((fm>0.5).sum())} pts >0.5m (max {float(fm.max()):.2f}m)")
print("saved _isolate.png")
