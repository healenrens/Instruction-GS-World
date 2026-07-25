import sys; sys.path.insert(0, "code")
import torch, numpy as np, imageio.v2 as iio, cv2
from igsw.gaussians import GaussianSet, render_gaussianset
c = torch.load("data/libero_video_v3/epi000000_train.pt", map_location="cuda", weights_only=False)
tr = c["traj"].cuda().float(); K = int(c["Kf"]); seg = c["seg_per_g"].cuda()
N = tr.shape[1]; nf = int(c.get("n_fill", 0))
fill = torch.zeros(N, dtype=torch.bool, device="cuda"); fill[N - nf:] = True
disp = (tr[K] - tr[0]).norm(dim=-1)
sets = [("moved disp>1cm", disp > 0.01), ("static id8", (seg == 8) & (disp <= 0.01)),
        ("fill", fill), ("bg id0", (seg == 0) & ~fill)]
sc = c["scales"].cuda() * 4.0; vm = c["viewmat"][None].cuda().float(); Ki = c["K_intr"][None].cuda().float()
W, H = int(c["W"]), int(c["H"])
rows = []
for name, m in sets:
    if int(m.sum()) < 1:
        im = np.zeros((H, W, 3), np.uint8)
        cv2.putText(im, f"{name} EMPTY", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        rows.append(im)
        continue
    g = GaussianSet(tr[K][m].contiguous(), c["quats"].cuda()[m].contiguous(), sc[m].contiguous(),
                    c["opacities"].cuda()[m].contiguous(), c["colors"].cuda()[m].contiguous(), None)
    col, _, _ = render_gaussianset(g, vm, Ki, W, H)
    im = (col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    cv2.putText(im, f"{name} n={int(m.sum())}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    rows.append(im)
grid = np.concatenate([np.concatenate(rows[:2], 1), np.concatenate(rows[2:], 1)], 0)
iio.imwrite("outputs/review_v3/_isolate_t12.png", grid[::2, ::2])
print("saved OK")
