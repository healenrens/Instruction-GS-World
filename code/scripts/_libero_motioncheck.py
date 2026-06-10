import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio, numpy as np
from igsw.gaussians import GaussianSet, render_gaussianset

c = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
tr = c["traj"].cuda().float(); K = int(c["Kf"]); isobj = c["is_obj"].cuda().bool()
disp = (tr[K] - tr[0]).norm(dim=-1)
print(f"N={disp.numel()} n_obj={int(isobj.sum())}  scene_r={float(c.get('scene_r',0)):.3f}")
print(f"ALL disp: max {float(disp.max()):.3f} p99 {float(disp.quantile(0.99)):.3f}  movers(>1cm)={int((disp>0.01).sum())}")
if int(isobj.sum()) > 0:
    od = disp[isobj]
    print(f"OBJ disp: mean {float(od.mean()):.3f} max {float(od.max()):.3f} p50 {float(od.median()):.3f}")

sc = c["scales"].cuda() * 4.0; vm = c["viewmat"][None].cuda().float(); Ki = c["K_intr"][None].cuda().float()
W = int(c["W"]); H = int(c["H"])


def R(mn):
    g = GaussianSet(mn.cuda(), c["quats"].cuda(), sc, c["opacities"].cuda(), c["colors"].cuda(), None)
    col, _, _ = render_gaussianset(g, vm, Ki, W, H)
    return col[0].clamp(0, 1).cpu().numpy()


# rows t=0,6,12: cols  REAL frame_t | GT-motion render
rows = []
for t in [0, 6, K]:
    real = (c["gt_rgb"][t].float() / 255.0).cpu().numpy()
    rows.append(np.concatenate([real, R(tr[t])], 1))
iio.imwrite("outputs/clean/_libero_gtmotion.png", (np.concatenate(rows, 0) * 255).astype(np.uint8))
print("saved _libero_gtmotion.png: rows t0/6/12, cols REAL | GT-motion")
