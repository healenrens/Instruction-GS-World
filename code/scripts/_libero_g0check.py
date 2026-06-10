import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio
from igsw.gaussians import GaussianSet, render_gaussianset

c = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
real = c["gt_rgb"][0].cuda().float() / 255.0
sc = c["scales"] * (float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
g = GaussianSet(c["means"].cuda(), c["quats"].cuda(), sc.cuda(), c["opacities"].cuda(), c["colors"].cuda(), None)
col, _, _ = render_gaussianset(g, c["viewmat"][None].cuda().float(), c["K_intr"][None].cuda().float(), int(c["W"]), int(c["H"]))
side = torch.cat([real, col[0].clamp(0, 1)], 1)
iio.imwrite("outputs/clean/_libero_g0check.png", (side * 255).to(torch.uint8).cpu().numpy())
mn = c["means"]
print(f"focal={float(c['K_intr'][0,0]):.1f}  N={len(mn)}  "
      f"x[{mn[:,0].min():.2f},{mn[:,0].max():.2f}] y[{mn[:,1].min():.2f},{mn[:,1].max():.2f}] z[{mn[:,2].min():.2f},{mn[:,2].max():.2f}]")
print("saved _libero_g0check.png: left=REAL  right=g0 render")
