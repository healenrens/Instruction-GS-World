import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio, numpy as np
from igsw.gaussians import GaussianSet, render_gaussianset

c = torch.load("data/libero_video/epi000400_heldtask.pt", map_location="cuda", weights_only=False)
mn = c["means"]; uv = c["uv"]
print(f"N={len(mn)}  means x[{mn[:,0].min():.2f},{mn[:,0].max():.2f}] "
      f"y[{mn[:,1].min():.2f},{mn[:,1].max():.2f}] z[{mn[:,2].min():.2f},{mn[:,2].max():.2f}]")
print(f"uv  u[{uv[:,0].min():.0f},{uv[:,0].max():.0f}] v[{uv[:,1].min():.0f},{uv[:,1].max():.0f}]  (image is {int(c['W'])}x{int(c['H'])})")
print(f"stored focal={float(c['K_intr'][0,0]):.1f}")

real = c["gt_rgb"][0].cuda().float() / 255.0
H = int(c["H"]); W = int(c["W"])
vm = torch.eye(4, device="cuda")[None]


def R(fx):
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], device="cuda").float()[None]
    g = GaussianSet(mn, c["quats"], c["scales"], c["opacities"], c["colors"], None)
    col, _, _ = render_gaussianset(g, vm, K, W, H)
    return col[0].clamp(0, 1)


focals = [200, 300, 450, float(c["K_intr"][0, 0])]
top = torch.cat([real, R(focals[0])], 1)
bot = torch.cat([R(focals[1]), R(focals[2])], 1)
grid = torch.cat([top, bot], 0)
iio.imwrite("outputs/clean/_libero_focal.png", (grid * 255).to(torch.uint8).cpu().numpy())
print(f"saved _libero_focal.png: TL=REAL  TR=fx{focals[0]}  BL=fx{focals[1]}  BR=fx{focals[2]:.0f}")
