import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio
from igsw.gaussians import GaussianSet, render_gaussianset

c = torch.load("data/libero_video/epi000400_heldtask.pt", map_location="cuda", weights_only=False)
mn = c["means"].float(); uv = c["uv"].float()
u, v = uv[:, 0], uv[:, 1]; z = mn[:, 2]
H = int(c["H"]); W = int(c["W"])
f = 618.0; cx = W / 2.0; cy = H / 2.0
vm = torch.eye(4, device="cuda")[None]
K = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], device="cuda").float()[None]
real = c["gt_rgb"][0].cuda().float() / 255.0


def render(xn, yn):
    pts = torch.stack([xn, yn, z], 1)
    g = GaussianSet(pts, c["quats"], c["scales"], c["opacities"], c["colors"], None)
    col, _, _ = render_gaussianset(g, vm, K, W, H)
    return col[0].clamp(0, 1)


# variant A: uv as (u=col, v=row)
A = render((u - cx) * z / f, (v - cy) * z / f)
# variant B: swapped (in case uv is stored (row,col))
B = render((v - cx) * z / f, (u - cy) * z / f)
grid = torch.cat([torch.cat([real, A], 1), torch.cat([real, B], 1)], 0)
iio.imwrite("outputs/clean/_libero_rebuild.png", (grid * 255).to(torch.uint8).cpu().numpy())
print("saved _libero_rebuild.png  TOP: REAL | rebuild(u,v)   BOT: REAL | rebuild(swap)")
