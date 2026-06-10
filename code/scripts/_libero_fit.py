import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio
from igsw.gaussians import GaussianSet, render_gaussianset

c = torch.load("data/libero_video/epi000400_heldtask.pt", map_location="cuda", weights_only=False)
mn = c["means"].float(); uv = c["uv"].float()
x, y, z = mn[:, 0], mn[:, 1], mn[:, 2]
u, v = uv[:, 0], uv[:, 1]
xz = (x / z); yz = (y / z)
ones = torch.ones_like(xz)


def fit(t, target):
    A = torch.stack([t, ones], 1)
    sol = torch.linalg.lstsq(A, target[:, None]).solution[:, 0]
    pred = A @ sol
    res = (pred - target).abs()
    return float(sol[0]), float(sol[1]), float(res.median()), float(res.quantile(0.9))


a, b, ru, ru9 = fit(xz, u)
cc, d, rv, rv9 = fit(yz, v)
print(f"u = {a:.1f}*(x/z) + {b:.1f}   resid med {ru:.1f}px p90 {ru9:.1f}px   -> fx={a:.1f} cx={b:.1f}")
print(f"v = {cc:.1f}*(y/z) + {d:.1f}   resid med {rv:.1f}px p90 {rv9:.1f}px   -> fy={cc:.1f} cy={d:.1f}")
print(f"sign(fx)={'+' if a>0 else '-'}  sign(fy)={'+' if cc>0 else '-'}  (negative => that axis is flipped vs gsplat OpenCV)")

# corrected render: flip the axis whose fitted focal is negative, use |focal|
H = int(c["H"]); W = int(c["W"])
mnc = mn.clone()
if a < 0: mnc[:, 0] = -mnc[:, 0]
if cc < 0: mnc[:, 1] = -mnc[:, 1]
K = torch.tensor([[abs(a), 0, b], [0, abs(cc), d], [0, 0, 1]], device="cuda").float()[None]
vm = torch.eye(4, device="cuda")[None]
g = GaussianSet(mnc, c["quats"], c["scales"], c["opacities"], c["colors"], None)
col, _, _ = render_gaussianset(g, vm, K, W, H)
real = c["gt_rgb"][0].cuda().float() / 255.0
side = torch.cat([real, col[0].clamp(0, 1)], 1)
iio.imwrite("outputs/clean/_libero_fitcam.png", (side * 255).to(torch.uint8).cpu().numpy())
print("saved _libero_fitcam.png: left=REAL  right=g0 rendered with FITTED camera")
