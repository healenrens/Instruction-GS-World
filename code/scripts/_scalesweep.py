import sys; sys.path.insert(0, "code")
import torch, numpy as np, imageio.v2 as iio, cv2
from igsw.gaussians import GaussianSet, render_gaussianset
c = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
real = (c["gt_rgb"][0].cuda().float() / 255.0)
vm = c["viewmat"][None].cuda().float(); Ki = c["K_intr"][None].cuda().float()
W, H = int(c["W"]), int(c["H"])
def render(mult):
    g = GaussianSet(c["means"].cuda(), c["quats"].cuda(), (c["scales"].cuda() * mult).contiguous(),
                    c["opacities"].cuda(), c["colors"].cuda(), None)
    col, a, _ = render_gaussianset(g, vm, Ki, W, H)
    return col[0].clamp(0, 1), a[0].clamp(0, 1)
panels = [(real, "REAL")]
for m in [1.0, 2.0, 3.0, 4.0]:
    col, a = render(m)
    cov = float((a > 0.5).float().mean())
    l1 = float((col - real).abs().mean())
    p = (col.cpu().numpy() * 255).astype(np.uint8).copy()
    cv2.putText(p, f"x{m:.0f} cov={cov:.2f} L1={l1:.3f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    panels.append((torch.from_numpy(p), f"x{m}"))
    print(f"scale x{m}: coverage {cov:.3f}  L1-vs-REAL {l1:.4f}")
ims = [p[0].cpu().numpy() if isinstance(p[0], torch.Tensor) and p[0].dtype != torch.uint8 else (p[0].cpu().numpy() if isinstance(p[0], torch.Tensor) else p[0]) for p in panels]
ims = [(im * 255).astype(np.uint8) if im.dtype != np.uint8 else im for im in ims]
grid = np.concatenate([np.concatenate(ims[:3], 1), np.concatenate(ims[3:] + [np.zeros_like(ims[0])], 1)], 0)
iio.imwrite("outputs/review_pi3/_scalesweep.png", grid)
print("saved")
