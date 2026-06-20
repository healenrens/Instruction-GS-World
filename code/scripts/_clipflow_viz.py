"""Flow viz from a clip .pt: project traj[0]->traj[Kf] to image, draw top-120 mover arrows (same style
as robotwin_spatrack_gt) for GT-quality comparison. Usage: _clipflow_viz.py <clip.pt> <out.png> [label]"""
import sys, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm.tokens import project_to_uv  # noqa: E402

clip, out = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else "Pi3+CoTracker"
c = torch.load(clip, weights_only=False)
K, vm, tr, Kf = c["K_intr"], c["viewmat"], c["traj"], c["Kf"]
rgb = c["gt_rgb"][0].numpy()
u0 = project_to_uv(tr[0], K, vm).numpy(); uK = project_to_uv(tr[Kf], K, vm).numpy()
d2 = np.linalg.norm(uK - u0, axis=1)
fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(rgb)
for i in np.argsort(-d2)[:120]:
    if d2[i] < 1: continue
    ax.annotate("", xy=(uK[i, 0], uK[i, 1]), xytext=(u0[i, 0], u0[i, 1]),
                arrowprops=dict(arrowstyle="->", color=plt.cm.turbo(min(d2[i] / 80, 1)), lw=0.9, alpha=0.8))
ax.set_title(f"{label}: {os.path.basename(out)} top-120 movers, med2d {np.median(d2):.0f}px", fontsize=9)
ax.axis("off")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"saved {out} (N={len(d2)})")
