"""Render the TRAINING-DATA review video for a clip: per frame t=0..Kf two columns
   [ REAL video frame | GT-motion render (the supervision target traj[t], x4 splat scales) ].
This is the DATA itself — no model. Usage: _libero_data_video.py <clip.pt> <out.mp4>"""
import sys; sys.path.insert(0, "code")
import torch, numpy as np, imageio.v2 as iio, cv2
from igsw.gaussians import GaussianSet, render_gaussianset

clip_p, out_p = sys.argv[1], sys.argv[2]
c = torch.load(clip_p, map_location="cuda", weights_only=False)
tr = c["traj"].cuda().float(); K = int(c["Kf"])
sc = c["scales"] * (float(sys.argv[3]) if len(sys.argv) > 3 else 1.5); vm = c["viewmat"][None].float(); Ki = c["K_intr"][None].float()
W = int(c["W"]); H = int(c["H"])
disp = (tr[K] - tr[0]).norm(dim=-1)
print(f"{clip_p}: N={tr.shape[1]} movers={int((disp>0.01).sum())} max_disp={float(disp.max()):.2f}m  instr={c['instruction']!r}")


def R(mn):
    g = GaussianSet(mn, c["quats"], sc, c["opacities"], c["colors"], None)
    col, _, _ = render_gaussianset(g, vm, Ki, W, H)
    return (col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def label(im, txt):
    im = im.copy(); cv2.putText(im, txt, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return im


frames = []
for t in range(K + 1):
    real = c["gt_rgb"][t].cpu().numpy()
    row = np.concatenate([label(real, f"REAL t={t}"), label(R(tr[t]), "TRAIN DATA (GT traj render)")], 1)
    frames.append(row)
frames = [frames[0]] * 2 + frames + [frames[-1]] * 3
iio.mimwrite(out_p, frames, fps=2, quality=8)
print(f"saved {out_p}")
