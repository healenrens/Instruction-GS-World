import sys; sys.path.insert(0, "code")
import numpy as np, torch
from scripts.video_gt import load_episode_full, pick_window, cotracker_grid
from cotracker.predictor import CoTrackerPredictor

rgb, msk, ooi, instr, n = load_episode_full(400)
widx = pick_window(ooi, n, 48)
rgb_win = rgb[widx]                                            # [48,256,256,3]
oo0 = ooi[widx[0]] > 0
ys, xs = np.where(oo0)
sel = np.random.RandomState(0).choice(len(xs), min(500, len(xs)), replace=False)
q_xy = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)     # [Q,2] source px
print(f"window {widx[0]}..{widx[-1]}  Q={len(q_xy)} object query px")
ct = CoTrackerPredictor(checkpoint="checkpoints/cotracker/scaled_offline.pth", v2=False, offline=True).cuda()
tr, vis = cotracker_grid(ct, rgb_win, q_xy, device="cuda")    # [T,Q,2],[T,Q]
disp = np.linalg.norm(tr[-1] - tr[0], axis=1)
print(f"CoTracker track disp (px): mean {disp.mean():.1f}  median {np.median(disp):.1f}  max {disp.max():.1f}")
print(f"vis frac @last {float((vis[-1] > 0.5).mean()):.2f}   tracks shape {tr.shape}")
print(f"sample q[0]={q_xy[0]}  track[0]={tr[0,0]}  track[-1]={tr[-1,0]}")
print(f"centroid: f0 {tr[0].mean(0)}  f-1 {tr[-1].mean(0)}  disp {np.linalg.norm(tr[-1].mean(0)-tr[0].mean(0)):.1f}px")
