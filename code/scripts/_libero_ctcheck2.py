import sys; sys.path.insert(0, "code")
import numpy as np, torch, cv2
from scripts.video_gt import load_episode_full, pick_window, cotracker_grid
from cotracker.predictor import CoTrackerPredictor

rgb, msk, ooi, instr, n = load_episode_full(400)
widx = pick_window(ooi, n, 48)
rgb_win = rgb[widx]
oo0 = (ooi[widx[0]] > 0).astype(np.uint8)
nL, lab, stats, cent = cv2.connectedComponentsWithStats(oo0, 8)
areas = stats[1:, cv2.CC_STAT_AREA]
big = 1 + int(np.argmax(areas))
clean = (lab == big).astype(np.uint8)
clean = cv2.erode(clean, np.ones((3, 3), np.uint8))
print(f"raw mask {int(oo0.sum())}px  n_CC={nL-1}  CC areas(top5)={sorted(areas)[::-1][:5]}  largest-eroded={int(clean.sum())}px")
ys, xs = np.where(clean > 0)
sel = np.random.RandomState(0).choice(len(xs), min(500, len(xs)), replace=False)
q_xy = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
ct = CoTrackerPredictor(checkpoint="checkpoints/cotracker/scaled_offline.pth", v2=False, offline=True).cuda()
tr, vis = cotracker_grid(ct, rgb_win, q_xy, device="cuda")
disp = np.linalg.norm(tr[-1] - tr[0], axis=1)
print(f"CLEANED-CC disp(px): mean {disp.mean():.1f} median {np.median(disp):.1f} max {disp.max():.1f}")
print(f"  frac static(<2px) {float((disp<2).mean()):.2f}  moving(>10px) {float((disp>10).mean()):.2f}")
print(f"  centroid disp {np.linalg.norm(tr[-1].mean(0)-tr[0].mean(0)):.1f}px")
