import sys; sys.path.insert(0, "code")
import numpy as np
from scripts.video_gt import load_episode_full, pick_window

epi = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rgb, msk, ooi, instr, n = load_episode_full(epi)
print(f"epi {epi}: n={n} frames  instr={instr!r}")
cs = []
for i in range(n):
    oo = ooi[i] > 0
    cs.append(np.argwhere(oo)[:, [1, 0]].mean(0) if oo.sum() > 0 else np.array([np.nan, np.nan]))
cs = np.array(cs)
valid = ~np.isnan(cs[:, 0])
sz = np.array([(ooi[i] > 0).sum() for i in range(n)])
print(f"obj visible {int(valid.sum())}/{n} frames;  mask size min={sz.min()} max={sz.max()} f0={sz[0]}")
print(f"centroid 2D span: x[{np.nanmin(cs[:,0]):.0f},{np.nanmax(cs[:,0]):.0f}] y[{np.nanmin(cs[:,1]):.0f},{np.nanmax(cs[:,1]):.0f}]  (256px img)")
print(f"total 2D path len = {np.nansum(np.linalg.norm(np.diff(cs,axis=0),axis=1)):.0f}px")
w = pick_window(ooi, n, 48)
print(f"pick_window(win=48) -> frames {w[0]}..{w[-1]}")
# per-window object 2D displacement (start->end) for the chosen window
c0, c1 = cs[w[0]], cs[w[-1]]
print(f"  chosen window obj centroid: start {c0} end {c1}  disp={np.linalg.norm(c1-c0):.1f}px")
# where is the BIGGEST start->end displacement window?
best, bi = -1, 0
for s in range(0, max(1, n - 48)):
    e = min(s + 48, n) - 1
    if valid[s] and valid[e]:
        d = np.linalg.norm(cs[e] - cs[s])
        if d > best:
            best, bi = d, s
print(f"MAX start->end disp window: frames {bi}..{min(bi+48,n)-1}  disp={best:.1f}px")
