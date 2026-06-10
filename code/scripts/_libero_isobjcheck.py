import sys; sys.path.insert(0, "code")
import torch, numpy as np
from scripts.video_gt import load_episode_full

c = torch.load("data/libero_video/_test400.pt", map_location="cpu", weights_only=False)
uv = c["uv"].numpy(); isobj = c["is_obj"].numpy().astype(bool)
W = int(c["W"]); H = int(c["W"])
uvs = uv / 2.0                                                  # model px -> source px (s_to_model=2)
print(f"ALL uv_src: x[{uvs[:,0].min():.0f},{uvs[:,0].max():.0f}] y[{uvs[:,1].min():.0f},{uvs[:,1].max():.0f}]")
uo = uvs[isobj]
print(f"is_obj N={int(isobj.sum())}  uv_src centroid=({uo[:,0].mean():.0f},{uo[:,0].mean():.0f}) "
      f"x[{uo[:,0].min():.0f},{uo[:,0].max():.0f}] y[{uo[:,1].min():.0f},{uo[:,1].max():.0f}]")
rgb, msk, ooi, instr, n = load_episode_full(400)
f0 = c["widx"][0] if "widx" in c else 65
oo = ooi[f0] > 0
ys, xs = np.where(oo)
print(f"ooi frame {f0}: centroid=({xs.mean():.0f},{ys.mean():.0f}) x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
# the moving salad cluster lives near the SMALLER mask component; basket is the bottom-left blob
import cv2
nL, lab, stats, cent = cv2.connectedComponentsWithStats(oo.astype(np.uint8), 8)
for k in range(1, nL):
    cx, cy = cent[k]; area = stats[k, cv2.CC_STAT_AREA]
    nobj = int(((np.abs(uo[:, 0] - cx) < 18) & (np.abs(uo[:, 1] - cy) < 18)).sum())
    print(f"  CC{k}: area={area} centroid=({cx:.0f},{cy:.0f})  is_obj Gaussians near it={nobj}")
