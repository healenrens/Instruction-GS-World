import sys; sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
import numpy as np
from PIL import Image
from video_gt import load_episode_full
from openvocab_seg import load_models, _amg_masks, _gd_detect, _device
rgb, msk, ooi, instr, n = load_episode_full(0)
gt = msk[0]
pil = Image.fromarray(rgb[0])
models = load_models(_device())
# 1) what does GroundingDINO see for robot/basket?
dets = _gd_detect(models, pil, ["robotic arm", "robot", "basket"], box_thresh=0.18, text_thresh=0.18)
print("GD dets:", [(d["label"], round(d["score"], 2), [int(x) for x in d["box"]]) for d in dets][:6])
# 2) what masks does AMG keep?
masks = _amg_masks(models, pil)
print("AMG kept %d masks" % len(masks))
for i, m in enumerate(masks):
    a = int(m.sum()); ys, xs = np.where(m); cx, cy = int(xs.mean()), int(ys.mean())
    # which GT id does this mask mostly land on?
    gt_here = gt[m]
    vals, cnts = np.unique(gt_here, return_counts=True)
    top = vals[cnts.argmax()]
    print("  mask%d: area=%d centroid=(%d,%d) mostly-GT-id=%d (%.0f%%)" % (i, a, cx, cy, top, 100 * cnts.max() / a))
# 3) GT object sizes for reference
print("GT entities:", {int(i): int((gt == i).sum()) for i in np.unique(gt)})
