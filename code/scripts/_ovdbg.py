import sys, numpy as np
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
from video_gt import load_episode_full
import openvocab_seg as ov
from PIL import Image

epi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rgb, msk, ooi, instr, n = load_episode_full(epi)
m = ov.load_models()
pil = Image.fromarray(rgb[0])
noun = ov.parse_noun(instr)
print("instr:", instr, "| noun:", noun)

print("=== fixed phrases @ thr 0.08 ===")
for d in ov._gd_detect(m, pil, [noun, "basket", "robot arm", "robot gripper"], box_thresh=0.08, text_thresh=0.08):
    b = d["box"]
    print("  label=%-28r score=%.3f box=[%.0f,%.0f,%.0f,%.0f]" % (d["label"], d["score"], b[0], b[1], b[2], b[3]))

print("=== alt phrases for arm/gripper @ thr 0.08 ===")
for d in ov._gd_detect(m, pil, ["robotic arm", "gripper", "robot hand", "metal gripper claw", "robot"], box_thresh=0.08, text_thresh=0.08):
    b = d["box"]
    print("  label=%-28r score=%.3f box=[%.0f,%.0f,%.0f,%.0f]" % (d["label"], d["score"], b[0], b[1], b[2], b[3]))

print("=== full LIBERO noun vocab @ thr 0.15 ===")
for d in ov._gd_detect(m, pil, ov.LIBERO_NOUNS, box_thresh=0.15, text_thresh=0.12):
    b = d["box"]
    print("  label=%-28r score=%.3f box=[%.0f,%.0f,%.0f,%.0f]" % (d["label"], d["score"], b[0], b[1], b[2], b[3]))

print("=== GT centroids ===")
for idv in [1, 2, 8, 10]:
    ys, xs = np.where(msk[0] == idv)
    if len(xs):
        print("  id%-2d: cx=%.0f cy=%.0f bbox=[%d,%d,%d,%d] px=%d" % (
            idv, xs.mean(), ys.mean(), xs.min(), ys.min(), xs.max(), ys.max(), len(xs)))
