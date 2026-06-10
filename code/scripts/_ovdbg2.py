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
print("epi", epi, "instr:", instr, "| noun:", noun)

ys, xs = np.where(msk[0] == 1)
gt_cx, gt_cy = xs.mean(), ys.mean()
print("GT id1 centroid cx=%.0f cy=%.0f bbox=[%d,%d,%d,%d]" % (gt_cx, gt_cy, xs.min(), ys.min(), xs.max(), ys.max()))

# strategy A: query the SINGLE noun alone, take highest-score box
print("--- A: single noun alone ---")
dets = ov._gd_detect(m, pil, [noun], box_thresh=0.10, text_thresh=0.10)
dets = sorted(dets, key=lambda d: -d["score"])
for d in dets[:6]:
    b = d["box"]; bcx = (b[0]+b[2])/2; bcy=(b[1]+b[3])/2
    hit = "<<HIT" if abs(bcx-gt_cx)<20 and abs(bcy-gt_cy)<20 else ""
    print("  score=%.3f cx=%.0f cy=%.0f %s" % (d["score"], bcx, bcy, hit))

# strategy B: query ALL nouns at once, for the target noun take the box whose LABEL contains the noun
# AND has the highest score among label-exact matches
print("--- B: all nouns at once, label contains noun, top by score ---")
dets = ov._gd_detect(m, pil, ov.LIBERO_NOUNS, box_thresh=0.12, text_thresh=0.10)
cand = [d for d in dets if noun in d["label"]]
cand = sorted(cand, key=lambda d: -d["score"])
for d in cand[:8]:
    b = d["box"]; bcx = (b[0]+b[2])/2; bcy=(b[1]+b[3])/2
    hit = "<<HIT" if abs(bcx-gt_cx)<20 and abs(bcy-gt_cy)<20 else ""
    print("  score=%.3f cx=%.0f cy=%.0f label=%r %s" % (d["score"], bcx, bcy, d["label"], hit))

# strategy C: query noun EXCLUSIVELY in label (label == noun exactly), all-nouns prompt
print("--- C: all nouns at once, label EXACTLY == noun ---")
cand = [d for d in dets if d["label"].strip() == noun]
cand = sorted(cand, key=lambda d: -d["score"])
for d in cand[:8]:
    b = d["box"]; bcx = (b[0]+b[2])/2; bcy=(b[1]+b[3])/2
    hit = "<<HIT" if abs(bcx-gt_cx)<20 and abs(bcy-gt_cy)<20 else ""
    print("  score=%.3f cx=%.0f cy=%.0f %s" % (d["score"], bcx, bcy, hit))
