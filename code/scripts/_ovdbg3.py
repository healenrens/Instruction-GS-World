import sys, numpy as np
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
from video_gt import load_episode_full
import openvocab_seg as ov
from PIL import Image

NOUNS = ov.LIBERO_NOUNS


def cluster_boxes(dets, iou_thr=0.4):
    """Greedy spatial clustering of detection boxes into distinct object locations.
    Returns list of clusters; each cluster = dict(box=mean_box, members=[dets])."""
    def iou(a, b):
        x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
        iw = max(0., x1-x0); ih = max(0., y1-y0); inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0.
    clusters = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        placed = False
        for c in clusters:
            if iou(d["box"], c["box"]) > iou_thr:
                c["members"].append(d); placed = True; break
        if not placed:
            clusters.append(dict(box=d["box"], members=[d]))
    return clusters


epi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rgb, msk, ooi, instr, n = load_episode_full(epi)
m = ov.load_models()
pil = Image.fromarray(rgb[0])
target = ov.parse_noun(instr)
ys, xs = np.where(msk[0] == 1)
gt_cx, gt_cy = xs.mean(), ys.mean()
print("epi %d target=%r GT id1 cx=%.0f cy=%.0f" % (epi, target, gt_cx, gt_cy))

# For each noun separately, get detections, tag with the noun. Pool all.
all_dets = []
for nn in NOUNS:
    for d in ov._gd_detect(m, pil, [nn], box_thresh=0.12, text_thresh=0.10):
        # restrict to small object-sized boxes (drop scene-wide)
        b = d["box"]; area = (b[2]-b[0])*(b[3]-b[1])
        if area > 0.25*256*256:
            continue
        d["noun"] = nn
        all_dets.append(d)

clusters = cluster_boxes(all_dets, iou_thr=0.3)
# for each cluster, aggregate per-noun max score; the cluster's "identity" = argmax noun
print("clusters (cx,cy : top-noun score : target-noun score):")
best_target = None
for c in clusters:
    b = c["box"]; cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
    per_noun = {}
    for d in c["members"]:
        per_noun[d["noun"]] = max(per_noun.get(d["noun"], 0), d["score"])
    top_noun = max(per_noun, key=per_noun.get)
    tscore = per_noun.get(target, 0.0)
    hit = "<<GThit" if abs(cx-gt_cx) < 22 and abs(cy-gt_cy) < 22 else ""
    nmemb = len(c["members"])
    # "target relative score" = target score minus mean of other nouns at this cluster
    others = [v for k, v in per_noun.items() if k != target]
    rel = tscore - (np.mean(others) if others else 0)
    print("  cx=%-3.0f cy=%-3.0f n=%-2d top=%-16s top_s=%.3f  tgt_s=%.3f rel=%+.3f %s"
          % (cx, cy, nmemb, top_noun, per_noun[top_noun], tscore, rel, hit))

# selection rule candidates:
# rule1: cluster where target-noun absolute score is highest
r1 = max(clusters, key=lambda c: max([d["score"] for d in c["members"] if d["noun"] == target] or [0]))
# rule2: cluster where target-noun is the argmax noun AND target score highest among those
tgt_clusters = []
for c in clusters:
    per = {}
    for d in c["members"]:
        per[d["noun"]] = max(per.get(d["noun"], 0), d["score"])
    if per and max(per, key=per.get) == target:
        tgt_clusters.append((per[target], c))
r2 = max(tgt_clusters, key=lambda x: x[0])[1] if tgt_clusters else None


def show(name, c):
    if c is None:
        print("  %s: NONE" % name); return
    b = c["box"]; cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
    ok = abs(cx-gt_cx) < 22 and abs(cy-gt_cy) < 22
    print("  %s -> cx=%.0f cy=%.0f  %s" % (name, cx, cy, "CORRECT" if ok else "WRONG"))


print("SELECTION:")
show("rule1 (max target abs score)", r1)
show("rule2 (target is argmax noun)", r2)
