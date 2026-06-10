"""Open-vocabulary segmentation (GroundingDINO + SAM2) producing an id map that matches the LIBERO
`msk` schema (`binhng/libero_object_lerobot_mask_depth`, 256x256 uint8 id map). This lets the data
pipeline (video_gt.py / pi3_video_gt.py) drop the GT-mask dependency: it consumes the SAME id layout.

SCHEMA (exact — keep so the data scripts are unchanged):
  0  = background (table / floor / walls — everything not below)
  1  = the MANIPULATED object = the instruction's noun  ("pick up the {noun} and place it in the basket")
  2  = basket (the place target)
  3..7 = the OTHER table objects (distractors) — ids by left-to-right centroid-x (deterministic)
  8  = robot arm body
  10 = robot gripper / end-effector
  (id 9 unused)

PIPELINE (per frame):
  GroundingDINO (IDEA-Research/grounding-dino-tiny, zero-shot detection) with text phrases
      -> boxes + scores per phrase
  SAM2 (facebook/sam2.1-hiera-large) prompted with those boxes -> per-box masks
  assemble the id map by the schema:
      named noun  -> 1
      "basket"    -> 2
      "robot arm" -> 8
      "robot gripper" -> 10
      remaining detected table objects -> 3..7 by centroid-x
  overlaps resolved by a fixed paint ORDER (background distractors first, then specific entities on top,
  gripper last so it wins over the arm it sits inside).

API (verified, transformers 5.10.2, both weights in hf_cache, OFFLINE):
  from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor   # GroundingDINO
  from transformers import Sam2Model, Sam2Processor                            # SAM2

Importable: `from openvocab_seg import segment_frame`.
CLI: `--epi N` loads the episode (via video_gt.load_episode_full), segments frame-0, saves an id map +
colorized overlay to outputs/openvocab/epiN.png, and prints per-entity IoU vs GT msk[0]."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
GD_REPO = os.environ.get("GD_REPO", "IDEA-Research/grounding-dino-tiny")  # base = stronger (env override)
SAM2_REPO = "facebook/sam2.1-hiera-large"

# The 10 LIBERO-object nouns (candidate distractors / target objects).
LIBERO_NOUNS = [
    "alphabet soup", "bbq sauce", "butter", "chocolate pudding", "cream cheese",
    "ketchup", "milk", "orange juice", "salad dressing", "tomato sauce",
]

# Schema ids for the fixed entities.
ID_BG = 0
ID_OBJ = 1
ID_BASKET = 2
ID_DISTRACTOR_LO = 3
ID_DISTRACTOR_HI = 7
ID_ARM = 8
ID_GRIPPER = 10

# A colormap for the overlay (id -> RGB). ids 3..7 share a ramp.
_OVERLAY_COLORS = {
    0: (0, 0, 0),
    1: (255, 60, 60),      # named object — red
    2: (60, 120, 255),     # basket — blue
    3: (255, 200, 0),      # distractors — yellows/greens
    4: (0, 200, 120),
    5: (200, 0, 200),
    6: (0, 200, 200),
    7: (160, 100, 40),
    8: (255, 140, 0),      # arm — orange
    10: (0, 255, 0),       # gripper — green
}


# --------------------------------------------------------------------------- #
# model loading (cached module-global so repeated calls are cheap)
# --------------------------------------------------------------------------- #
_MODELS = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(device: str | None = None):
    """Load (and cache) GroundingDINO + SAM2. Returns a dict of (gd_proc, gd_model, sam_proc, sam_model)."""
    device = device or _device()
    if _MODELS.get("device") == device and "gd_model" in _MODELS:
        return _MODELS
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                              Sam2Model, Sam2Processor)
    dt = torch.float32
    gd_proc = AutoProcessor.from_pretrained(GD_REPO)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_REPO, torch_dtype=dt).to(device).eval()
    sam_proc = Sam2Processor.from_pretrained(SAM2_REPO)
    sam_model = Sam2Model.from_pretrained(SAM2_REPO, torch_dtype=dt).to(device).eval()
    _MODELS.update(device=device, gd_proc=gd_proc, gd_model=gd_model,
                   sam_proc=sam_proc, sam_model=sam_model)
    return _MODELS


# --------------------------------------------------------------------------- #
# instruction parsing
# --------------------------------------------------------------------------- #
def parse_noun(instruction: str) -> str:
    """Extract the manipulated-object noun from 'pick up the {noun} and place it in the basket'.
    Falls back to matching any LIBERO noun substring, else returns '' (caller treats as unknown)."""
    instr = (instruction or "").lower().strip()
    # exact LIBERO-object substring match is the most robust (the dataset uses these verbatim)
    for noun in LIBERO_NOUNS:
        if noun in instr:
            return noun
    # generic "pick up the X and place ..." parse
    import re
    m = re.search(r"pick up the (.+?) and place", instr)
    if m:
        return m.group(1).strip()
    m = re.search(r"the (.+?) (?:and|in|on|into)\b", instr)
    if m:
        return m.group(1).strip()
    return ""


# --------------------------------------------------------------------------- #
# GroundingDINO detection
# --------------------------------------------------------------------------- #
def _gd_detect(models, pil_img, phrases, box_thresh=0.25, text_thresh=0.20):
    """Run GroundingDINO on one image with a list of text `phrases`. GroundingDINO wants a single text
    prompt of lowercase phrases separated by ' . ' and ending in ' .'. Returns list of dicts:
    {box:[x0,y0,x1,y1], score:float, label:str} in pixel coords."""
    device = models["device"]
    gd_proc, gd_model = models["gd_proc"], models["gd_model"]
    text = ". ".join(p.lower() for p in phrases) + "."
    inputs = gd_proc(images=pil_img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    W, H = pil_img.size
    results = gd_proc.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=box_thresh, text_threshold=text_thresh,
        target_sizes=[(H, W)])[0]
    dets = []
    labels = results.get("text_labels", results.get("labels"))
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        dets.append(dict(box=[float(v) for v in box.tolist()],
                         score=float(score), label=str(label)))
    return dets


def _phrase_matches(det_label: str, phrase: str) -> bool:
    """GroundingDINO may return a label that is a sub/super-string of the queried phrase
    (it tokenizes and can merge/split). Treat as a match if either contains the other's words."""
    a = set(det_label.lower().split())
    b = set(phrase.lower().split())
    if not a or not b:
        return False
    return bool(a & b) and (a <= b or b <= a or len(a & b) >= 1)


# --------------------------------------------------------------------------- #
# SAM2 box -> mask
# --------------------------------------------------------------------------- #
def _sam2_masks(models, pil_img, boxes):
    """boxes: list of [x0,y0,x1,y1] pixel coords. Returns boolean masks [K,H,W] (one per box).
    Empty boxes -> returns empty array."""
    if len(boxes) == 0:
        W, H = pil_img.size
        return np.zeros((0, H, W), dtype=bool)
    device = models["device"]
    sam_proc, sam_model = models["sam_proc"], models["sam_model"]
    # Sam2Processor input_boxes format: list over images -> list of boxes -> [x0,y0,x1,y1]
    inputs = sam_proc(images=pil_img, input_boxes=[[list(map(float, b)) for b in boxes]],
                      return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)
    masks = sam_proc.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], binarize=True)[0]  # [K,1,H,W] or [K,H,W]
    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    return masks.bool().cpu().numpy()


# --------------------------------------------------------------------------- #
# the main entry point
# --------------------------------------------------------------------------- #
def segment_frame(rgb_uint8, instruction, distractor_vocab=None,
                  device: str | None = None,
                  box_thresh: float = 0.25, text_thresh: float = 0.20,
                  named_box_thresh: float = 0.15):
    """Segment one RGB frame into the LIBERO `msk` id map.

    Args
      rgb_uint8 : [H,W,3] uint8 image (LIBERO is 256x256).
      instruction : the task string; the noun is parsed as the manipulated (id-1) object.
      distractor_vocab : optional list of candidate distractor nouns; defaults to LIBERO_NOUNS.
      box_thresh / text_thresh : GroundingDINO thresholds for the general phrases.
      named_box_thresh : a LOWER threshold for the named object + basket (small objects need it).

    Returns
      id_map : [H,W] uint8 matching the schema.
    """
    device = device or _device()
    models = load_models(device)
    H, W = rgb_uint8.shape[:2]
    pil = Image.fromarray(rgb_uint8.astype(np.uint8))

    noun = parse_noun(instruction)
    distractor_vocab = list(distractor_vocab) if distractor_vocab is not None else list(LIBERO_NOUNS)

    # ---- 1) detect the SPECIFIC fixed entities (named object, basket, arm, gripper) ----
    # Run these at a LOWER threshold; the named LIBERO objects are small (~100-800 px).
    fixed_phrases = []
    if noun:
        fixed_phrases.append(noun)
    fixed_phrases += ["basket", "robot arm", "robot gripper"]
    fixed_dets = _gd_detect(models, pil, fixed_phrases,
                            box_thresh=named_box_thresh, text_thresh=text_thresh)

    # ---- 2) detect the table objects (distractors) with the full LIBERO noun vocab ----
    # query every candidate noun (minus the named one is fine to keep — we de-dup spatially later)
    # cast a wider net: the specific LIBERO nouns often miss (GroundingDINO can't tell "alphabet
    # soup" from "cream cheese"), so also query GENERIC SHAPES at a low threshold — every tabletop
    # object IS a bottle/can/box. NMS + size-filtering below dedup. (Identity-by-noun isn't needed:
    # the data pipeline picks the manipulated object by MOTION, not by which noun matched.)
    GENERIC = ["bottle", "can", "box", "carton", "jar", "container"]
    distr_dets = _gd_detect(models, pil, list(distractor_vocab) + GENERIC,
                            box_thresh=0.15, text_thresh=0.15)

    # ---- helper: pick the single best box matching a phrase from a det list ----
    def best_box(dets, phrase):
        cands = [d for d in dets if _phrase_matches(d["label"], phrase)]
        if not cands:
            return None
        return max(cands, key=lambda d: d["score"])

    # named object box (try fixed first; fall back to the distractor pass which uses the same noun)
    obj_det = best_box(fixed_dets, noun) if noun else None
    if obj_det is None and noun:
        obj_det = best_box(distr_dets, noun)
    basket_det = best_box(fixed_dets, "basket")
    arm_det = best_box(fixed_dets, "robot arm")
    grip_det = best_box(fixed_dets, "robot gripper")

    # ---- 3) collect distractor boxes: every LIBERO-noun detection EXCEPT the named object ----
    # de-dup overlapping boxes of the same/different noun by greedy NMS, keep highest score.
    def iou_box(a, b):
        x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
        iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    distr_cands = sorted(distr_dets, key=lambda d: -d["score"])
    distractor_boxes = []
    obj_box = obj_det["box"] if obj_det else None
    for d in distr_cands:
        b = d["box"]
        # skip if it IS the named object box (large overlap with it)
        if obj_box is not None and iou_box(b, obj_box) > 0.5:
            continue
        # NMS against already-kept distractors
        if any(iou_box(b, kb) > 0.5 for kb in distractor_boxes):
            continue
        # skip huge boxes (likely the whole scene / table) — > 60% of frame area
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area > 0.6 * W * H:
            continue
        distractor_boxes.append(b)

    # ---- 4) SAM2: turn every kept box into a mask, in ONE batched pass per category group ----
    # Order matters for the final paint; we run all boxes together then index back.
    box_list = []
    box_kind = []   # ('obj',), ('basket',), ('arm',), ('grip',), ('distr', k)
    if obj_det is not None:
        box_list.append(obj_det["box"]); box_kind.append(("obj",))
    if basket_det is not None:
        box_list.append(basket_det["box"]); box_kind.append(("basket",))
    if arm_det is not None:
        box_list.append(arm_det["box"]); box_kind.append(("arm",))
    if grip_det is not None:
        box_list.append(grip_det["box"]); box_kind.append(("grip",))
    for k, b in enumerate(distractor_boxes):
        box_list.append(b); box_kind.append(("distr", k))

    masks = _sam2_masks(models, pil, box_list)  # [K,H,W] bool

    mask_of = {}
    distr_masks = []
    for kind, m in zip(box_kind, masks):
        if kind[0] == "distr":
            distr_masks.append((kind[1], m))
        else:
            mask_of[kind[0]] = m

    # ---- 5) assemble the id map by the schema, with a fixed paint order ----
    id_map = np.zeros((H, W), dtype=np.uint8)

    # 5a) distractors first (lowest priority), id by left-to-right centroid x.
    distr_entries = []
    for _, m in distr_masks:
        if m.sum() < 8:
            continue
        ys, xs = np.where(m)
        cx = float(xs.mean())
        distr_entries.append((cx, m))
    distr_entries.sort(key=lambda e: e[0])  # left -> right
    next_id = ID_DISTRACTOR_LO
    for cx, m in distr_entries:
        if next_id > ID_DISTRACTOR_HI:
            break
        id_map[m] = next_id
        next_id += 1

    # 5b) basket on top of distractors (it is a clear specific target).
    if "basket" in mask_of and mask_of["basket"].sum() >= 8:
        id_map[mask_of["basket"]] = ID_BASKET

    # 5c) named object on top (the most important — it must win over any distractor box it overlaps).
    if "obj" in mask_of and mask_of["obj"].sum() >= 4:
        id_map[mask_of["obj"]] = ID_OBJ

    # 5d) robot: GroundingDINO usually LUMPS the whole robot into ONE box (the "robot arm" phrase
    # rarely fires separately from "robot gripper") -> the data pipeline only needs the robot as ONE
    # entity it can motion-cluster (§50 arm-clustering splits the articulated parts). So: if BOTH arm
    # and gripper fired distinctly, keep id8/id10 separate; otherwise assign the single robot mask to
    # id8 (arm) and leave id10 empty (the pipeline's id8 branch handles articulation).
    arm_m = mask_of.get("arm"); grip_m = mask_of.get("grip")
    arm_ok = arm_m is not None and arm_m.sum() >= 8
    grip_ok = grip_m is not None and grip_m.sum() >= 8
    # treat gripper as DISTINCT only if it is much smaller than the arm box (a real end-effector,
    # not the whole-robot lump mislabelled "gripper")
    distinct = arm_ok and grip_ok and grip_m.sum() < 0.6 * arm_m.sum()
    if arm_ok:
        id_map[arm_m] = ID_ARM
    if grip_ok:
        id_map[grip_m] = ID_GRIPPER if distinct else ID_ARM   # lumped robot -> id8
    return id_map


# --------------------------------------------------------------------------- #
# overlay + IoU helpers (for the CLI / validation)
# --------------------------------------------------------------------------- #
def colorize(id_map):
    """id map -> RGB uint8 using _OVERLAY_COLORS."""
    H, W = id_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for i in np.unique(id_map):
        out[id_map == i] = _OVERLAY_COLORS.get(int(i), (128, 128, 128))
    return out


def overlay_on_rgb(rgb_uint8, id_map, alpha=0.55):
    """Blend the colorized id map over the rgb (background id 0 stays as rgb)."""
    col = colorize(id_map).astype(np.float32)
    base = rgb_uint8.astype(np.float32)
    fg = id_map > 0
    out = base.copy()
    out[fg] = (1 - alpha) * base[fg] + alpha * col[fg]
    return out.clip(0, 255).astype(np.uint8)


def iou(a_mask, b_mask):
    inter = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()
    return float(inter) / float(union) if union > 0 else (1.0 if inter == 0 else 0.0)


def per_entity_iou(pred_id_map, gt_id_map):
    """Per-id IoU for the schema's fixed entities. Returns dict {entity_name: iou}."""
    out = {}
    for name, idv in [("obj(1)", ID_OBJ), ("basket(2)", ID_BASKET),
                      ("arm(8)", ID_ARM), ("grip(10)", ID_GRIPPER)]:
        out[name] = iou(pred_id_map == idv, gt_id_map == idv)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/openvocab")
    ap.add_argument("--box_thresh", type=float, default=0.25)
    ap.add_argument("--text_thresh", type=float, default=0.20)
    ap.add_argument("--named_box_thresh", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "code")
    sys.path.insert(0, "code/scripts")
    from video_gt import load_episode_full

    rgb, msk, ooi, instruction, n = load_episode_full(args.epi)
    frame0 = rgb[0]
    gt = msk[0]
    print(f"[openvocab] epi={args.epi} instruction={instruction!r} noun={parse_noun(instruction)!r}",
          flush=True)

    id_map = segment_frame(frame0, instruction, device=args.device,
                           box_thresh=args.box_thresh, text_thresh=args.text_thresh,
                           named_box_thresh=args.named_box_thresh)

    # IoU report
    ious = per_entity_iou(id_map, gt)
    # most important: does the predicted id-1 region overlap the GT id-1 region at all?
    pred1 = id_map == ID_OBJ
    gt1 = gt == ID_OBJ
    overlap_px = int(np.logical_and(pred1, gt1).sum())
    overlaps_gt1 = overlap_px > 0
    print(f"[openvocab] per-entity IoU: " +
          " ".join(f"{k}={v:.3f}" for k, v in ious.items()), flush=True)
    print(f"[openvocab] id-1 pred px={int(pred1.sum())} gt px={int(gt1.sum())} "
          f"overlap_px={overlap_px} named_overlaps_GT1={overlaps_gt1}", flush=True)

    # save overlay + id map
    os.makedirs(args.out_dir, exist_ok=True)
    import imageio.v3 as iio
    ov = overlay_on_rgb(frame0, id_map)
    gt_ov = overlay_on_rgb(frame0, gt)
    # side by side: rgb | pred overlay | gt overlay
    panel = np.concatenate([frame0, ov, gt_ov], axis=1)
    out_png = os.path.join(args.out_dir, f"epi{args.epi}.png")
    iio.imwrite(out_png, panel)
    np.save(os.path.join(args.out_dir, f"epi{args.epi}_idmap.npy"), id_map)
    print(f"[openvocab] saved overlay (rgb|pred|gt) -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
