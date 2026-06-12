"""§78 R3 m3: open-vocab segmentation on a REAL AgiBot frame (supermarket scene, open vocabulary).
Tests (a) GroundingDINO grounding of the instruction nouns + robot, (b) the full segment_frame_amg
libero-schema id map, on task_327 ep0 frame widx[0]. Saves a 3-panel visual for human audit."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from igsw.data.lerobot_agibot import list_tasks, AgiBotLeRobotTask          # noqa: E402
from openvocab_seg import load_models, _gd_detect, segment_frame_amg        # noqa: E402

TASK, EP, FRAME = "task_327", 0, 187
INSTR = "Place the held cucumber into the plastic bag in the shopping cart."
NOUNS = ["cucumber", "plastic bag", "shopping cart", "robot gripper", "robotic arm"]

root = next(r for r in list_tasks() if r.rstrip("/").endswith(TASK))
t = AgiBotLeRobotTask(root)
rgb = np.asarray(t.decode_frames(EP, "observation.images.head", [FRAME]))[0]   # [H,W,3]
H, W = rgb.shape[:2]
print(f"[ovtest] frame {FRAME}: {W}x{H}")

models = load_models("cuda")
from PIL import Image
pil = Image.fromarray(rgb)
dets = _gd_detect(models, pil, NOUNS, box_thresh=0.18, text_thresh=0.18)
print(f"[ovtest] GroundingDINO detections ({len(dets)}):")
for d in dets[:12]:
    print(f"   {d['label']:<22} score={d['score']:.2f} box={[int(x) for x in d['box']]}")

idm = segment_frame_amg(rgb, INSTR, device="cuda")
ids, cnt = np.unique(idm, return_counts=True)
print(f"[ovtest] segment_frame_amg ids: {dict(zip(ids.tolist(), cnt.tolist()))}")

ID_COL = {0: (0, 0, 0), 1: (0.9, 0.15, 0.15), 2: (0.15, 0.8, 0.3), 3: (0.95, 0.75, 0.1),
          4: (0.8, 0.45, 0.9), 5: (0.95, 0.55, 0.2), 6: (0.4, 0.85, 0.95), 7: (0.85, 0.4, 0.6),
          8: (0.2, 0.45, 0.95), 10: (0.1, 0.85, 0.85)}
seg_img = np.zeros((H, W, 3), np.float32)
for i in ids:
    seg_img[idm == i] = ID_COL.get(int(i), (1, 1, 1))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
axes[0].imshow(rgb); axes[0].axis("off"); axes[0].set_title(f"AgiBot head t{FRAME}\n{INSTR[:46]}", fontsize=8)
axes[1].imshow(rgb)
for d in dets[:12]:
    x0, y0, x1, y1 = d["box"]
    axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="yellow", lw=1.2))
    axes[1].text(x0, y0 - 3, f"{d['label'][:18]} {d['score']:.2f}", fontsize=5.5, color="yellow")
axes[1].axis("off"); axes[1].set_title("GroundingDINO (open vocab)", fontsize=8)
axes[2].imshow(seg_img); axes[2].axis("off")
axes[2].set_title("segment_frame_amg id map\nred=target green=container blue=arm", fontsize=8)
plt.tight_layout()
os.makedirs("viz/agibot", exist_ok=True)
plt.savefig("viz/agibot/r3_m3_openvocab.png", dpi=120, bbox_inches="tight")
print("[ovtest] saved viz/agibot/r3_m3_openvocab.png")
