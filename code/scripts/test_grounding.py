"""Validate the v7-min role-mask grounder (GroundingDINO->SAM2) on a real AgiBot frame."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.data import AgiBotLeRobotTask, list_tasks
from igsw.grounding import RoleGrounder

t = AgiBotLeRobotTask(list_tasks()[0])
ep, f = 0, 600
print("instruction:", t.language(ep))
frame = t.decode_frames(ep, "observation.images.head", [f])[0]   # [H,W,3] uint8
print("frame", frame.shape)

g = RoleGrounder(device="cuda")
roles = {"hand": "robot gripper", "object": "fruit", "container": "basket box", "shelf": "shelf rack"}
print("=== detections ===")
for label, box, score in g.detect(frame, list(dict.fromkeys(roles.values()))):
    print(f"  {label:18s} score={score:.3f} box={[round(x) for x in box]}")
masks = g.ground_roles(frame, roles)
print("=== role mask coverage (frac of pixels) ===")
for r, m in masks.items():
    print(f"  {r:10s} {m.mean():.4f}")

# save an overlay for visual sanity
os.makedirs("outputs/grounding", exist_ok=True)
from PIL import Image
ov = frame.astype(np.float32).copy()
colors = {"hand": [255, 0, 0], "object": [0, 255, 0], "container": [0, 0, 255], "shelf": [255, 255, 0]}
for r, m in masks.items():
    c = np.array(colors.get(r, [255, 255, 255]), np.float32)
    ov = ov * (1 - 0.5 * m[..., None]) + 0.5 * m[..., None] * c
Image.fromarray(ov.clip(0, 255).astype(np.uint8)).save("outputs/grounding/roles_ep0_f600.png")
print("[ok] overlay -> outputs/grounding/roles_ep0_f600.png")
