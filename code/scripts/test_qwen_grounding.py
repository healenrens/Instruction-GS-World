"""Does frozen Qwen's text->image attention give a meaningful, language-sensitive
relevance map? Test on a real frame with its true sub-task vs a different sub-task."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.data import AgiBotLeRobotTask, list_tasks
from igsw.dynamics.conditioning import QwenVLEncoder
from PIL import Image

t = AgiBotLeRobotTask(list_tasks()[0])
ep = 0
enc = QwenVLEncoder(device="cuda") if False else QwenVLEncoder()
enc.model.to("cuda")
dev = "cuda"

def overlay(frame, grid, path):
    g = torch.from_numpy(grid)[None, None].float()
    up = torch.nn.functional.interpolate(g, size=frame.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    ov = frame.astype(np.float32).copy()
    ov[..., 0] = np.clip(ov[..., 0] * (1 - 0.6 * up) + 0.6 * up * 255, 0, 255)  # red = high relevance
    Image.fromarray(ov.astype(np.uint8)).save(path)

os.makedirs("outputs/qwen_grounding", exist_ok=True)
# two frames in two different sub-tasks of the same episode
for f, tag in [(100, "f100"), (300, "f300")]:
    frame = t.decode_frames(ep, "observation.images.head", [f])[0]
    true_sub = t.subtask_text_at(ep, f)
    other_sub = "Place the held object into the plastic bag in the shopping cart." if f < 200 else "Retrieve cucumber from the shelf."
    print(f"\n[{tag}] true subtask: {true_sub!r}")
    maps = {}
    for sub, name in [(true_sub, "true"), (other_sub, "other")]:
        r = enc.encode_grounded(sub, frame, dev)
        rg = r["rel_grid"]
        if rg is None:
            print(f"   ({name}) rel_grid=None  grid={r['grid_hw']}"); continue
        rg_np = rg.float().cpu().numpy()
        maps[name] = rg_np
        # argmax location (which patch is most relevant)
        yi, xi = np.unravel_index(rg_np.argmax(), rg_np.shape)
        print(f"   ({name}) grid={r['grid_hw']} peak@(row{yi},col{xi}) mean={rg_np.mean():.3f} "
              f"frac>0.5={float((rg_np>0.5).mean()):.3f} text_tokens={int(r['text_mask'].sum())}")
        overlay(frame, rg_np, f"outputs/qwen_grounding/{tag}_{name}.png")
    if "true" in maps and "other" in maps:
        a, b = maps["true"], maps["other"]
        l1 = np.abs(a - b).mean()
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"   >>> LANGUAGE-SENSITIVITY true-vs-other: L1diff={l1:.3f}  corr={corr:.3f} "
              f"(low corr / high L1 => language actually moves the grounding)")
print("\n[ok] overlays -> outputs/qwen_grounding/  (red=high relevance; compare true vs other)")
