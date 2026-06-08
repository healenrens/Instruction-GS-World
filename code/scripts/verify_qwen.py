"""Prove the cached lang_hidden really came from Qwen3-VL-2B.

(1) inspect a cached clip's lang_hidden, (2) count the real params + GPU mem of the
loaded VLM, (3) re-decode the exact same frame + instruction, re-encode, and show
the re-encoded hidden matches the cached one (determinism => cache is genuine).
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.data import AgiBotLeRobotTask, list_tasks  # noqa: E402


def main():
    clip_path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob("/mnt/pfs/public/xuhaoming/instruct_gs_world/data/clips_v1/*.pt"))[0]
    rec = torch.load(clip_path, map_location="cpu", weights_only=False)
    print(f"[clip] {os.path.basename(clip_path)}")
    print(f"[clip] keys = {list(rec.keys())}")
    lh = rec["lang_hidden"]
    print(f"[clip] instruction = {rec['instruction']!r}")
    print(f"[clip] cached lang_hidden: shape={tuple(lh.shape)} dtype={lh.dtype} "
          f"mean={lh.float().mean():.4f} std={lh.float().std():.4f} "
          f"nonzero_frac={(lh!=0).float().mean():.3f}")
    print(f"[clip] cached lang_mask sum = {int(rec['lang_mask'].sum())}  task={rec['task']} ep={rec['ep']} f0={rec['f0']}")

    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated() / 1e9
    from igsw.dynamics.conditioning import QwenVLEncoder
    enc = QwenVLEncoder(device="cuda")
    n_tot = sum(p.numel() for p in enc.model.parameters())
    m1 = torch.cuda.memory_allocated() / 1e9
    print(f"\n[VLM] class={type(enc.model).__name__} hidden={enc.hidden_size}")
    print(f"[VLM] PARAMETERS = {n_tot/1e9:.3f} B   (weights occupy {m1-m0:.2f} GB on GPU)")
    sd_bytes = sum(p.numel()*p.element_size() for p in enc.model.parameters())
    print(f"[VLM] weight bytes = {sd_bytes/1e9:.2f} GB ({enc.model.dtype})")

    # re-decode the EXACT frame used at caching time and re-encode
    root = next(t for t in list_tasks() if os.path.basename(os.path.dirname(t)) == rec["task"])
    task = AgiBotLeRobotTask(root)
    frame0 = task.decode_frames(rec["ep"], "observation.images.head", [rec["f0"]])[0]
    h, m = enc.encode(rec["instruction"], image=frame0)
    h = h.cpu()
    print(f"\n[re-encode] shape={tuple(h.shape)} (cached {tuple(lh.shape)})")
    if h.shape == lh.shape:
        diff = (h.float() - lh.float()).abs()
        cos = torch.nn.functional.cosine_similarity(h.float().flatten(), lh.float().flatten(), dim=0)
        print(f"[re-encode] max|Δ|={diff.max():.4f} mean|Δ|={diff.mean():.5f} cosine={cos:.6f}")
        print("[verdict] MATCH => cached hidden genuinely produced by this 2B VLM"
              if cos > 0.999 else "[verdict] mismatch (investigate)")
    else:
        print("[verdict] shape differs (frame/proc mismatch) — compare stats instead")


if __name__ == "__main__":
    main()
