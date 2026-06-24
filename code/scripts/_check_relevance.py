"""RUNTIME CHECK for the relevance-placement fix: is QwenVLEncoder.relevance_grid (instruction<->image-patch
cosine on the frozen fast path) informative + language-sensitive on RoboTwin2 frame-0? If near-uniform or
instruction-insensitive, relevance placement is no better than pure entropy and we should not switch to it."""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2
from igsw.dynamics.conditioning import QwenVLEncoder


def mv_in(inputs, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def main():
    dev = "cuda"
    enc = QwenVLEncoder(attn_impl="eager").to(dev).eval()        # eager so encode_grounded attentions work
    clips = sorted(glob.glob("data/rt2_joint/*_train.pt"))[:6]
    os.makedirs("logs/relcheck", exist_ok=True)
    WRONG = "a scenic photo of snowy mountains under a clear blue sky"
    print("idx | instr | ATTN-ROLLOUT std (uniform~0) | real_vs_wrong_L1 (lang-sens) | corr (<1 good)", flush=True)
    for i, cp in enumerate(clips):
        c = torch.load(cp, map_location=dev, weights_only=False)
        rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
        instr = c.get("instruction", "")
        gr = enc.encode_grounded(instr, rgb0, dev, grounding=True)
        gw = enc.encode_grounded(WRONG, rgb0, dev, grounding=True)
        rel_r, ghw = gr["rel_grid"], gr["grid_hw"]
        rel_w = gw["rel_grid"]
        if rel_r is None:
            print(f"{i} | {instr[:40]!r} | rel_grid=None (rollout failed)", flush=True); continue
        r = rel_r.float().cpu().numpy()
        w = rel_w.float().cpu().numpy() if rel_w is not None else None
        std = float(r.std())
        l1 = float(np.abs(r - w).mean()) if w is not None else -1.0
        corr = float(np.corrcoef(r.flatten(), w.flatten())[0, 1]) if (w is not None and w.shape == r.shape) else -1.0
        print(f"{i} | {instr[:40]!r} | std={std:.3f} | real_vs_wrong_L1={l1:.3f} | corr={corr:+.2f}  ghw={ghw}", flush=True)
        rr = cv2.resize(r, (rgb0.shape[1], rgb0.shape[0]))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(rgb0); ax[0].set_title(instr[:55], fontsize=8); ax[0].axis("off")
        ax[1].imshow(rgb0); ax[1].imshow(rr, alpha=0.55, cmap="jet")
        ax[1].set_title("relevance (real instr)", fontsize=8); ax[1].axis("off")
        plt.tight_layout(); plt.savefig(f"logs/relcheck/rel_{i}.png", dpi=85); plt.close()
    print("saved overlays -> logs/relcheck/rel_*.png", flush=True)


if __name__ == "__main__":
    main()
