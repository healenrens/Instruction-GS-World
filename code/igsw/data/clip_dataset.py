"""Dataset of cached clips (see scripts/cache_clips.py).

Each item is one lifted clip: dense canonical Gaussians G0, per-frame cameras,
GT frames, and frozen Qwen3-VL hidden states. Gaussian counts vary per clip, so
training uses batch_size=1 per GPU (DDP gives the effective batch). Tensors are
returned on CPU; the trainer moves them to the GPU.
"""

from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import Dataset

from ..gaussians.types import GaussianSet


class ClipDataset(Dataset):
    def __init__(self, root: str, require_lang: bool = True):
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no .pt clips under {root}")
        self.require_lang = require_lang

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rec = torch.load(self.files[i], map_location="cpu", weights_only=False)
        g = rec["g0"]
        g0 = GaussianSet(
            means=g["means"].float(), quats=g["quats"].float(), scales=g["scales"].float(),
            opacities=g["opacities"].float(), colors=g["colors"].float(), features=None,
        )
        gt = rec["gt"].float() / 255.0           # [K+1,H,W,3]
        out = {
            "g0": g0, "gt": gt, "Ks": rec["Ks"].float(), "viewmats": rec["viewmats"].float(),
            "H": rec["H"], "W": rec["W"], "K": rec["K"], "stride": rec["stride"],
            "instruction": rec["instruction"], "path": self.files[i],
        }
        if "lang_hidden" in rec:
            out["lang_hidden"] = rec["lang_hidden"]   # [L,2048] bf16
            out["lang_mask"] = rec["lang_mask"].bool()
        elif self.require_lang:
            raise KeyError(f"clip {self.files[i]} has no cached lang_hidden")
        return out


def identity_collate(batch):
    """batch_size==1 passthrough (variable-N dense sets can't be stacked)."""
    assert len(batch) == 1
    return batch[0]
