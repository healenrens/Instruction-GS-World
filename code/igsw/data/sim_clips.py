"""SimClipDataset — loads the CLEAN ManiSkill sim clips (maniskill_gt.py / gen_sim_dataset.py)
for the GENERALIZATION-phase trainer (agent.md §39 scale-up).

Each `.pt` clip holds a frame-0 dense GaussianSet (means/quats/scales/opacities/colors), each
Gaussian's frame-0 pixel `uv` + actor id `seg_per_g`, the EXACT analytic per-Gaussian world
trajectory `traj` [Kf+1,N,3] (the clean GT), the static camera (`K_intr`,`viewmat`,`H`,`W`), the
`instruction`, and the GT-self-check `val_psnr`. NO tracking/lifting needed at train time — the clip
is already the supervised target.

The dataset returns ONE clip dict per item (batch_size=1, variable N) with everything the trainer
needs: the dense g0 tensors (raw), the per-Gaussian uv + seg + full traj. The trainer samples the M
controls (mover-biased) and slices the GT trajectory itself (so the control sampling can be reseeded
per epoch). This is a MAP-style dataset over the train split; with DDP the DistributedSampler shards
clips across ranks. Held-out clips are loaded by a separate split filter for eval only.
"""
from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import Dataset


def list_clips(root: str, splits=("train",)):
    """All clip paths in `root` whose filename split-suffix is in `splits`.
    Filenames are {env}_s{seed}_{split}.pt (gen_sim_dataset). The 2 original hand clips
    (pickcube_s0.pt / stackcube_s1.pt) have NO split suffix -> treated as 'train'."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*.pt"))):
        name = os.path.basename(p)[:-3]
        parts = name.rsplit("_", 1)
        split = parts[1] if len(parts) == 2 and parts[1] in ("train", "heldseed", "heldtask") else "train"
        if split in splits:
            out.append(p)
    return out


class SimClipDataset(Dataset):
    """Map-style dataset over a fixed list of sim clip files (one split). Returns the raw clip dict
    on CPU; the trainer moves to GPU + samples controls. Clips are loaded lazily (mmap-friendly)."""

    def __init__(self, root: str, splits=("train",), max_clips: int = 0):
        self.paths = list_clips(root, splits)
        if max_clips > 0:
            self.paths = self.paths[:max_clips]
        if not self.paths:
            raise RuntimeError(f"no sim clips with splits={splits} under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        c = torch.load(p, map_location="cpu", weights_only=False)
        c["path"] = p
        return c


def sim_collate(batch):
    """batch_size==1 passthrough (variable-N dense sets cannot be stacked)."""
    assert len(batch) == 1
    return batch[0]
