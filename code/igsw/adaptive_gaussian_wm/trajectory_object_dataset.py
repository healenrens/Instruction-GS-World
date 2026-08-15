"""Raw-video chunks with deterministic contiguous observation blackouts."""

from __future__ import annotations

import torch

from .temporal_object_dataset import TemporalObjectVideoDataset, _stable_integer


TRAJECTORY_OBJECT_VIDEO_CONTRACT = "trajectory_object_video_v1"


class TrajectoryObjectVideoDataset(TemporalObjectVideoDataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        chunk_lengths: str = "8,16,24,32",
        temporal_strides: str = "1,2,3,4",
        blackout_fraction: float = 0.25,
        max_items: int = 0,
        seed: int = 17,
        record_manifest_hash: bool = True,
    ):
        if not 0.0 < blackout_fraction <= 0.5:
            raise ValueError("v49 blackout fraction must be in (0,0.5]")
        self.blackout_fraction = float(blackout_fraction)
        super().__init__(
            cache_root,
            split,
            chunk_lengths,
            temporal_strides,
            0.0,
            max_items,
            seed,
            record_manifest_hash,
        )
        self.contract_label = "raw-video-only trajectory object-state chunks"

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        frames = int(sample["chunk_length"])
        blackout = max(1, min(frames - 2, round(frames * self.blackout_fraction)))
        available = frames - blackout - 1
        start = 1 + _stable_integer(
            "v49-blackout",
            self.seed,
            int(sample["sequence_index"]),
            int(sample["control_indices"][0]),
            frames,
        ) % available
        observed = torch.ones(frames, dtype=torch.bool)
        observed[start : start + blackout] = False
        sample["observation_mask"] = observed
        sample["blackout_start"] = torch.tensor(start, dtype=torch.long)
        sample["blackout_length"] = torch.tensor(blackout, dtype=torch.long)
        return sample

