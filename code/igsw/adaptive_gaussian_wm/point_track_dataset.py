"""Raw-video clips for point-track supervised object-state learning."""

from __future__ import annotations

import torch

from .temporal_object_dataset import TemporalObjectVideoDataset


POINT_TRACK_VIDEO_CONTRACT = "point_track_object_video_v1"


class PointTrackObjectVideoDataset(TemporalObjectVideoDataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        chunk_lengths: str = "8,16,24,32",
        temporal_strides: str = "1,2,3,4",
        max_items: int = 0,
        seed: int = 17,
    ):
        super().__init__(
            cache_root,
            split,
            chunk_lengths,
            temporal_strides,
            0.0,
            max_items,
            seed,
            False,
        )
        self.contract_label = "raw-video-only point-track object-state clips"

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        sample["observation_mask"] = torch.ones(
            int(sample["chunk_length"]), dtype=torch.bool
        )
        return sample
