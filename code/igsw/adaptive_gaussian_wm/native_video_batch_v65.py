"""Pad heterogeneous native-resolution clips without resizing their content."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


def collate_native_video_batch_v65(samples):
    maximum_height = max(sample["video_rgb"].shape[-2] for sample in samples)
    maximum_width = max(sample["video_rgb"].shape[-1] for sample in samples)
    padded = []
    for sample in samples:
        height, width = sample["video_rgb"].shape[-2:]
        pad = (0, maximum_width - width, 0, maximum_height - height)
        item = dict(sample)
        item["video_rgb"] = F.pad(sample["video_rgb"], pad)
        item["video_pixel_valid"] = F.pad(sample["video_pixel_valid"], pad)
        item["native_image_hw"] = torch.tensor((height, width), dtype=torch.long)
        padded.append(item)
    return default_collate(padded)
