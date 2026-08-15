"""Qualitative assignment exports for v48 held-video evaluation."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


_PALETTE = (
    (0.90, 0.22, 0.21),
    (0.20, 0.55, 0.91),
    (0.18, 0.72, 0.42),
    (0.96, 0.66, 0.16),
    (0.58, 0.35, 0.82),
    (0.10, 0.70, 0.74),
    (0.91, 0.36, 0.65),
    (0.55, 0.49, 0.42),
    (0.35, 0.74, 0.91),
    (0.72, 0.27, 0.32),
    (0.42, 0.68, 0.26),
    (0.96, 0.49, 0.12),
    (0.45, 0.42, 0.78),
    (0.18, 0.63, 0.58),
    (0.80, 0.47, 0.70),
    (0.62, 0.62, 0.62),
)


def save_assignment_visualizations(
    video_rgb: torch.Tensor,
    assignment: torch.Tensor,
    grid_hw: tuple[int, int],
    output_dir: str,
    chunk_length: int,
    first_item_index: int,
    maximum_items: int,
) -> list[str]:
    if video_rgb.ndim != 5 or assignment.ndim != 4:
        raise ValueError("v48 visualization tensors have invalid ranks")
    if assignment.shape[:2] != video_rgb.shape[:2]:
        raise ValueError("v48 visualization batch or time dimensions differ")
    if assignment.shape[2] != grid_hw[0] * grid_hw[1]:
        raise ValueError("v48 assignment count differs from the DINO grid")
    if assignment.shape[-1] > len(_PALETTE):
        raise ValueError("v48 visualization palette is smaller than the slot count")
    os.makedirs(output_dir, exist_ok=True)
    palette = assignment.new_tensor(_PALETTE).float()
    created = []
    count = min(len(video_rgb), max(0, maximum_items - first_item_index))
    for batch_index in range(count):
        rgb = video_rgb[batch_index, -1].float() / 255.0
        rgb = F.interpolate(
            rgb[None], size=(224, 224), mode="bilinear", align_corners=False
        )[0]
        final_assignment = assignment[batch_index, -1].float()
        confidence, owner = final_assignment.max(dim=-1)
        color = palette[owner].reshape(*grid_hw, 3).permute(2, 0, 1)
        color = F.interpolate(color[None], size=(224, 224), mode="nearest")[0]
        confidence = confidence.reshape(1, *grid_hw)
        confidence = F.interpolate(
            confidence[None], size=(224, 224), mode="bilinear", align_corners=False
        )[0]
        alpha = 0.25 + 0.45 * confidence.clamp(0.0, 1.0)
        overlay = rgb * (1.0 - alpha) + color * alpha
        panel = torch.cat((rgb, overlay, color), dim=2).clamp(0.0, 1.0)
        item_index = first_item_index + batch_index
        path = os.path.join(
            output_dir,
            f"assignment_h{chunk_length}_item{item_index:04d}.png",
        )
        save_image(panel, path)
        created.append(os.path.abspath(path))
    return created
