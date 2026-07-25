"""RGB grid conversion for the modality-matched unstructured baseline."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_grid_shape(batch: dict[str, torch.Tensor]) -> tuple[int, int]:
    grid_hw = batch.get("feature_grid_hw")
    if grid_hw is None or grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
        raise ValueError("feature_grid_hw must have shape [B,2]")
    if not bool((grid_hw == grid_hw[:1]).all()):
        raise ValueError("feature grid dimensions differ within the batch")
    height, width = (int(value) for value in grid_hw[0])
    if height <= 0 or width <= 0:
        raise ValueError("feature grid dimensions must be positive")
    return height, width


def _content_shapes(valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if valid.ndim != 4:
        raise ValueError("RGB valid mask must have shape [B,T,H,W]")
    height = valid.any(dim=-1).sum(dim=-1)
    width = valid.any(dim=-2).sum(dim=-1)
    if not bool(((height > 0) & (width > 0)).all()):
        raise ValueError("RGB valid mask contains an empty frame")
    expected = (
        torch.arange(valid.shape[-2], device=valid.device)[None, None, :, None]
        < height[:, :, None, None]
    ) & (
        torch.arange(valid.shape[-1], device=valid.device)[None, None, None, :]
        < width[:, :, None, None]
    )
    if not bool((valid == expected).all()):
        raise ValueError("RGB valid mask must be a top-left content rectangle")
    return height, width


def rgb_grid_from_frames(
    rgb: torch.Tensor,
    valid: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Crop valid RGB content and resize it to a dense DINO-aligned grid."""
    if rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError("RGB frames must have shape [B,T,3,H,W]")
    if valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
        raise ValueError("RGB frames and valid masks do not align")
    if min(grid_height, grid_width) <= 0:
        raise ValueError("RGB feature grid dimensions must be positive")
    content_height, content_width = _content_shapes(valid)
    batches = []
    for batch_index in range(rgb.shape[0]):
        frames = []
        for time_index in range(rgb.shape[1]):
            height = int(content_height[batch_index, time_index])
            width = int(content_width[batch_index, time_index])
            frame = rgb[batch_index, time_index, :, :height, :width]
            resized = F.interpolate(
                frame[None].float() / 255.0,
                size=(grid_height, grid_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            frames.append(resized.permute(1, 2, 0).reshape(-1, 3))
        batches.append(torch.stack(frames))
    return torch.stack(batches)


def batch_rgb_grids(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    required = (
        "history_rgb",
        "history_rgb_valid",
        "future_rgb",
        "future_rgb_valid",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"matched flat RGB batch is missing {missing}")
    grid_height, grid_width = feature_grid_shape(batch)
    return (
        rgb_grid_from_frames(
            batch["history_rgb"],
            batch["history_rgb_valid"],
            grid_height,
            grid_width,
        ),
        rgb_grid_from_frames(
            batch["future_rgb"],
            batch["future_rgb_valid"],
            grid_height,
            grid_width,
        ),
    )


def render_rgb_grid(
    grid: torch.Tensor,
    valid: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Bilinearly render [B,Q,N,3] predictions into padded RGB images."""
    if grid.ndim != 4 or grid.shape[-1] != 3:
        raise ValueError("RGB grid must have shape [B,Q,N,3]")
    if grid.shape[2] != grid_height * grid_width:
        raise ValueError("RGB token count and feature grid dimensions differ")
    if valid.shape[:2] != grid.shape[:2] or valid.ndim != 4:
        raise ValueError("render RGB valid mask must have shape [B,Q,H,W]")
    content_height, content_width = _content_shapes(valid)
    padded_height, padded_width = valid.shape[-2:]
    batches = []
    for batch_index in range(grid.shape[0]):
        queries = []
        for query_index in range(grid.shape[1]):
            height = int(content_height[batch_index, query_index])
            width = int(content_width[batch_index, query_index])
            source = grid[batch_index, query_index].reshape(
                grid_height,
                grid_width,
                3,
            ).permute(2, 0, 1)
            resized = F.interpolate(
                source[None].float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            queries.append(
                F.pad(
                    resized,
                    (0, padded_width - width, 0, padded_height - height),
                )
            )
        batches.append(torch.stack(queries))
    return torch.stack(batches)
