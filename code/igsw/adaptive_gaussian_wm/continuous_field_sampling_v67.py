"""Native-resolution continuous coordinate and scale sampling for v67."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NativeContinuousCropsV67:
    rgb: torch.Tensor
    valid_fraction: torch.Tensor
    effective_scales: torch.Tensor


def stratified_query_coordinates_v67(
    sequence_index: torch.Tensor,
    side: int,
    jitter_fraction: float,
) -> torch.Tensor:
    """Return deterministic per-video quadrature points in normalized coordinates."""
    device = sequence_index.device
    cell = torch.arange(side, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(cell, cell, indexing="ij")
    point = torch.arange(side * side, device=device, dtype=torch.float32)[None]
    sequence = sequence_index.float()[:, None]
    jitter_x = torch.frac(torch.sin(sequence * 12.9898 + point * 78.233) * 43758.5453)
    jitter_y = torch.frac(torch.sin(sequence * 39.3467 + point * 11.135) * 24634.6345)
    jitter_x = (jitter_x * 2.0 - 1.0) * jitter_fraction
    jitter_y = (jitter_y * 2.0 - 1.0) * jitter_fraction
    x = x.reshape(1, -1) + 0.5 + jitter_x
    y = y.reshape(1, -1) + 0.5 + jitter_y
    return torch.stack((x / side * 2.0 - 1.0, y / side * 2.0 - 1.0), dim=-1)


def evenly_spaced_query_indices_v67(
    candidate_count: int,
    query_count: int,
    device: torch.device,
) -> torch.Tensor:
    stride = candidate_count / query_count
    index = (torch.arange(query_count, device=device).float() + 0.5) * stride
    return index.floor().long().clamp_max(candidate_count - 1)


def context_coordinate_mask_v67(
    candidate_count: int,
    context_fraction: float,
    sequence_index: torch.Tensor,
) -> torch.Tensor:
    count = max(1, min(candidate_count - 1, round(candidate_count * context_fraction)))
    point = torch.arange(candidate_count, device=sequence_index.device)[None]
    offset = sequence_index.long()[:, None].remainder(candidate_count)
    permutation_key = (point * 131 + offset * 197).remainder(candidate_count)
    order = permutation_key.argsort(dim=-1)
    mask = torch.zeros_like(order, dtype=torch.bool)
    mask.scatter_(1, order[:, :count], True)
    return mask


def _native_to_padded_grid(
    coordinates: torch.Tensor,
    native_hw: torch.Tensor,
    padded_hw: tuple[int, int],
) -> torch.Tensor:
    height = native_hw[:, 0].float().clamp_min(2.0)
    width = native_hw[:, 1].float().clamp_min(2.0)
    padded_height, padded_width = padded_hw
    pixel = coordinates.float().clone()
    pixel[..., 0] = (pixel[..., 0] + 1.0) * 0.5 * (width - 1.0)[:, None, None]
    pixel[..., 1] = (pixel[..., 1] + 1.0) * 0.5 * (height - 1.0)[:, None, None]
    pixel[..., 0] = pixel[..., 0] / max(padded_width - 1, 1) * 2.0 - 1.0
    pixel[..., 1] = pixel[..., 1] / max(padded_height - 1, 1) * 2.0 - 1.0
    return pixel


def sample_native_multiscale_crops_v67(
    video_rgb: torch.Tensor,
    video_pixel_valid: torch.Tensor,
    native_hw: torch.Tensor,
    coordinates: torch.Tensor,
    scales: torch.Tensor,
    multipliers: tuple[float, ...],
    crop_side: int,
) -> NativeContinuousCropsV67:
    """Sample differentiable local RGB observations without resizing the frame."""
    if video_rgb.ndim != 5 or video_rgb.shape[2] != 3:
        raise ValueError("v67 RGB must have shape [B,T,3,H,W]")
    batch, frames, _, height, width = video_rgb.shape
    if coordinates.shape[:2] != (batch, frames) or coordinates.shape[-1] != 2:
        raise ValueError("v67 coordinates must have shape [B,T,P,2]")
    if scales.shape != coordinates.shape[:-1]:
        raise ValueError("v67 scales must have shape [B,T,P]")
    points = coordinates.shape[2]
    base_grid = _native_to_padded_grid(coordinates, native_hw, (height, width))
    short_side = native_hw.amin(dim=-1).float().clamp_min(2.0)
    axis = torch.linspace(-1.0, 1.0, crop_side, device=video_rgb.device)
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack((dx, dy), dim=-1)
    flat_rgb = video_rgb.reshape(batch * frames, 3, height, width).float() / 255.0
    flat_valid = video_pixel_valid.reshape(batch * frames, 1, height, width).float()
    crops, validity, effective = [], [], []
    for multiplier in multipliers:
        scale = scales.float() * float(multiplier)
        radius_px = 0.5 * scale * short_side[:, None, None]
        offset = unit[None, None, None] * radius_px[..., None, None, None]
        offset_x = offset[..., 0] / max(width - 1, 1) * 2.0
        offset_y = offset[..., 1] / max(height - 1, 1) * 2.0
        grid = base_grid[..., None, None, :].expand(-1, -1, -1, crop_side, crop_side, -1)
        grid = grid + torch.stack((offset_x, offset_y), dim=-1)
        flat_grid = grid.reshape(batch * frames, points * crop_side, crop_side, 2)
        sampled = F.grid_sample(
            flat_rgb,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_valid = F.grid_sample(
            flat_valid,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(batch, frames, 3, points, crop_side, crop_side)
        crops.append(sampled.permute(0, 1, 3, 2, 4, 5))
        sampled_valid = sampled_valid.reshape(
            batch, frames, points, crop_side, crop_side
        )
        validity.append(sampled_valid.mean(dim=(-1, -2)))
        effective.append(scale)
    return NativeContinuousCropsV67(
        rgb=torch.stack(crops, dim=3),
        valid_fraction=torch.stack(validity, dim=3),
        effective_scales=torch.stack(effective, dim=3),
    )
