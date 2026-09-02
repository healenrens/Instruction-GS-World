"""Native-pixel tiled perception fields and local token aggregation for v65."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .continuous_object_observation_v62 import FrozenSiglipVideoRuntimeV62
from .fixed_teacher_projection_v61 import fixed_group_projection_v61
from .frozen_video_encoder import FrozenDinoVideoRuntime


@dataclass(frozen=True)
class NativeTokenFieldV65:
    features: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor
    image_hw: torch.Tensor


@dataclass(frozen=True)
class NativeLocalFeatureFieldV65:
    dino: NativeTokenFieldV65
    siglip: NativeTokenFieldV65


@dataclass(frozen=True)
class LocalPointFeaturesV65:
    features: torch.Tensor
    valid: torch.Tensor


def _tile_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def _native_tiles(rgb, pixel_valid, size: int, stride: int):
    batch, frames, _, height, width = rgb.shape
    padded_height = max(height, size)
    padded_width = max(width, size)
    pad = (0, padded_width - width, 0, padded_height - height)
    padded_rgb = F.pad(rgb, pad)
    padded_valid = F.pad(pixel_valid, pad)
    starts = [
        (top, left)
        for top in _tile_starts(padded_height, size, stride)
        for left in _tile_starts(padded_width, size, stride)
    ]
    tiles = torch.stack(
        [
            padded_rgb[..., top : top + size, left : left + size]
            for top, left in starts
        ],
        dim=2,
    )
    validity = torch.stack(
        [
            padded_valid[..., top : top + size, left : left + size]
            for top, left in starts
        ],
        dim=2,
    )
    tiles = tiles.reshape(batch * frames * len(starts), 3, size, size)
    validity = validity.reshape(batch * frames * len(starts), size, size)
    return tiles, validity, starts, (height, width), batch, frames


def _token_coordinates(starts, grid: int, patch: int, native_hw, device):
    offset = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) * patch
    local_y, local_x = torch.meshgrid(offset, offset, indexing="ij")
    local = torch.stack((local_x, local_y), dim=-1).reshape(-1, 2)
    absolute_coordinates = []
    for top, left in starts:
        absolute = local + torch.tensor((left, top), device=device)
        absolute_coordinates.append(absolute)
    absolute = torch.cat(absolute_coordinates, dim=0)
    coordinates = absolute[None].expand(len(native_hw), -1, -1).clone()
    height = native_hw[:, 0].float().clamp_min(2.0)[:, None]
    width = native_hw[:, 1].float().clamp_min(2.0)[:, None]
    coordinates[..., 0] = coordinates[..., 0] / (width - 1.0) * 2.0 - 1.0
    coordinates[..., 1] = coordinates[..., 1] / (height - 1.0) * 2.0 - 1.0
    return coordinates


class NativeTiledPerceptionRuntimeV65:
    """Encode overlapping native-pixel tiles without whole-frame downscaling."""

    def __init__(
        self,
        config,
        device,
        amp,
        dino_checkpoint,
        siglip_checkpoint,
        dino_frame_batch,
        siglip_frame_batch,
    ):
        self.config = config
        self.device = device
        self.dino = FrozenDinoVideoRuntime(
            config, device, amp, dino_frame_batch, dino_checkpoint
        )
        self.siglip = FrozenSiglipVideoRuntimeV62(
            siglip_checkpoint, device, siglip_frame_batch
        )

    @torch.no_grad()
    def _encode_dino(self, tiles, validity, starts, native_hw, batch, frames):
        outputs, masks = [], []
        for start in range(0, len(tiles), self.dino.frame_batch):
            stop = min(start + self.dino.frame_batch, len(tiles))
            images = tiles[start:stop].float() / 255.0
            valid = validity[start:stop, None].float()
            images = ((images - self.dino.mean) / self.dino.std) * valid
            encoded = self.dino.backbone.forward_features(images.to(self.dino.dtype))
            patches = encoded[:, self.dino.prefix_tokens :].float()
            patches = F.normalize(patches, dim=-1, eps=1e-6)
            patches = fixed_group_projection_v61(
                patches, self.config.semantic_dim
            ).to(self.dino.dtype)
            outputs.append(patches)
            masks.append(
                F.avg_pool2d(valid, self.dino.patch_size, self.dino.patch_size)
                .flatten(1)
                .ge(0.5)
            )
        grid = self.config.native_tile_size // self.dino.patch_size
        per_tile = grid * grid
        tile_count = len(starts)
        features = torch.cat(outputs).reshape(
            batch, frames, tile_count * per_tile, self.config.semantic_dim
        )
        valid = torch.cat(masks).reshape(batch, frames, tile_count * per_tile)
        coordinates = _token_coordinates(
            starts, grid, self.dino.patch_size, native_hw, features.device
        )
        coordinates = coordinates[:, None].expand(batch, frames, -1, -1)
        return NativeTokenFieldV65(features, coordinates, valid, native_hw)

    @torch.no_grad()
    def _encode_siglip(self, tiles, validity, starts, native_hw, batch, frames):
        outputs, masks = [], []
        for start in range(0, len(tiles), self.siglip.frame_batch):
            stop = min(start + self.siglip.frame_batch, len(tiles))
            images = tiles[start:stop].float() / 255.0
            valid = validity[start:stop, None].float()
            encoded = self.siglip.model(pixel_values=(images - 0.5) / 0.5)
            patches = F.normalize(encoded.last_hidden_state.float(), dim=-1)
            patches = fixed_group_projection_v61(
                patches, self.config.semantic_dim
            ).to(self.dino.dtype)
            outputs.append(patches)
            masks.append(
                F.avg_pool2d(valid, self.siglip.patch_size, self.siglip.patch_size)
                .flatten(1)
                .ge(0.5)
            )
        grid = self.config.native_tile_size // self.siglip.patch_size
        per_tile = grid * grid
        tile_count = len(starts)
        features = torch.cat(outputs).reshape(
            batch, frames, tile_count * per_tile, self.config.semantic_dim
        )
        valid = torch.cat(masks).reshape(batch, frames, tile_count * per_tile)
        coordinates = _token_coordinates(
            starts, grid, self.siglip.patch_size, native_hw, features.device
        )
        coordinates = coordinates[:, None].expand(batch, frames, -1, -1)
        return NativeTokenFieldV65(features, coordinates, valid, native_hw)

    @torch.no_grad()
    def __call__(self, batch) -> NativeLocalFeatureFieldV65:
        rgb = batch["video_rgb"]
        pixel_valid = batch["video_pixel_valid"]
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("v65 expects [B,T,3,H,W] uint8 native RGB")
        native_hw = batch.get("native_image_hw")
        if native_hw is None:
            native_hw = torch.tensor(
                rgb.shape[-2:], device=rgb.device, dtype=torch.long
            )[None].expand(len(rgb), -1)
        tiles, validity, starts, _, batch_size, frames = _native_tiles(
            rgb,
            pixel_valid,
            self.config.native_tile_size,
            self.config.native_tile_stride,
        )
        dino = self._encode_dino(
            tiles, validity, starts, native_hw, batch_size, frames
        )
        siglip = self._encode_siglip(
            tiles, validity, starts, native_hw, batch_size, frames
        )
        return NativeLocalFeatureFieldV65(dino=dino, siglip=siglip)


def select_native_token_frames_v65(field: NativeTokenFieldV65, indices):
    index = torch.as_tensor(indices, device=field.features.device, dtype=torch.long)
    return NativeTokenFieldV65(
        features=field.features.index_select(1, index),
        coordinates=field.coordinates.index_select(1, index),
        valid=field.valid.index_select(1, index),
        image_hw=field.image_hw,
    )


def pool_native_local_features_v65(
    field: NativeTokenFieldV65,
    query_coordinates: torch.Tensor,
    radii_pixels: tuple[float, ...],
    tokens_per_scale: int,
    point_chunk: int = 64,
) -> LocalPointFeaturesV65:
    if query_coordinates.ndim != 4 or query_coordinates.shape[-1] != 2:
        raise ValueError("v65 local queries must have shape [B,T,P,2]")
    if query_coordinates.shape[:2] != field.features.shape[:2]:
        raise ValueError("v65 local query frames differ from token field")
    batch, frames, points = query_coordinates.shape[:3]
    native_hw = field.image_hw.to(field.features.device)
    height = native_hw[:, 0].float().clamp_min(2.0)
    width = native_hw[:, 1].float().clamp_min(2.0)
    token_xy = field.coordinates.float().clone()
    token_xy[..., 0] = (
        (token_xy[..., 0] + 1.0) * 0.5 * (width - 1.0)[:, None, None]
    )
    token_xy[..., 1] = (
        (token_xy[..., 1] + 1.0) * 0.5 * (height - 1.0)[:, None, None]
    )
    query_xy = query_coordinates.float().clone()
    query_xy[..., 0] = (
        (query_xy[..., 0] + 1.0) * 0.5 * (width - 1.0)[:, None, None]
    )
    query_xy[..., 1] = (
        (query_xy[..., 1] + 1.0) * 0.5 * (height - 1.0)[:, None, None]
    )
    flat_features = field.features.flatten(0, 1).float()
    flat_tokens = token_xy.flatten(0, 1)
    flat_valid = field.valid.flatten(0, 1)
    flat_queries = query_xy.flatten(0, 1)
    pooled_chunks, valid_chunks = [], []
    topk = min(tokens_per_scale, flat_features.shape[1])
    for start in range(0, points, point_chunk):
        stop = min(start + point_chunk, points)
        query = flat_queries[:, start:stop]
        distance = (query[:, :, None] - flat_tokens[:, None]).square().sum(dim=-1)
        distance = distance.masked_fill(~flat_valid[:, None], float("inf"))
        nearest_distance, nearest_index = distance.topk(topk, dim=-1, largest=False)
        batch_index = torch.arange(len(flat_features), device=field.features.device)
        selected = flat_features[batch_index[:, None, None], nearest_index]
        selected_valid = torch.isfinite(nearest_distance)
        scales, scale_valid = [], []
        for radius in radii_pixels:
            weight = torch.exp(-nearest_distance / (2.0 * radius**2))
            local = nearest_distance <= radius**2
            weight = weight * selected_valid.float() * local.float()
            denominator = weight.sum(dim=-1, keepdim=True)
            pooled = (selected * weight[..., None]).sum(dim=-2)
            pooled = pooled / denominator.clamp_min(1e-6)
            valid = denominator[..., 0] > 1e-6
            scales.append(F.normalize(pooled, dim=-1, eps=1e-6))
            scale_valid.append(valid)
        scales = torch.stack(scales, dim=-2)
        scale_valid = torch.stack(scale_valid, dim=-1)
        combined = (scales * scale_valid[..., None].float()).sum(dim=-2)
        combined = combined / scale_valid.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled_chunks.append(F.normalize(combined, dim=-1, eps=1e-6))
        valid_chunks.append(scale_valid.any(dim=-1))
    features = torch.cat(pooled_chunks, dim=1).reshape(
        batch, frames, points, field.features.shape[-1]
    )
    valid = torch.cat(valid_chunks, dim=1).reshape(batch, frames, points)
    in_frame = query_coordinates.abs().amax(dim=-1) <= 1.0
    return LocalPointFeaturesV65(features, valid & in_frame)
