"""Frozen CoTracker evidence for pure-video object supervision."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PointTrackEvidence:
    coordinates: torch.Tensor
    visibility: torch.Tensor
    residual_flow: torch.Tensor
    motion_salience: torch.Tensor
    query_times: torch.Tensor
    sampled_features: torch.Tensor


def _sample_grid(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    batch, frames, points = coordinates.shape[:3]
    height, width = grid_hw
    if values.shape[:3] != (batch, frames, height * width):
        raise ValueError("point-track sampling grid differs from feature layout")
    channels = values.shape[-1]
    image = values.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
    grid = coordinates.reshape(batch * frames, points, 1, 2).to(image.dtype)
    sampled = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0].permute(0, 2, 1).reshape(batch, frames, points, channels)


def sample_patch_field(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    return _sample_grid(values, coordinates, grid_hw)


class FrozenPointTrackerRuntime:
    """Run a required local CoTracker checkpoint outside model/checkpoint state."""

    def __init__(
        self,
        config,
        device: torch.device,
        checkpoint_path: str,
        sequence_batch: int = 1,
    ):
        if device.type != "cuda":
            raise ValueError("v50 point tracking requires CUDA")
        if sequence_batch != 1:
            raise ValueError(
                "CoTracker3 offline requires tracker sequence batch 1"
            )
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"local CoTracker checkpoint is missing: {checkpoint_path}")
        from cotracker.predictor import CoTrackerPredictor

        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.sequence_batch = int(sequence_batch)
        self.model = CoTrackerPredictor(
            checkpoint=checkpoint_path,
            v2=False,
            offline=True,
        ).to(device).eval()
        self.model.requires_grad_(False)

    def _queries(self, batch: int, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        side = self.config.tracker_grid_side
        size = self.config.tracker_image_size
        axis = (
            torch.arange(side, device=self.device, dtype=torch.float32) + 0.5
        ) * (size / side)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack((x, y), dim=-1).reshape(-1, 2)
        times = torch.tensor(
            [round(fraction * (frames - 1)) for fraction in self.config.tracker_anchor_fractions],
            device=self.device,
            dtype=torch.float32,
        )
        query_xy = xy.repeat(len(times), 1)
        query_time = times[:, None].expand(-1, len(xy)).reshape(-1)
        query = torch.cat((query_time[:, None], query_xy), dim=-1)
        return query[None].expand(batch, -1, -1).clone(), query_time.long()

    @torch.no_grad()
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        patches: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> PointTrackEvidence:
        rgb = batch["video_rgb"]
        pixel_valid = batch["video_pixel_valid"]
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("v50 tracker expects [B,T,3,H,W] uint8 RGB")
        batch_size, frames = rgb.shape[:2]
        size = self.config.tracker_image_size
        resized = F.interpolate(
            rgb.flatten(0, 1).float(),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, frames, 3, size, size)
        validity = F.interpolate(
            pixel_valid.flatten(0, 1)[:, None].float(),
            size=(size, size),
            mode="nearest",
        ).reshape(batch_size, frames, size, size)
        queries, query_times = self._queries(batch_size, frames)
        tracks, visibility = [], []
        for start in range(0, batch_size, self.sequence_batch):
            stop = min(start + self.sequence_batch, batch_size)
            predicted, visible = self.model(resized[start:stop], queries=queries[start:stop])
            tracks.append(predicted.float())
            visibility.append(visible.bool())
        tracks = torch.cat(tracks)
        visibility = torch.cat(visibility)
        normalized = tracks.clone()
        normalized[..., 0] = normalized[..., 0] / max(size - 1, 1) * 2.0 - 1.0
        normalized[..., 1] = normalized[..., 1] / max(size - 1, 1) * 2.0 - 1.0
        in_frame = normalized.abs().amax(dim=-1) <= 1.0
        sampled_validity = _sample_grid(
            validity.reshape(batch_size, frames, size * size, 1),
            normalized,
            (size, size),
        )[..., 0]
        visibility = visibility & in_frame & (sampled_validity >= 0.5)
        sampled_features = _sample_grid(patches.float(), normalized, grid_hw)
        sampled_features = F.normalize(sampled_features, dim=-1, eps=1e-6)
        flow = normalized[:, 1:] - normalized[:, :-1]
        pair_visible = visibility[:, 1:] & visibility[:, :-1]
        weight = pair_visible.float()
        global_flow = (flow * weight[..., None]).sum(dim=2, keepdim=True)
        global_flow = global_flow / weight.sum(dim=2, keepdim=True).clamp_min(1.0)[..., None]
        residual = flow - global_flow
        speed = residual.norm(dim=-1) * weight
        scale = torch.quantile(speed, 0.75, dim=2, keepdim=True).clamp_min(0.01)
        salience = (speed / scale).clamp(0.0, 1.0) * weight
        tensors = (normalized, residual, salience, sampled_features)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("v50 frozen point tracker produced non-finite evidence")
        return PointTrackEvidence(
            coordinates=normalized.detach(),
            visibility=visibility.detach(),
            residual_flow=residual.detach(),
            motion_salience=salience.detach(),
            query_times=query_times.detach(),
            sampled_features=sampled_features.detach(),
        )
