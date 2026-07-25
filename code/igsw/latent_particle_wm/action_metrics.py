"""Geometry and stochastic-coverage metrics for correlated action fields."""
from __future__ import annotations

from collections import defaultdict

import torch


class MetricAccumulator:
    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        selected = values[mask]
        if selected.numel():
            self.sums[name] += float(selected.sum())
            self.counts[name] += selected.numel()

    def add_clips(self, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        for clip_value, clip_mask in zip(values, mask):
            if bool(clip_mask.any()):
                self.sums[name] += float(clip_value[clip_mask].mean())
                self.counts[name] += 1

    def scalar(self, name: str, value: float, count: int = 1) -> None:
        self.sums[name] += value
        self.counts[name] += count

    def result(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in sorted(self.sums)
            if self.counts[name]
        }


def predicted_xyz(batch: dict, motion: torch.Tensor) -> torch.Tensor:
    sample_dims = motion.ndim - 3
    image_hw = batch["image_hw"].float()
    scale = torch.stack((image_hw[:, 1], image_hw[:, 0]), dim=-1)
    for _ in range(sample_dims):
        scale = scale[None]
    uv0 = batch["control_uv"]
    means = batch["control_means"]
    intrinsics = batch["intrinsics"]
    for _ in range(sample_dims):
        uv0 = uv0[None]
        means = means[None]
        intrinsics = intrinsics[None]
    uv = uv0 + motion[..., :2] * scale[..., None, :]
    depth = means[..., 2] * motion[..., 2].clamp(-1.5, 1.5).exp()
    fx = intrinsics[..., 0, 0][..., None]
    fy = intrinsics[..., 1, 1][..., None]
    cx = intrinsics[..., 0, 2][..., None]
    cy = intrinsics[..., 1, 2][..., None]
    x = (uv[..., 0] - cx) * depth / fx
    y = (uv[..., 1] - cy) * depth / fy
    return torch.stack((x, y, depth), dim=-1)


def point_errors(
    batch: dict,
    field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_dims = field.ndim - 3
    image_hw = batch["image_hw"].float()
    scale = torch.stack((image_hw[:, 1], image_hw[:, 0]), dim=-1)
    target = batch["target"]
    target_xyz = batch["target_xyz"]
    for _ in range(sample_dims):
        scale = scale[None]
        target = target[None]
        target_xyz = target_xyz[None]
    flow_error = ((field[..., :2] - target[..., :2]) * scale[..., None, :]).norm(dim=-1)
    xyz_error_cm = (predicted_xyz(batch, field[..., :3]) - target_xyz).norm(dim=-1) * 100.0
    return flow_error, xyz_error_cm


def add_prediction_metrics(
    accumulator: MetricAccumulator,
    prefix: str,
    batch: dict,
    field: torch.Tensor,
) -> None:
    flow_error, xyz_error = point_errors(batch, field)
    valid = batch["motion_valid"]
    mover = valid & (batch["target"][..., :2].norm(dim=-1) > 0.01)
    accumulator.add(f"{prefix}/flow_epe_px_point", flow_error, valid)
    accumulator.add(f"{prefix}/flow_epe_mover_px_point", flow_error, mover)
    accumulator.add(f"{prefix}/xyz_epe_cm_point", xyz_error, valid)
    accumulator.add(f"{prefix}/xyz_epe_mover_cm_point", xyz_error, mover)
    accumulator.add_clips(f"{prefix}/flow_epe_px_clip", flow_error, valid)
    accumulator.add_clips(f"{prefix}/flow_epe_mover_px_clip", flow_error, mover)
    accumulator.add_clips(f"{prefix}/xyz_epe_cm_clip", xyz_error, valid)
    accumulator.add_clips(f"{prefix}/xyz_epe_mover_cm_clip", xyz_error, mover)


def add_prior_coverage_metrics(
    accumulator: MetricAccumulator,
    batch: dict,
    sample_fields: torch.Tensor,
    sample_counts: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> None:
    flow_error, xyz_error = point_errors(batch, sample_fields)
    valid = batch["motion_valid"]
    mover = valid & (batch["target"][..., :2].norm(dim=-1) > 0.01)
    image_hw = batch["image_hw"].float()
    scale = torch.stack((image_hw[:, 1], image_hw[:, 0]), dim=-1)
    flow_samples = sample_fields[..., :2] * scale[None, :, None]
    diversity = flow_samples.std(dim=0, correction=0).norm(dim=-1)
    accumulator.add("prior/diversity_px_point", diversity, valid)
    accumulator.add("prior/diversity_mover_px_point", diversity, mover)

    for count in sample_counts:
        if count > len(sample_fields):
            continue
        best_flow = flow_error[:count].amin(dim=0)
        best_xyz = xyz_error[:count].amin(dim=0)
        prefix = f"prior_best_{count}"
        accumulator.add(f"{prefix}/flow_epe_px_point", best_flow, valid)
        accumulator.add(f"{prefix}/flow_epe_mover_px_point", best_flow, mover)
        accumulator.add(f"{prefix}/xyz_epe_cm_point", best_xyz, valid)
        accumulator.add(f"{prefix}/xyz_epe_mover_cm_point", best_xyz, mover)
        for clip in range(valid.shape[0]):
            clip_valid = valid[clip]
            clip_mover = mover[clip]
            if bool(clip_valid.any()):
                sample_clip = flow_error[:count, clip, clip_valid].mean(dim=-1)
                accumulator.scalar(f"{prefix}/flow_epe_px_clip_oracle", float(sample_clip.min()))
                sample_xyz = xyz_error[:count, clip, clip_valid].mean(dim=-1)
                accumulator.scalar(f"{prefix}/xyz_epe_cm_clip_oracle", float(sample_xyz.min()))
            if bool(clip_mover.any()):
                sample_mover = flow_error[:count, clip, clip_mover].mean(dim=-1)
                accumulator.scalar(
                    f"{prefix}/flow_epe_mover_px_clip_oracle",
                    float(sample_mover.min()),
                )
