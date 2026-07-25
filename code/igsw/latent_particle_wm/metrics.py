"""Evaluation metrics spanning accuracy, stochastic coverage, and geometry."""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def camera_xyz(uv: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    cx = intrinsics[:, 0, 2][:, None]
    cy = intrinsics[:, 1, 2][:, None]
    x = (uv[..., 0] - cx) / fx * z
    y = (uv[..., 1] - cy) / fy * z
    return torch.stack((x, y, z), dim=-1)


def prediction_xyz(batch: dict, prediction: torch.Tensor) -> torch.Tensor:
    height = batch["image_hw"][:, 0].float()[:, None]
    width = batch["image_hw"][:, 1].float()[:, None]
    uv = batch["uv0"].clone()
    uv[..., 0] += prediction[..., 0] * width
    uv[..., 1] += prediction[..., 1] * height
    z = batch["z0"] * prediction[..., 2].clamp(-2.0, 2.0).exp()
    return camera_xyz(uv, z, batch["intrinsics"])


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    selected = value[mask]
    if selected.numel() == 0:
        return 0.0, 0
    return float(selected.sum()), int(selected.numel())


def _local_pair_error(
    predicted_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, int]:
    particle_count = predicted_xyz.shape[1]
    side = round(particle_count**0.5)
    if side * side != particle_count:
        raise ValueError(f"particle count must be square, got {particle_count}")
    pred = predicted_xyz.reshape(len(predicted_xyz), side, side, 3)
    target = target_xyz.reshape(len(target_xyz), side, side, 3)
    mask = valid.reshape(len(valid), side, side)
    errors, masks = [], []
    for dimension in (1, 2):
        if dimension == 1:
            pred_distance = (pred[:, 1:] - pred[:, :-1]).norm(dim=-1)
            target_distance = (target[:, 1:] - target[:, :-1]).norm(dim=-1)
            pair_mask = mask[:, 1:] & mask[:, :-1]
        else:
            pred_distance = (pred[:, :, 1:] - pred[:, :, :-1]).norm(dim=-1)
            target_distance = (target[:, :, 1:] - target[:, :, :-1]).norm(dim=-1)
            pair_mask = mask[:, :, 1:] & mask[:, :, :-1]
        errors.append((pred_distance - target_distance).abs())
        masks.append(pair_mask)
    return _safe_mean(torch.cat([value.flatten() for value in errors]), torch.cat([value.flatten() for value in masks]))


class MetricAccumulator:
    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, total: float, count: int) -> None:
        self.sums[name] += total
        self.counts[name] += count

    def mean(self, name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        total, count = _safe_mean(value, mask)
        self.add(name, total, count)

    def compute(self) -> dict[str, float]:
        return {
            key: self.sums[key] / max(self.counts[key], 1)
            for key in sorted(self.sums)
        }


def add_point_metrics(
    accumulator: MetricAccumulator,
    prefix: str,
    batch: dict,
    prediction: torch.Tensor,
    visibility_logits: torch.Tensor,
) -> None:
    valid = batch["valid"]
    visible = batch["visible"]
    motion_valid = batch["motion_valid"]
    target = batch["target"]
    height = batch["image_hw"][:, 0].float()[:, None]
    width = batch["image_hw"][:, 1].float()[:, None]
    scale = torch.stack((width, height), dim=-1)
    gt_flow_px = target[..., :2] * scale
    pred_flow_px = prediction[..., :2] * scale
    gt_magnitude = gt_flow_px.norm(dim=-1)
    pred_magnitude = pred_flow_px.norm(dim=-1)
    mover = (target[..., :2].norm(dim=-1) > 0.01) & motion_valid
    static = ~mover & motion_valid
    flow_error = (pred_flow_px - gt_flow_px).norm(dim=-1)
    accumulator.mean(f"{prefix}/flow_epe_px", flow_error, motion_valid)
    accumulator.mean(f"{prefix}/flow_epe_mover_px", flow_error, mover)
    accumulator.mean(
        f"{prefix}/flow_dcos_mover",
        F.cosine_similarity(pred_flow_px, gt_flow_px, dim=-1),
        mover,
    )
    gt_mag_total, mover_count = _safe_mean(gt_magnitude, mover)
    pred_mag_total, _ = _safe_mean(pred_magnitude, mover)
    accumulator.add(f"{prefix}/magnitude_ratio", pred_mag_total / max(gt_mag_total, 1e-8), 1 if mover_count else 0)
    accumulator.mean(f"{prefix}/static_flow_px", pred_magnitude, static)
    accumulator.mean(
        f"{prefix}/depth_log_mae",
        (prediction[..., 2] - target[..., 2]).abs(),
        motion_valid,
    )
    accumulator.mean(
        f"{prefix}/appearance_l1",
        (prediction[..., 3:] - target[..., 3:]).abs().mean(dim=-1),
        visible,
    )
    predicted_visible = visibility_logits.sigmoid() >= 0.5
    accumulator.mean(
        f"{prefix}/visibility_accuracy",
        (predicted_visible == visible).float(),
        valid,
    )
    predicted_xyz = prediction_xyz(batch, prediction)
    xyz_error = (predicted_xyz - batch["target_xyz"]).norm(dim=-1) * 100.0
    accumulator.mean(f"{prefix}/xyz_epe_cm", xyz_error, motion_valid)
    accumulator.mean(f"{prefix}/xyz_epe_mover_cm", xyz_error, mover)
    pair_total, pair_count = _local_pair_error(predicted_xyz, batch["target_xyz"], motion_valid)
    accumulator.add(f"{prefix}/local_pair_cm", pair_total * 100.0, pair_count)

    predicted_mover = prediction[..., :2].norm(dim=-1) > 0.01
    true_positive = (predicted_mover & mover).sum().item()
    accumulator.add(
        f"{prefix}/mover_precision",
        true_positive / max((predicted_mover & motion_valid).sum().item(), 1),
        1,
    )
    accumulator.add(f"{prefix}/mover_recall", true_positive / max(mover.sum().item(), 1), 1)


def add_sample_metrics(
    accumulator: MetricAccumulator,
    batch: dict,
    samples: torch.Tensor,
) -> None:
    """samples is [S,B,N,6]; score coverage per clip before averaging."""
    valid = batch["motion_valid"]
    target = batch["target"]
    height = batch["image_hw"][:, 0].float()[None, :, None]
    width = batch["image_hw"][:, 1].float()[None, :, None]
    scale = torch.stack((width.expand_as(height), height), dim=-1)
    target_flow = target[None, ..., :2] * scale
    sample_flow = samples[..., :2] * scale
    point_error = (sample_flow - target_flow).norm(dim=-1)
    weight = valid.float()[None]
    clip_error = (point_error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    best_error = clip_error.min(dim=0).values
    accumulator.add("prior_best/flow_epe_px", float(best_error.sum()), len(best_error))
    mover = (target[..., :2].norm(dim=-1) > 0.01) & valid
    mover_weight = mover.float()[None]
    mover_count = mover_weight.sum(dim=-1)
    mover_clip_error = (point_error * mover_weight).sum(dim=-1) / mover_count.clamp_min(1.0)
    has_mover = mover_count[0] > 0
    best_mover_error = mover_clip_error.min(dim=0).values
    accumulator.add(
        "prior_best/flow_epe_mover_px",
        float(best_mover_error[has_mover].sum()),
        int(has_mover.sum()),
    )

    sample_xyz = torch.stack([prediction_xyz(batch, sample) for sample in samples])
    xyz_error = (sample_xyz - batch["target_xyz"][None]).norm(dim=-1) * 100.0
    clip_xyz = (xyz_error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    best_xyz = clip_xyz.min(dim=0).values
    accumulator.add("prior_best/xyz_epe_cm", float(best_xyz.sum()), len(best_xyz))
    mover_clip_xyz = (xyz_error * mover_weight).sum(dim=-1) / mover_count.clamp_min(1.0)
    best_mover_xyz = mover_clip_xyz.min(dim=0).values
    accumulator.add(
        "prior_best/xyz_epe_mover_cm",
        float(best_mover_xyz[has_mover].sum()),
        int(has_mover.sum()),
    )

    flow_std = sample_flow.std(dim=0).norm(dim=-1)
    accumulator.mean("prior_samples/diversity_px", flow_std, valid)
    accumulator.mean("prior_samples/diversity_mover_px", flow_std, mover)
    mean_error = (sample_flow.mean(dim=0) - target_flow[0]).norm(dim=-1)
    selected_std = flow_std[valid]
    selected_error = mean_error[valid]
    if selected_std.numel() > 1 and selected_std.std() > 1e-8 and selected_error.std() > 1e-8:
        correlation = torch.corrcoef(torch.stack((selected_std, selected_error)))[0, 1]
        accumulator.add("prior_samples/uncertainty_error_corr", float(correlation), 1)


def active_mover_recall(batch: dict) -> tuple[float, int]:
    mover = (batch["target"][..., :2].norm(dim=-1) > 0.01) & batch["motion_valid"]
    denominator = mover.sum().item()
    if not denominator:
        return 0.0, 0
    return float((mover & batch["active"]).sum().item() / denominator), 1
