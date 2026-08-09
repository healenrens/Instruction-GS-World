"""Relative geometry and lifecycle losses for compact root/region states."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _relative_covariance_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    predicted_area = torch.linalg.det(predicted.float()).clamp_min(1e-8)
    target_area = torch.linalg.det(target.float()).clamp_min(1e-8)
    denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    predicted_log_area = predicted_area.log()
    target_log_area = target_area.log()
    predicted_log_area -= (
        predicted_log_area * weight
    ).sum(dim=1, keepdim=True) / denominator
    target_log_area -= (
        target_log_area * weight
    ).sum(dim=1, keepdim=True) / denominator
    area = F.smooth_l1_loss(predicted_log_area, target_log_area, reduction="none")
    predicted_shape = predicted / predicted_area.sqrt()[..., None, None]
    target_shape = target / target_area.sqrt()[..., None, None]
    shape = F.smooth_l1_loss(
        predicted_shape, target_shape, reduction="none"
    ).mean(dim=(-1, -2))
    return 0.25 * area + 0.1 * shape


def _region_pair_error(
    predicted_relative: torch.Tensor,
    predicted_covariance: torch.Tensor,
    predicted_presence: torch.Tensor,
    predicted_visibility: torch.Tensor,
    target_relative: torch.Tensor,
    target_covariance: torch.Tensor,
    target_presence: torch.Tensor,
    target_visibility: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    center = F.smooth_l1_loss(
        predicted_relative, target_relative, reduction="none"
    ).mean(dim=-1)
    covariance = _relative_covariance_error(
        predicted_covariance, target_covariance, weight
    )
    lifecycle = F.binary_cross_entropy_with_logits(
        torch.logit(predicted_presence.float().clamp(1e-5, 1.0 - 1e-5)),
        target_presence.float(),
        reduction="none",
    )
    visibility = F.binary_cross_entropy_with_logits(
        torch.logit(predicted_visibility.float().clamp(1e-5, 1.0 - 1e-5)),
        target_visibility.float(),
        reduction="none",
    )
    return _weighted_mean(center + covariance + lifecycle + visibility, weight)


def region_geometry_error(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    object_mass = target.owner[..., :-2].sum(dim=-1)
    scene_mass = target.owner[..., -2]
    weight = target.presence * (object_mass + 0.1 * scene_mass)
    if sample_valid is not None:
        weight = weight * sample_valid[:, None]
    return _region_pair_error(
        prediction.future_relative_center[:, index],
        prediction.future_covariance[:, index],
        prediction.future_presence[:, index],
        prediction.future_visibility[:, index],
        target.relative_center,
        target.covariance,
        target.presence,
        target.visibility,
        weight,
    )


def region_identity_error(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    object_mass = target.owner[..., :-2].sum(dim=-1)
    weight = target.presence * object_mass
    if sample_valid is not None:
        weight = weight * sample_valid[:, None]
    error = 1.0 - F.cosine_similarity(
        prediction.future_identity_key[:, index].float(),
        target.identity_key.float(),
        dim=-1,
    )
    return _weighted_mean(error, weight)


def root_geometry_error(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = target.existence
    if sample_valid is not None:
        weight = weight * sample_valid[:, None]
    relation = F.smooth_l1_loss(
        prediction.future_relations[:, index].float(),
        target.relations.float(),
        reduction="none",
    ).mean(dim=(-1, -2))
    lifecycle = F.binary_cross_entropy_with_logits(
        prediction.future_existence_logits[:, index].float(),
        target.existence.float(),
        reduction="none",
    )
    visibility = F.binary_cross_entropy_with_logits(
        prediction.future_visibility_logits[:, index].float(),
        target.visibility.float(),
        reduction="none",
    )
    return _weighted_mean(relation + lifecycle + visibility, weight)


def path_geometry_error(
    direct_root,
    direct_region,
    rollout_root,
    rollout_region,
    root_weight: torch.Tensor,
    region_weight: torch.Tensor,
) -> torch.Tensor:
    root_relation = F.smooth_l1_loss(
        rollout_root.future_relations[:, 0].float(),
        direct_root.future_relations[:, 1].detach().float(),
        reduction="none",
    ).mean(dim=(-1, -2))
    root_lifecycle = F.smooth_l1_loss(
        rollout_root.future_existence[:, 0].float(),
        direct_root.future_existence[:, 1].detach().float(),
        reduction="none",
    )
    root_visibility = F.smooth_l1_loss(
        rollout_root.future_visibility[:, 0].float(),
        direct_root.future_visibility[:, 1].detach().float(),
        reduction="none",
    )
    root = _weighted_mean(
        root_relation + root_lifecycle + root_visibility, root_weight
    )
    region = _region_pair_error(
        rollout_region.future_relative_center[:, 0],
        rollout_region.future_covariance[:, 0],
        rollout_region.future_presence[:, 0],
        rollout_region.future_visibility[:, 0],
        direct_region.future_relative_center[:, 1].detach(),
        direct_region.future_covariance[:, 1].detach(),
        direct_region.future_presence[:, 1].detach(),
        direct_region.future_visibility[:, 1].detach(),
        region_weight,
    )
    return root + region


def path_region_identity_error(
    direct_region,
    rollout_region,
    weight: torch.Tensor,
) -> torch.Tensor:
    error = 1.0 - F.cosine_similarity(
        rollout_region.future_identity_key[:, 0].float(),
        direct_region.future_identity_key[:, 1].detach().float(),
        dim=-1,
    )
    return _weighted_mean(error, weight)
