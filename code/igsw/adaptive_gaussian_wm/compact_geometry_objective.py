"""Relative geometry and lifecycle losses for compact root/region states."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .object_lifecycle import (
    balanced_continuous_focal_loss,
    balanced_continuous_probability_loss,
)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _balanced_binary_error(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target.float()
    valid = torch.ones_like(target)
    if sample_valid is not None:
        valid = valid * sample_valid[:, None].to(valid.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), target, reduction="none"
    )
    positive = valid * target
    negative = valid * (1.0 - target)
    positive_count = positive.sum()
    negative_count = negative.sum()
    positive_loss = (loss * positive).sum() / positive_count.clamp_min(1.0)
    negative_loss = (loss * negative).sum() / negative_count.clamp_min(1.0)
    positive_available = (positive_count > 0).to(loss.dtype)
    negative_available = (negative_count > 0).to(loss.dtype)
    return (
        positive_available * positive_loss + negative_available * negative_loss
    ) / (positive_available + negative_available).clamp_min(1.0)


def _relative_covariance_terms(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = predicted.float()
    target = target.float()
    predicted_diagonal = predicted.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
    target_diagonal = target.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
    predicted_log_axis = 0.5 * predicted_diagonal.log()
    target_log_axis = 0.5 * target_diagonal.log()
    denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    predicted_log_axis = predicted_log_axis - (
        predicted_log_axis * weight[..., None]
    ).sum(dim=1, keepdim=True) / denominator[..., None]
    target_log_axis = target_log_axis - (
        target_log_axis * weight[..., None]
    ).sum(dim=1, keepdim=True) / denominator[..., None]
    relative_scale = F.smooth_l1_loss(
        predicted_log_axis,
        target_log_axis,
        reduction="none",
    ).mean(dim=-1)
    predicted_correlation = predicted[..., 0, 1] / torch.sqrt(
        predicted_diagonal[..., 0] * predicted_diagonal[..., 1]
    )
    target_correlation = target[..., 0, 1] / torch.sqrt(
        target_diagonal[..., 0] * target_diagonal[..., 1]
    )
    correlation = F.smooth_l1_loss(
        predicted_correlation.clamp(-0.999, 0.999),
        target_correlation.clamp(-0.999, 0.999),
        reduction="none",
    )
    return 0.25 * relative_scale, 0.1 * correlation


def _region_pair_terms(
    predicted_relative: torch.Tensor,
    predicted_covariance: torch.Tensor,
    predicted_presence_logits: torch.Tensor,
    predicted_visibility_logits: torch.Tensor,
    target_relative: torch.Tensor,
    target_covariance: torch.Tensor,
    target_presence: torch.Tensor,
    target_visibility: torch.Tensor,
    geometry_weight: torch.Tensor,
    sample_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    center = F.smooth_l1_loss(
        predicted_relative, target_relative, reduction="none"
    ).mean(dim=-1)
    relative_scale, correlation = _relative_covariance_terms(
        predicted_covariance, target_covariance, geometry_weight
    )
    presence = _balanced_binary_error(
        predicted_presence_logits,
        target_presence,
        sample_valid,
    )
    visibility_error = F.binary_cross_entropy_with_logits(
        predicted_visibility_logits.float(),
        target_visibility.float(),
        reduction="none",
    )
    visibility_weight = target_presence.float()
    if sample_valid is not None:
        visibility_weight = visibility_weight * sample_valid[:, None]
    return {
        "center": _weighted_mean(center, geometry_weight),
        "relative_scale": _weighted_mean(relative_scale, geometry_weight),
        "correlation": _weighted_mean(correlation, geometry_weight),
        "presence": presence,
        "visibility": _weighted_mean(visibility_error, visibility_weight),
    }


def _sum_terms(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(terms.values())).sum()


def region_geometry_terms(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    object_mass = target.owner[..., :-2].sum(dim=-1)
    scene_mass = target.owner[..., -2]
    weight = target.presence * (object_mass + 0.1 * scene_mass)
    if sample_valid is not None:
        weight = weight * sample_valid[:, None]
    return _region_pair_terms(
        prediction.future_relative_center[:, index],
        prediction.future_covariance[:, index],
        prediction.future_presence_logits[:, index],
        prediction.future_visibility_logits[:, index],
        target.relative_center,
        target.covariance,
        target.presence,
        target.visibility,
        weight,
        sample_valid,
    )


def region_geometry_error(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    return _sum_terms(
        region_geometry_terms(prediction, target, index, sample_valid)
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


def root_geometry_terms(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
    current=None,
    lifecycle_gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    weight = target.existence
    if sample_valid is not None:
        weight = weight * sample_valid[:, None]
    relation = F.smooth_l1_loss(
        prediction.future_relations[:, index].float(),
        target.relations.float(),
        reduction="none",
    ).mean(dim=(-1, -2))
    factorized = (
        current is not None
        and prediction.future_survival_logits is not None
        and prediction.future_birth_logits is not None
        and prediction.future_observability_logits is not None
    )
    if factorized:
        valid = torch.ones_like(target.existence).float()
        if sample_valid is not None:
            valid = valid * sample_valid[:, None].float()
        current_existence = current.existence.detach().float()
        presence = balanced_continuous_probability_loss(
            prediction.future_existence[:, index],
            target.existence,
            valid,
        )
        visibility = 0.5 * balanced_continuous_probability_loss(
            prediction.future_visibility[:, index],
            target.visibility,
            valid,
        )
        survival = 0.25 * balanced_continuous_focal_loss(
            prediction.future_survival_logits[:, index],
            target.existence,
            current_existence * valid,
            lifecycle_gamma,
        )
        birth = 0.25 * balanced_continuous_focal_loss(
            prediction.future_birth_logits[:, index],
            target.existence,
            (1.0 - current_existence) * valid,
            lifecycle_gamma,
        )
        conditional_observation = (
            target.visibility.float()
            / (target.existence.float() * target.in_frame.float()).clamp_min(1e-4)
        ).clamp(0.0, 1.0)
        observability = 0.5 * balanced_continuous_focal_loss(
            prediction.future_observability_logits[:, index],
            conditional_observation,
            target.existence.float() * target.in_frame.float() * valid,
            lifecycle_gamma,
        )
    else:
        presence = _balanced_binary_error(
            prediction.future_existence_logits[:, index],
            target.existence,
            sample_valid,
        )
        visibility_error = F.binary_cross_entropy_with_logits(
            prediction.future_visibility_logits[:, index].float(),
            target.visibility.float(),
            reduction="none",
        )
        visibility = _weighted_mean(visibility_error, weight)
        zero = presence * 0.0
        survival = zero
        birth = zero
        observability = zero
    return {
        "relation": _weighted_mean(relation, weight),
        "presence": presence,
        "visibility": visibility,
        "survival": survival,
        "birth": birth,
        "observability": observability,
    }


def root_geometry_error(
    prediction,
    target,
    index: int,
    sample_valid: torch.Tensor | None = None,
    current=None,
    lifecycle_gamma: float = 2.0,
) -> torch.Tensor:
    return _sum_terms(
        root_geometry_terms(
            prediction,
            target,
            index,
            sample_valid,
            current,
            lifecycle_gamma,
        )
    )


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
    region_terms = _region_pair_terms(
        rollout_region.future_relative_center[:, 0],
        rollout_region.future_covariance[:, 0],
        rollout_region.future_presence_logits[:, 0],
        rollout_region.future_visibility_logits[:, 0],
        direct_region.future_relative_center[:, 1].detach(),
        direct_region.future_covariance[:, 1].detach(),
        torch.sigmoid(direct_region.future_presence_logits[:, 1].detach()),
        torch.sigmoid(direct_region.future_visibility_logits[:, 1].detach()),
        region_weight,
    )
    return root + _sum_terms(region_terms)


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
