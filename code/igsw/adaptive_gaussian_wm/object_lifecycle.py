"""Factorized track presence, discovery, and observation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ObjectLifecyclePrediction:
    survival_logits: torch.Tensor
    birth_logits: torch.Tensor
    observability_logits: torch.Tensor
    survival: torch.Tensor
    birth: torch.Tensor
    observability: torch.Tensor
    existence: torch.Tensor
    in_frame: torch.Tensor
    visibility: torch.Tensor
    existence_logits: torch.Tensor
    visibility_logits: torch.Tensor


def stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float().clamp(1e-4, 1.0 - 1e-4))


def soft_in_frame(center: torch.Tensor) -> torch.Tensor:
    if center.shape[-1] != 2:
        raise ValueError("object center must end with dimension two")
    margin = 1.0 - center.abs().amax(dim=-1)
    return torch.sigmoid(10.0 * margin)


def factorized_lifecycle_prediction(
    current_existence: torch.Tensor,
    current_visible_presence: torch.Tensor,
    future_center: torch.Tensor,
    survival_delta: torch.Tensor,
    birth_delta: torch.Tensor,
    observability_delta: torch.Tensor,
    survival_prior: float,
    birth_prior: float,
) -> ObjectLifecyclePrediction:
    expected = future_center.shape[:-1]
    for name, value in (
        ("current_existence", current_existence),
        ("current_visible_presence", current_visible_presence),
        ("survival_delta", survival_delta),
        ("birth_delta", birth_delta),
        ("observability_delta", observability_delta),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    current_existence = current_existence.float().clamp(0.0, 1.0)
    current_visible_presence = current_visible_presence.float().clamp(0.0, 1.0)
    survival_logits = stable_logit(
        current_existence.new_full((), survival_prior)
    ) + survival_delta.float()
    birth_logits = stable_logit(
        current_existence.new_full((), birth_prior)
    ) + birth_delta.float()
    observed_ratio = (
        current_visible_presence / current_existence.clamp_min(1e-4)
    ).clamp(0.05, 0.95)
    conditional_visibility = torch.where(
        current_existence > 1e-3,
        observed_ratio,
        torch.full_like(observed_ratio, 0.5),
    )
    observability_logits = stable_logit(conditional_visibility) + observability_delta.float()
    survival = torch.sigmoid(survival_logits)
    birth = torch.sigmoid(birth_logits)
    observability = torch.sigmoid(observability_logits)
    existence = current_existence * survival + (1.0 - current_existence) * birth
    in_frame = soft_in_frame(future_center)
    visibility = existence * in_frame * observability
    return ObjectLifecyclePrediction(
        survival_logits=survival_logits,
        birth_logits=birth_logits,
        observability_logits=observability_logits,
        survival=survival,
        birth=birth,
        observability=observability,
        existence=existence,
        in_frame=in_frame,
        visibility=visibility,
        existence_logits=stable_logit(existence),
        visibility_logits=stable_logit(visibility),
    )


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def balanced_continuous_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float().clamp(0.0, 1.0)
    error = F.binary_cross_entropy_with_logits(
        logits.float(),
        target,
        reduction="none",
    )
    focal = (probability - target).abs().pow(gamma)
    positive = weight.float() * target
    negative = weight.float() * (1.0 - target)
    positive_loss = _weighted_mean(error * focal, positive)
    negative_loss = _weighted_mean(error * focal, negative)
    positive_valid = (positive.sum() > 0).to(error.dtype)
    negative_valid = (negative.sum() > 0).to(error.dtype)
    return (
        positive_loss * positive_valid + negative_loss * negative_valid
    ) / (positive_valid + negative_valid).clamp_min(1.0)


def object_lifecycle_loss(
    output: dict,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = (
        "predicted_future_survival_logits",
        "predicted_future_birth_logits",
        "predicted_future_observability_logits",
    )
    if any(name not in output for name in required):
        raise ValueError("factorized lifecycle outputs are missing")
    current = output.get(
        "target_history_track_presence",
        output["target_history_existence"],
    )[:, -1, None].detach().float()
    target_presence = output.get(
        "target_future_track_presence",
        output["target_future_existence"],
    ).detach().float()
    target_observation = output.get(
        "target_future_observation_confidence",
        output["target_future_visibility"],
    ).detach().float()
    target_in_frame = output["target_future_in_frame"].detach().float()
    horizon = output["future_horizon_valid"].detach().float()[..., None]
    survival_weight = current * horizon
    birth_weight = (1.0 - current) * horizon
    presence = balanced_continuous_focal_loss(
        output.get(
            "predicted_future_track_presence_logits",
            output["predicted_future_existence_logits"],
        ),
        target_presence,
        horizon,
        gamma,
    )
    survival = balanced_continuous_focal_loss(
        output["predicted_future_survival_logits"],
        target_presence,
        survival_weight,
        gamma,
    )
    birth = balanced_continuous_focal_loss(
        output["predicted_future_birth_logits"],
        target_presence,
        birth_weight,
        gamma,
    )
    conditional_observation = (
        target_observation
        / (target_presence * target_in_frame).clamp_min(1e-4)
    ).clamp(0.0, 1.0)
    observable_weight = target_presence * target_in_frame * horizon
    observability = balanced_continuous_focal_loss(
        output["predicted_future_observability_logits"],
        conditional_observation,
        observable_weight,
        gamma,
    )
    prediction = output.get(
        "predicted_future_track_presence",
        output["predicted_future_existence"],
    ).float()
    retention = _weighted_mean(
        F.relu(target_presence - prediction),
        survival_weight * target_presence,
    )
    hierarchy = F.relu(
        output["predicted_future_visibility"].float() - prediction
    ).mean()
    total = (
        presence
        + 0.5 * survival
        + 0.5 * birth
        + 0.5 * observability
        + retention
    )
    return total, {
        "geometry_track_presence": presence,
        "lifecycle_track_retention": survival,
        "lifecycle_track_discovery": birth,
        "lifecycle_observation": observability,
        "lifecycle_presence_retention_error": retention,
        "lifecycle_hierarchy_violation": hierarchy,
    }
