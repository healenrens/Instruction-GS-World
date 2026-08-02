"""Detached v40 diagnostics for identity, lifecycle, and relative transport."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .diagnostic_statistics import ratio_moments
from .relative_transport import support_normalized_displacement


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.float()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    return (value.float() * expanded).sum() / expanded.sum().clamp_min(1.0)


def _event_metrics(output: dict) -> dict[str, torch.Tensor]:
    presence = output.get(
        "predicted_future_track_presence",
        output["predicted_future_existence"],
    ).detach().float()
    visibility = output["predicted_future_visibility"].detach().float()
    in_frame = output["predicted_future_in_frame"].detach().float()
    survival = output["predicted_future_survival"].detach().float()
    birth = output["predicted_future_birth"].detach().float()
    observability = output["predicted_future_observability"].detach().float()
    target_presence = output.get(
        "target_future_track_presence",
        output["target_future_existence"],
    ).detach().float()
    target_observation = output.get(
        "target_future_observation_confidence",
        output["target_future_visibility"],
    ).detach().float()
    target_in_frame = output["target_future_in_frame"].detach().float()
    current = output.get(
        "target_history_track_presence",
        output["target_history_existence"],
    )[:, -1, None].detach().float()
    valid = output["future_horizon_valid"].detach().float()[..., None]
    valid = valid.expand_as(target_presence)
    current_positive = (current >= 0.5).expand_as(target_presence)
    target_positive = target_presence >= 0.5
    survival_positive = current_positive & target_positive
    birth_positive = (~current_positive) & target_positive
    persistent_absence = (~current_positive) & (~target_positive)
    survival_binary = survival >= 0.5
    birth_binary = birth >= 0.5
    result = {
        "lifecycle_survival_probability_mean": _weighted_mean(survival, valid),
        "lifecycle_birth_probability_mean": _weighted_mean(birth, valid),
        "lifecycle_observability_probability_mean": _weighted_mean(
            observability,
            valid,
        ),
        "lifecycle_visibility_above_track_presence_max": F.relu(
            visibility - presence
        ).amax(),
        "lifecycle_visibility_above_in_frame_max": F.relu(
            visibility - in_frame
        ).amax(),
        "lifecycle_observation_brier": _weighted_mean(
            (
                observability
                - (
                    target_observation
                    / (target_presence * target_in_frame).clamp_min(1e-4)
                ).clamp(0.0, 1.0)
            ).square(),
            valid * target_presence * target_in_frame,
        ),
    }
    for name, numerator, denominator in (
        (
            "lifecycle_track_retention_recall",
            survival_binary & survival_positive,
            survival_positive,
        ),
        (
            "lifecycle_track_discovery_recall",
            birth_binary & birth_positive,
            birth_positive,
        ),
        (
            "lifecycle_inactive_capacity_recall",
            (~birth_binary) & persistent_absence,
            persistent_absence,
        ),
    ):
        result.update(
            ratio_moments(
                name,
                (numerator.float() * valid).sum(),
                (denominator.float() * valid).sum(),
            )
        )
    return result


def _identity_metrics(output: dict) -> dict[str, torch.Tensor]:
    keys = output["online_history_identity_keys"].detach().float()
    visibility = output["online_history_visibility"].detach().float()
    presence = output.get(
        "online_history_track_presence",
        output["online_history_existence"],
    ).detach().float()
    similarity = output["online_history_identity_similarity"].detach().float()
    result = {
        "identity_visible_correction_similarity": _weighted_mean(
            similarity,
            visibility,
        ),
    }
    if keys.shape[1] == 1:
        zero = keys.sum() * 0.0
        result.update(
            identity_temporal_cosine_drift=zero,
            identity_occluded_cosine_drift=zero,
        )
        return result
    drift = 1.0 - F.cosine_similarity(keys[:, 1:], keys[:, :-1], dim=-1)
    persistent = torch.minimum(presence[:, 1:], presence[:, :-1])
    result.update(
        identity_temporal_cosine_drift=_weighted_mean(drift, persistent),
        identity_occluded_cosine_drift=_weighted_mean(
            drift,
            persistent * (1.0 - visibility[:, 1:]),
        ),
    )
    return result


def _transport_metrics(output: dict) -> dict[str, torch.Tensor]:
    online_center = output["online_history_centers"][:, -1, None].detach().float()
    online_scale = (
        output["online_history_relative_scale"][:, -1, None].detach().float()
    )
    target_center = output["target_history_centers"][:, -1, None].detach().float()
    target_scale = (
        output["target_history_relative_scale"][:, -1, None].detach().float()
    )
    prediction = output["predicted_future_centers"].detach().float()
    target = output["target_future_centers"].detach().float()
    predicted_transport = support_normalized_displacement(
        online_center,
        prediction,
        online_scale,
    )
    target_transport = support_normalized_displacement(
        target_center,
        target,
        target_scale,
    )
    weight = (
        output["target_history_visibility"][:, -1, None].detach().float()
        * output["target_future_visibility"].detach().float()
        * output["future_horizon_valid"].detach().float()[..., None]
    )
    prediction_error = (predicted_transport - target_transport).square().sum(-1).sqrt()
    persistence_error = target_transport.square().sum(-1).sqrt()
    return {
        "transport_support_error": _weighted_mean(prediction_error, weight),
        "transport_persistence_error": _weighted_mean(persistence_error, weight),
        "transport_absolute_gain_over_persistence": _weighted_mean(
            persistence_error - prediction_error,
            weight,
        ),
        "transport_units_saturation_rate": (
            output["predicted_future_transport_units"].detach().float().abs()
            > 0.95 * float(output["transport_max_support_units"])
        ).float().mean(),
    }


def lifecycle_transport_diagnostics(output: dict) -> dict[str, torch.Tensor]:
    required = (
        "predicted_future_survival",
        "predicted_future_birth",
        "predicted_future_observability",
        "predicted_future_transport_units",
        "online_history_identity_keys",
    )
    missing = [name for name in required if output.get(name) is None]
    if missing:
        raise ValueError(f"v40 diagnostic outputs are missing: {missing}")
    result = _event_metrics(output)
    result.update(_identity_metrics(output))
    result.update(_transport_metrics(output))
    return result
