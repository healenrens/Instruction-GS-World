"""Held metrics and promotion checks for v40 object transport and lifecycle."""

from __future__ import annotations

import torch

from .relative_transport import support_normalized_displacement


def _per_sample_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(value).float()
    axes = tuple(range(1, value.ndim))
    return (value.float() * weight).sum(dim=axes) / weight.sum(dim=axes).clamp_min(1.0)


def add_v40_held_metrics(metrics, history_length: int, result: dict) -> None:
    online_center = result["online_history_centers"][:, -1, None].float()
    online_scale = result["online_history_relative_scale"][:, -1, None].float()
    target_center = result["target_history_centers"][:, -1, None].float()
    target_scale = result["target_history_relative_scale"][:, -1, None].float()
    target_transport = support_normalized_displacement(
        target_center,
        result["target_future_centers"].float(),
        target_scale,
    )
    predicted_transport = support_normalized_displacement(
        online_center,
        result["predicted_future_centers"].float(),
        online_scale,
    )
    transport_error = (predicted_transport - target_transport).square().sum(-1).sqrt()
    persistence_transport = target_transport.square().sum(-1).sqrt()
    target_existence = result["target_future_existence"].float()
    current_existence = result["target_history_existence"][:, -1, None].float()
    current_existence = current_existence.expand_as(target_existence)
    predicted_existence = result["predicted_future_existence"].float()
    target_visibility = result["target_future_visibility"].float()
    current_visibility = result["target_history_visibility"][:, -1, None].float()
    current_visibility = current_visibility.expand_as(target_visibility)
    predicted_visibility = result["predicted_future_visibility"].float()
    horizon_valid = result["future_horizon_valid"].float()
    object_weight = target_visibility * current_visibility
    names = ("short", "goal")
    for index, name in enumerate(names):
        valid = horizon_valid[:, index]
        transport = _per_sample_mean(
            transport_error[:, index],
            object_weight[:, index],
        )
        persistence = _per_sample_mean(
            persistence_transport[:, index],
            object_weight[:, index],
        )
        existence = (predicted_existence[:, index] - target_existence[:, index]).square().mean(-1)
        existence_persistence = (
            current_existence[:, index] - target_existence[:, index]
        ).square().mean(-1)
        visibility = (
            predicted_visibility[:, index] - target_visibility[:, index]
        ).square().mean(-1)
        visibility_persistence = (
            current_visibility[:, index] - target_visibility[:, index]
        ).square().mean(-1)
        for scope in (f"h{history_length}", "all"):
            prefix = f"v40/{scope}/{name}"
            metrics.add(f"{prefix}_transport", transport, valid)
            metrics.add(f"{prefix}_transport_persistence", persistence, valid)
            metrics.add(f"{prefix}_existence_brier", existence, valid)
            metrics.add(
                f"{prefix}_existence_persistence_brier",
                existence_persistence,
                valid,
            )
            metrics.add(f"{prefix}_visibility_brier", visibility, valid)
            metrics.add(
                f"{prefix}_visibility_persistence_brier",
                visibility_persistence,
                valid,
            )
    if history_length > 1:
        keys = result["online_history_identity_keys"].float()
        drift = 1.0 - torch.nn.functional.cosine_similarity(
            keys[:, 1:],
            keys[:, :-1],
            dim=-1,
        )
        persistent = torch.minimum(
            result["online_history_existence"][:, 1:].float(),
            result["online_history_existence"][:, :-1].float(),
        )
        occluded = persistent * (
            1.0 - result["online_history_visibility"][:, 1:].float()
        )
        for scope in (f"h{history_length}", "all"):
            metrics.add(f"v40/{scope}/identity_drift", drift, persistent)
            metrics.add(f"v40/{scope}/identity_occluded_drift", drift, occluded)


def v40_acceptance(stage: str, means: dict) -> dict[str, bool]:
    checks = {}
    horizons = ("short", "goal") if stage == "posterior" else ("short",)
    for horizon in horizons:
        prefix = f"v40/all/{horizon}"
        checks[f"{horizon}_transport_beats_persistence"] = (
            means[f"{prefix}_transport"]
            < means[f"{prefix}_transport_persistence"]
        )
        checks[f"{horizon}_existence_beats_persistence"] = (
            means[f"{prefix}_existence_brier"]
            <= means[f"{prefix}_existence_persistence_brier"]
        )
        checks[f"{horizon}_visibility_beats_persistence"] = (
            means[f"{prefix}_visibility_brier"]
            <= means[f"{prefix}_visibility_persistence_brier"]
        )
    return checks
