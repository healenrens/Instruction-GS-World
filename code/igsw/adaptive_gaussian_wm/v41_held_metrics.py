"""Held metrics for v41 correspondence, track presence, and relative transport."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .relative_transport import support_normalized_displacement


def _per_horizon_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(value).float()
    return (value.float() * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def _horizon_metrics(result: dict) -> dict[str, torch.Tensor]:
    online_center = result["online_history_centers"][:, -1, None].float()
    online_scale = result["online_history_relative_scale"][:, -1, None].float()
    target_center = result["target_history_centers"][:, -1, None].float()
    target_scale = result["target_history_relative_scale"][:, -1, None].float()
    target_transport = support_normalized_displacement(
        target_center,
        result["target_future_centers"].float(),
        target_scale,
    )
    prediction_transport = support_normalized_displacement(
        online_center,
        result["predicted_future_centers"].float(),
        online_scale,
    )
    transport_error = (
        prediction_transport - target_transport
    ).square().sum(-1).sqrt()
    persistence_transport = target_transport.square().sum(-1).sqrt()
    target_presence = result["target_future_track_presence"].float()
    current_presence = result[
        "target_history_track_presence"
    ][:, -1, None].float().expand_as(target_presence)
    predicted_presence = result["predicted_future_track_presence"].float()
    target_observation = result[
        "target_future_observation_confidence"
    ].float()
    current_observation = result[
        "target_history_observation_confidence"
    ][:, -1, None].float().expand_as(target_observation)
    predicted_observation = result["predicted_future_visibility"].float()
    object_weight = target_observation * current_observation
    return {
        "transport": _per_horizon_mean(transport_error, object_weight),
        "transport_persistence": _per_horizon_mean(
            persistence_transport,
            object_weight,
        ),
        "presence": (predicted_presence - target_presence).square().mean(-1),
        "presence_persistence": (
            current_presence - target_presence
        ).square().mean(-1),
        "presence_event": (current_presence - target_presence).abs().mean(-1),
        "observation": (
            predicted_observation - target_observation
        ).square().mean(-1),
        "observation_persistence": (
            current_observation - target_observation
        ).square().mean(-1),
        "observation_event": (
            current_observation - target_observation
        ).abs().mean(-1),
    }


def add_v41_held_metrics(metrics, history_length: int, result: dict) -> None:
    values = _horizon_metrics(result)
    horizon_valid = result["future_horizon_valid"].float()
    for index, name in enumerate(("short", "goal")):
        valid = horizon_valid[:, index]
        for scope in (f"h{history_length}", "all"):
            prefix = f"v41/{scope}/{name}"
            for metric_name, metric_value in values.items():
                metrics.add(
                    f"{prefix}_{metric_name}",
                    metric_value[:, index],
                    valid,
                )
    association = result["online_history_association"].float()
    unmatched = result["online_history_association_unmatched"].float()
    row_error = (
        association.sum(dim=-1) + unmatched - 1.0
    ).abs().amax(dim=(-1, -2))
    discovery = result["online_history_association_discovery"].float()
    column_error = (
        association.sum(dim=-2) + discovery - 1.0
    ).abs().amax(dim=(-1, -2))
    entropy = result["online_history_association_entropy"].float().mean(
        dim=(-1, -2)
    )
    for scope in (f"h{history_length}", "all"):
        metrics.add(f"v41/{scope}/association_row_mass_error", row_error)
        metrics.add(f"v41/{scope}/association_column_mass_error", column_error)
        metrics.add(f"v41/{scope}/association_entropy", entropy)
    if history_length > 1:
        keys = result["online_history_identity_keys"].float()
        drift = 1.0 - F.cosine_similarity(keys[:, 1:], keys[:, :-1], dim=-1)
        persistent = torch.minimum(
            result["online_history_track_presence"][:, 1:].float(),
            result["online_history_track_presence"][:, :-1].float(),
        )
        unobserved = persistent * (
            1.0 - result["online_history_observation_confidence"][:, 1:].float()
        )
        for scope in (f"h{history_length}", "all"):
            metrics.add(f"v41/{scope}/identity_drift", drift, persistent)
            metrics.add(f"v41/{scope}/identity_unobserved_drift", drift, unobserved)


def v41_acceptance(stage: str, means: dict) -> dict[str, bool]:
    checks = {
        "association_mass_is_conserved": (
            means["v41/all/association_row_mass_error"] < 1e-4
            and means["v41/all/association_column_mass_error"] < 1e-4
        ),
    }
    horizons = ("short", "goal") if stage == "posterior" else ("short",)
    for horizon in horizons:
        prefix = f"v41/all/{horizon}"
        checks[f"{horizon}_transport_beats_persistence"] = (
            means[f"{prefix}_transport"]
            < means[f"{prefix}_transport_persistence"]
        )
        checks[f"{horizon}_presence_beats_persistence"] = (
            means[f"{prefix}_presence"]
            < means[f"{prefix}_presence_persistence"]
        )
        checks[f"{horizon}_has_presence_change_evidence"] = (
            means[f"{prefix}_presence_event"] > 1e-4
        )
        checks[f"{horizon}_observation_beats_persistence"] = (
            means[f"{prefix}_observation"]
            < means[f"{prefix}_observation_persistence"]
        )
        checks[f"{horizon}_has_observation_change_evidence"] = (
            means[f"{prefix}_observation_event"] > 1e-4
        )
    return checks
