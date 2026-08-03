"""Held metrics for stable correspondence and calibrated track presence."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v41_held_metrics import _horizon_metrics


def add_v42_held_metrics(metrics, history_length: int, result: dict) -> None:
    values = _horizon_metrics(result)
    horizon_valid = result["future_horizon_valid"].float()
    for index, name in enumerate(("short", "goal")):
        valid = horizon_valid[:, index]
        for scope in (f"h{history_length}", "all"):
            prefix = f"v42/{scope}/{name}"
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
        metrics.add(f"v42/{scope}/association_row_mass_error", row_error)
        metrics.add(f"v42/{scope}/association_column_mass_error", column_error)
        metrics.add(f"v42/{scope}/association_entropy", entropy)
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
            metrics.add(f"v42/{scope}/identity_drift", drift, persistent)
            metrics.add(f"v42/{scope}/identity_unobserved_drift", drift, unobserved)


def v42_acceptance(stage: str, means: dict) -> dict[str, bool]:
    checks = {
        "association_mass_is_conserved": (
            means["v42/all/association_row_mass_error"] < 5e-4
            and means["v42/all/association_column_mass_error"] < 5e-4
        ),
    }
    horizons = ("short", "goal") if stage == "posterior" else ("short",)
    for horizon in horizons:
        prefix = f"v42/all/{horizon}"
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
