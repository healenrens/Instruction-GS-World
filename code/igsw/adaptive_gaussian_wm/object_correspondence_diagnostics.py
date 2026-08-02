"""Detached diagnostics for causal track-to-observation correspondence."""

from __future__ import annotations

import torch


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.float()
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(value)
    return (value.float() * weight).sum() / weight.sum().clamp_min(1.0)


def object_correspondence_diagnostics(output: dict) -> dict[str, torch.Tensor]:
    required = (
        "online_history_association",
        "online_history_association_match",
        "online_history_association_unmatched",
        "online_history_association_discovery",
        "online_history_association_entropy",
        "online_history_association_support_distance",
        "online_history_observation_confidence",
        "online_history_track_presence",
        "online_history_birth_evidence",
    )
    missing = [name for name in required if output.get(name) is None]
    if missing:
        raise ValueError(f"correspondence diagnostic outputs are missing: {missing}")
    association = output["online_history_association"].detach().float()
    match = output["online_history_association_match"].detach().float()
    unmatched = output["online_history_association_unmatched"].detach().float()
    discovery = output["online_history_association_discovery"].detach().float()
    entropy = output["online_history_association_entropy"].detach().float()
    distance = output[
        "online_history_association_support_distance"
    ].detach().float()
    observation = output[
        "online_history_observation_confidence"
    ].detach().float()
    presence = output["online_history_track_presence"].detach().float()
    birth = output["online_history_birth_evidence"].detach().float()
    row_error = (association.sum(dim=-1) + unmatched - 1.0).abs()
    column_error = (association.sum(dim=-2) + discovery - 1.0).abs()
    persistent_unobserved = presence * (1.0 - observation)
    return {
        "correspondence_match_probability_mean": _weighted_mean(match, presence),
        "correspondence_unmatched_probability_mean": _weighted_mean(
            unmatched,
            presence,
        ),
        "correspondence_discovery_probability_mean": discovery.mean(),
        "correspondence_entropy": _weighted_mean(entropy, observation),
        "correspondence_support_distance": _weighted_mean(distance, observation),
        "correspondence_row_mass_max_error": row_error.amax(),
        "correspondence_column_mass_max_error": column_error.amax(),
        "track_presence_mean": presence.mean(),
        "track_observation_confidence_mean": observation.mean(),
        "track_persistent_unobserved_rate": persistent_unobserved.mean(),
        "track_birth_evidence_mean": birth.mean(),
    }
