"""Detached diagnostics for full-DINO change-only object readout."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .change_readout_objective import relative_change_targets


CHANGE_READOUT_REQUIRED_DIAGNOSTICS = frozenset(
    {
        "readout_change_feature",
        "readout_change_persistence_feature",
        "readout_change_gain_over_persistence",
        "readout_static_feature",
        "readout_static_persistence_feature",
        "readout_static_gain_over_persistence",
        "readout_residual_change_rms",
        "readout_residual_static_rms",
        "readout_background_fraction",
        "readout_potential_change_fraction",
        "readout_active_change_fraction",
        "readout_partition_sum_max_error",
        "readout_assignment_shift",
        "readout_current_identity_max_error",
    }
)


def validate_change_readout_diagnostic_contract(metrics: dict[str, float]) -> None:
    missing = CHANGE_READOUT_REQUIRED_DIAGNOSTICS.difference(metrics)
    if missing:
        raise ValueError(f"missing change readout diagnostics: {sorted(missing)}")
    nonfinite = {
        name
        for name in CHANGE_READOUT_REQUIRED_DIAGNOSTICS
        if not math.isfinite(metrics[name])
    }
    if nonfinite:
        raise ValueError(f"non-finite change readout diagnostics: {sorted(nonfinite)}")
    fractions = (
        "readout_background_fraction",
        "readout_potential_change_fraction",
        "readout_active_change_fraction",
    )
    invalid = {name for name in fractions if not 0.0 <= metrics[name] <= 1.0}
    if invalid:
        raise ValueError(f"invalid change readout fractions: {sorted(invalid)}")


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _feature_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1
    )
    return mse + 0.1 * cosine


def _weighted_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _weighted_mean(value.float().square().mean(dim=-1), weight).sqrt()


@torch.no_grad()
def change_residual_readout_diagnostics(
    batch: dict[str, torch.Tensor],
    output: dict,
) -> dict[str, torch.Tensor]:
    state = output.get("change_future_readout")
    reference = output.get("change_reference_readout")
    if state is None or reference is None:
        raise ValueError("change diagnostics require future and reference states")
    prediction = output["rendered_future_features"].detach().float()
    target = batch["future_features"].detach().float()
    current = batch["history_features"][:, -1:].detach().float().expand_as(target)
    active_target, _, _ = relative_change_targets(batch)
    valid = batch["future_valid"].float()
    change_weight = active_target * valid
    static_weight = (1.0 - active_target) * valid
    predicted_error = _feature_error(prediction, target)
    persistence_error = _feature_error(current, target)
    change_feature = _weighted_mean(predicted_error, change_weight)
    change_persistence = _weighted_mean(persistence_error, change_weight)
    static_feature = _weighted_mean(predicted_error, static_weight)
    static_persistence = _weighted_mean(persistence_error, static_weight)
    residual = prediction - current

    slots = output["history_slot_states"][-1]
    tokens = output["history_token_states"][-1]
    token_weight = tokens.activation.squeeze(-1).detach().float()
    denominator = token_weight.sum().clamp_min(1.0)
    background_fraction = (
        slots.background_assignment.detach().float() * token_weight
    ).sum() / denominator
    potential_fraction = (
        slots.potential_change.detach().float() * token_weight
    ).sum() / denominator
    partition_sum = slots.assignment.detach().float().sum(dim=-1) + (
        slots.background_assignment.detach().float()
    )
    assignment_shift = (
        state.destination_assignment.detach().float()
        - reference.destination_assignment.detach().float()
    ).abs()
    assignment_shift = _weighted_mean(
        assignment_shift.mean(dim=2), batch["future_valid"].float()
    )
    identity = output["residual_reference_features"].detach().float()
    identity_target = batch["history_features"][:, -1:].detach().float().expand_as(
        identity
    )
    return {
        "readout_change_feature": change_feature,
        "readout_change_persistence_feature": change_persistence,
        "readout_change_gain_over_persistence": change_persistence - change_feature,
        "readout_static_feature": static_feature,
        "readout_static_persistence_feature": static_persistence,
        "readout_static_gain_over_persistence": static_persistence - static_feature,
        "readout_residual_change_rms": _weighted_rms(residual, change_weight),
        "readout_residual_static_rms": _weighted_rms(residual, static_weight),
        "readout_background_fraction": background_fraction,
        "readout_potential_change_fraction": potential_fraction,
        "readout_active_change_fraction": state.active_change_map.detach().float().mean(),
        "readout_partition_sum_max_error": (partition_sum - 1.0).abs().max(),
        "readout_assignment_shift": assignment_shift,
        "readout_current_identity_max_error": (identity - identity_target).abs().max(),
    }
