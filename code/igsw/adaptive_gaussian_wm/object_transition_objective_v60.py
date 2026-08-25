"""Magnitude-aware objective for gated residual object transitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad
from .object_transition_objective_v59 import (
    persistence_transition_errors_v59,
    transition_prediction_errors_v59,
)


BASELINES = ("zero", "shuffled", "persistence")


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _gain(correct, baseline, weight):
    correct_mean = _weighted_mean(correct, weight)
    baseline_mean = _weighted_mean(baseline, weight)
    return (baseline_mean - correct_mean) / baseline_mean.clamp_min(1e-6)


def _source_target(target):
    return type(
        "SourceTarget",
        (),
        {
            "future_semantic": target.source_semantic[:, None].expand_as(
                target.future_semantic
            ),
            "future_geometry": target.source_geometry[:, None].expand_as(
                target.future_geometry
            ),
            "future_visibility": target.source_visibility[:, None].expand_as(
                target.future_visibility
            ),
        },
    )()


def _correlation(first, second, weight):
    mass = weight.sum().clamp_min(1.0)
    first_mean = (first * weight).sum() / mass
    second_mean = (second * weight).sum() / mass
    first_centered = (first - first_mean) * weight.sqrt()
    second_centered = (second - second_mean) * weight.sqrt()
    covariance = (first_centered * second_centered).sum()
    denominator = (
        first_centered.square().sum() * second_centered.square().sum()
    ).sqrt()
    return covariance / denominator.clamp_min(1e-6)


def object_transition_objective_v60(output, target, config):
    valid = target.pair_valid.float()
    active = target.motion_active.float()
    strength = target.change_strength.float() * valid
    low_change = valid * (strength <= config.low_change_threshold).float()
    high_change = valid * (strength >= config.high_change_threshold).float()
    errors = {
        "correct": transition_prediction_errors_v59(output["correct"], target, config),
        "zero": transition_prediction_errors_v59(output["zero"], target, config),
        "shuffled": transition_prediction_errors_v59(
            output["shuffled"], target, config
        ),
        "persistence": persistence_transition_errors_v59(target, config),
    }
    source = _source_target(target)
    base_source = transition_prediction_errors_v59(output["base"], source, config)[0]
    correct_source = transition_prediction_errors_v59(
        output["correct"], source, config
    )[0]

    prediction = _weighted_mean(errors["correct"][0], valid)
    base_anchor = _weighted_mean(base_source, valid)
    gate_calibration = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output["change_gate_logits"].float(),
            target.change_strength.float(),
            reduction="none",
        ),
        valid,
    )
    no_change_consistency = _weighted_mean(correct_source, valid * (1.0 - strength))
    required_fraction = 1.0 - config.intervention_margin * strength
    intervention = sum(
        _weighted_mean(
            F.relu(
                errors["correct"][0] - required_fraction * errors[baseline][0].detach()
            ),
            strength,
        )
        for baseline in BASELINES
    )

    effect = gather_batch_with_grad(output["effect"].flatten(0, 1))
    effect_std = effect.float().std(dim=0, correction=0)
    effect_variance = F.relu(config.minimum_effect_std - effect_std).mean()
    total = (
        prediction
        + config.base_anchor_weight * base_anchor
        + config.gate_calibration_weight * gate_calibration
        + config.no_change_consistency_weight * no_change_consistency
        + config.intervention_weight * intervention
        + config.effect_variance_weight * effect_variance
    )

    gate = output["change_gate"].float()
    residual = (
        1.0
        - F.cosine_similarity(
            output["correct"].semantic.float(), output["base"].semantic.float(), dim=-1
        )
    ) + F.smooth_l1_loss(
        output["correct"].geometry.float(),
        output["base"].geometry.float(),
        reduction="none",
    ).mean(dim=-1)
    parts = {
        "total": total,
        "transition_prediction": prediction,
        "base_source_anchor": base_anchor,
        "gate_calibration": gate_calibration,
        "no_change_consistency": no_change_consistency,
        "intervention_ranking": intervention,
        "effect_variance": effect_variance,
        "effect_std": effect_std.mean(),
        "effect_abs_mean": effect.float().abs().mean(),
        "change_strength_mean": _weighted_mean(strength, valid),
        "change_gate_mean": _weighted_mean(gate, valid),
        "change_gate_mae": _weighted_mean(
            (gate - target.change_strength.float()).abs(), valid
        ),
        "change_gate_correlation": _correlation(
            gate, target.change_strength.float(), valid
        ),
        "low_change_fraction": low_change.mean(),
        "high_change_fraction": high_change.mean(),
        "residual_magnitude": _weighted_mean(residual, valid),
        "low_change_residual_magnitude": _weighted_mean(residual, low_change),
        "transition_valid_fraction": valid.mean(),
        "motion_active_fraction": active.mean(),
    }
    for name, values in errors.items():
        parts[f"{name}_active_error"] = _weighted_mean(values[0], active)
        parts[f"{name}_change_weighted_error"] = _weighted_mean(values[0], strength)
        parts[f"{name}_low_change_error"] = _weighted_mean(values[0], low_change)
        parts[f"{name}_high_change_error"] = _weighted_mean(values[0], high_change)
    for baseline in BASELINES:
        parts[f"correct_gain_over_{baseline}"] = _gain(
            errors["correct"][0], errors[baseline][0], active
        )
        parts[f"change_weighted_gain_over_{baseline}"] = _gain(
            errors["correct"][0], errors[baseline][0], strength
        )
        parts[f"low_change_gain_over_{baseline}"] = _gain(
            errors["correct"][0], errors[baseline][0], low_change
        )
        parts[f"high_change_gain_over_{baseline}"] = _gain(
            errors["correct"][0], errors[baseline][0], high_change
        )
    parts.update(
        {
            "correct_semantic_error": _weighted_mean(errors["correct"][1], valid),
            "correct_geometry_error": _weighted_mean(errors["correct"][2], valid),
            "correct_lifecycle_error": _weighted_mean(errors["correct"][3], valid),
            "effect_prediction_delta": (
                output["correct"].semantic.float() - output["zero"].semantic.float()
            )
            .square()
            .mean()
            .sqrt(),
        }
    )
    for index, horizon in enumerate(config.dynamic_horizons):
        horizon_active = active[:, index]
        horizon_strength = strength[:, index]
        parts[f"h{horizon}_active_fraction"] = horizon_active.mean()
        parts[f"h{horizon}_change_strength"] = _weighted_mean(
            target.change_strength[:, index], valid[:, index]
        )
        parts[f"h{horizon}_gate_mae"] = _weighted_mean(
            (gate[:, index] - target.change_strength[:, index]).abs(), valid[:, index]
        )
        for baseline in BASELINES:
            parts[f"h{horizon}_gain_over_{baseline}"] = _gain(
                errors["correct"][0][:, index],
                errors[baseline][0][:, index],
                horizon_active,
            )
            parts[f"h{horizon}_change_weighted_gain_over_{baseline}"] = _gain(
                errors["correct"][0][:, index],
                errors[baseline][0][:, index],
                horizon_strength,
            )
    nonfinite = [
        name for name, value in parts.items() if not bool(torch.isfinite(value))
    ]
    if nonfinite:
        raise RuntimeError(
            "v60 objective contains non-finite terms: " + ", ".join(nonfinite)
        )
    return total, parts
