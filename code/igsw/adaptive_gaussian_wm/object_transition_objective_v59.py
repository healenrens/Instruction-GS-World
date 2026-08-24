"""Predictive and intervention objectives for object-level latent effects."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _prediction_errors(prediction, target, config):
    semantic = 1.0 - F.cosine_similarity(
        prediction.semantic.float(), target.future_semantic.float(), dim=-1
    )
    geometry_scale = prediction.geometry.new_tensor((2.0, 2.0, 8.0, 8.0, 8.0))
    geometry = F.smooth_l1_loss(
        prediction.geometry.float() * geometry_scale,
        target.future_geometry.float() * geometry_scale,
        reduction="none",
    ).mean(dim=-1)
    lifecycle = F.binary_cross_entropy_with_logits(
        prediction.visibility_logits.float(),
        target.future_visibility.float(),
        reduction="none",
    )
    total = (
        config.semantic_loss_weight * semantic
        + config.geometry_loss_weight * geometry
        + config.lifecycle_loss_weight * lifecycle
    )
    return total, semantic, geometry, lifecycle


def _persistence_errors(target, config):
    semantic = 1.0 - torch.einsum(
        "bd,bkd->bk", target.source_semantic.float(), target.future_semantic.float()
    )
    geometry_scale = target.future_geometry.new_tensor((2.0, 2.0, 8.0, 8.0, 8.0))
    source_geometry = target.source_geometry[:, None].expand_as(target.future_geometry)
    geometry = F.smooth_l1_loss(
        source_geometry.float() * geometry_scale,
        target.future_geometry.float() * geometry_scale,
        reduction="none",
    ).mean(dim=-1)
    source_visibility = target.source_visibility[:, None].expand_as(
        target.future_visibility
    )
    source_visibility_logits = torch.logit(
        source_visibility.float().clamp(1e-5, 1.0 - 1e-5)
    )
    lifecycle = F.binary_cross_entropy_with_logits(
        source_visibility_logits,
        target.future_visibility.float(),
        reduction="none",
    )
    total = (
        config.semantic_loss_weight * semantic
        + config.geometry_loss_weight * geometry
        + config.lifecycle_loss_weight * lifecycle
    )
    return total, semantic, geometry, lifecycle


def _relative_gain(correct, baseline, weight):
    correct_mean = _weighted_mean(correct, weight)
    baseline_mean = _weighted_mean(baseline, weight)
    return (baseline_mean - correct_mean) / baseline_mean.clamp_min(1e-6)


def object_transition_objective_v59(output, target, config):
    valid = target.pair_valid.float()
    active = target.motion_active.float()
    correct = _prediction_errors(output["correct"], target, config)
    zero = _prediction_errors(output["zero"], target, config)
    shuffled = _prediction_errors(output["shuffled"], target, config)
    persistence = _persistence_errors(target, config)

    prediction = _weighted_mean(correct[0], valid)
    source_target = type(
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
    zero_anchor = _weighted_mean(
        _prediction_errors(output["zero"], source_target, config)[0], valid
    )
    required_fraction = 1.0 - config.intervention_margin
    intervention = sum(
        _weighted_mean(
            F.relu(correct[0] - required_fraction * baseline.detach()), active
        )
        for baseline in (zero[0], shuffled[0], persistence[0])
    )

    effect = gather_batch_with_grad(output["effect"].flatten(0, 1))
    effect_std = effect.float().std(dim=0, correction=0)
    effect_variance = F.relu(config.minimum_effect_std - effect_std).mean()
    total = (
        prediction
        + config.zero_anchor_weight * zero_anchor
        + config.intervention_weight * intervention
        + config.effect_variance_weight * effect_variance
    )
    active_weight = active
    parts = {
        "total": total,
        "transition_prediction": prediction,
        "zero_effect_source_anchor": zero_anchor,
        "intervention_ranking": intervention,
        "effect_variance": effect_variance,
        "effect_std": effect_std.mean(),
        "effect_abs_mean": effect.float().abs().mean(),
        "transition_valid_fraction": valid.mean(),
        "motion_active_fraction": active.mean(),
        "correct_active_error": _weighted_mean(correct[0], active_weight),
        "zero_active_error": _weighted_mean(zero[0], active_weight),
        "shuffled_active_error": _weighted_mean(shuffled[0], active_weight),
        "persistence_active_error": _weighted_mean(persistence[0], active_weight),
        "correct_gain_over_zero": _relative_gain(correct[0], zero[0], active_weight),
        "correct_gain_over_shuffled": _relative_gain(
            correct[0], shuffled[0], active_weight
        ),
        "correct_gain_over_persistence": _relative_gain(
            correct[0], persistence[0], active_weight
        ),
        "correct_semantic_error": _weighted_mean(correct[1], valid),
        "correct_geometry_error": _weighted_mean(correct[2], valid),
        "correct_lifecycle_error": _weighted_mean(correct[3], valid),
        "effect_prediction_delta": (
            output["correct"].semantic.float() - output["zero"].semantic.float()
        )
        .square()
        .mean()
        .sqrt(),
    }
    for index, horizon in enumerate(config.dynamic_horizons):
        horizon_active = active[:, index]
        parts[f"h{horizon}_active_fraction"] = horizon_active.mean()
        parts[f"h{horizon}_gain_over_zero"] = _relative_gain(
            correct[0][:, index], zero[0][:, index], horizon_active
        )
        parts[f"h{horizon}_gain_over_shuffled"] = _relative_gain(
            correct[0][:, index], shuffled[0][:, index], horizon_active
        )
        parts[f"h{horizon}_gain_over_persistence"] = _relative_gain(
            correct[0][:, index], persistence[0][:, index], horizon_active
        )
    nonfinite = [
        name for name, value in parts.items() if not bool(torch.isfinite(value))
    ]
    if nonfinite:
        raise RuntimeError(
            "v59 objective contains non-finite terms: " + ", ".join(nonfinite)
        )
    return total, parts
