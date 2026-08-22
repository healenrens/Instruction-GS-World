"""Metrics aligned with the explicit v56 Object State contract."""

from __future__ import annotations

import torch

from .object_state_target_v52 import component_motion_targets


def aligned_state_probe_tensors(
    mapped_dynamic: torch.Tensor,
    output: dict,
    groups: torch.Tensor,
    motion_active_threshold: float,
) -> dict[str, torch.Tensor]:
    teacher = output["teacher"]
    prediction = output["prediction"]
    target_motion, motion_weight, _ = component_motion_targets(teacher)
    horizons = target_motion.shape[-2]
    motion_group = groups[:, None, None, None].expand_as(motion_weight)
    active = target_motion.norm(dim=-1) >= motion_active_threshold

    lifecycle_weight = (
        teacher.lifecycle_known.float() * teacher.object_confidence[:, None]
    )
    lifecycle_group = groups[:, None, None].expand_as(teacher.visibility)
    return {
        "component_motion_feature": mapped_dynamic[:, :, :, None]
        .expand(-1, -1, -1, horizons, -1)
        .reshape(-1, mapped_dynamic.shape[-1])
        .detach()
        .cpu(),
        "component_motion_prediction": prediction.motion.reshape(-1, 2)
        .detach()
        .cpu(),
        "component_motion_target": target_motion.reshape(-1, 2).detach().cpu(),
        "component_motion_weight": motion_weight.reshape(-1).detach().cpu(),
        "component_motion_active_weight": (
            motion_weight * active.float()
        ).reshape(-1).detach().cpu(),
        "component_motion_group": motion_group.reshape(-1).detach().cpu(),
        "visibility_state_prediction": prediction.visibility.reshape(-1)
        .detach()
        .cpu(),
        "visibility_state_target": teacher.visibility.reshape(-1).float()
        .detach()
        .cpu(),
        "visibility_state_weight": lifecycle_weight.reshape(-1).detach().cpu(),
        "visibility_state_group": lifecycle_group.reshape(-1).detach().cpu(),
        "presence_state_prediction": prediction.presence.reshape(-1).detach().cpu(),
        "presence_state_target": teacher.presence.reshape(-1).float().detach().cpu(),
        "presence_state_weight": lifecycle_weight.reshape(-1).detach().cpu(),
    }


def _held_group_masks(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique = groups.unique(sorted=True)
    if len(unique) < 2:
        raise ValueError("aligned state evaluation needs at least two episode groups")
    train_groups = unique[::2]
    train = (groups[:, None] == train_groups[None]).any(dim=1)
    return train, ~train


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def held_group_vector_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    groups: torch.Tensor,
) -> dict[str, float]:
    keep = weight > 1e-4
    prediction, target = prediction[keep].float(), target[keep].float()
    weight, groups = weight[keep].float(), groups[keep]
    if len(target) < 16:
        raise ValueError("aligned motion evaluation needs at least 16 valid vectors")
    train, test = _held_group_masks(groups)
    train_weight, test_weight = weight[train], weight[test]
    train_mean = (target[train] * train_weight[:, None]).sum(dim=0)
    train_mean = train_mean / train_weight.sum().clamp_min(1e-8)
    error = (prediction[test] - target[test]).square().mean(dim=-1)
    mean_error = (target[test] - train_mean).square().mean(dim=-1)
    zero_error = target[test].square().mean(dim=-1)
    mse = _weighted_mean(error, test_weight)
    mean_mse = _weighted_mean(mean_error, test_weight).clamp_min(1e-8)
    zero_mse = _weighted_mean(zero_error, test_weight).clamp_min(1e-8)
    return {
        "mse": float(mse),
        "mean_baseline_mse": float(mean_mse),
        "mean_relative_gain": float((mean_mse - mse) / mean_mse),
        "zero_baseline_mse": float(zero_mse),
        "zero_relative_gain": float((zero_mse - mse) / zero_mse),
        "target_rms": float(_weighted_mean(target[test].square().mean(-1), test_weight).sqrt()),
        "prediction_rms": float(
            _weighted_mean(prediction[test].square().mean(-1), test_weight).sqrt()
        ),
        "vector_count": float(test.sum()),
    }


def held_group_binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    groups: torch.Tensor,
) -> dict[str, float]:
    keep = weight > 1e-4
    prediction = prediction[keep].float().clamp(0.0, 1.0)
    target, weight, groups = target[keep].float(), weight[keep].float(), groups[keep]
    if len(target) < 16:
        raise ValueError("visibility evaluation needs at least 16 known states")
    train, test = _held_group_masks(groups)
    prevalence = _weighted_mean(target[train], weight[train])
    brier = _weighted_mean((prediction[test] - target[test]).square(), weight[test])
    baseline = _weighted_mean(
        (prevalence - target[test]).square(), weight[test]
    ).clamp_min(1e-8)
    positive = weight[test] * (target[test] >= 0.5).float()
    negative = weight[test] * (target[test] < 0.5).float()
    if float(positive.sum()) == 0.0 or float(negative.sum()) == 0.0:
        raise ValueError("visibility evaluation requires visible and occluded states")
    classified = prediction[test] >= 0.5
    visible_recall = _weighted_mean(classified.float(), positive)
    occluded_recall = _weighted_mean((~classified).float(), negative)
    return {
        "brier": float(brier),
        "constant_brier": float(baseline),
        "brier_relative_gain": float((baseline - brier) / baseline),
        "balanced_accuracy": float(0.5 * (visible_recall + occluded_recall)),
        "visible_recall": float(visible_recall),
        "occluded_recall": float(occluded_recall),
        "known_count": float(test.sum()),
        "visible_weight": float(positive.sum()),
        "occluded_weight": float(negative.sum()),
    }


def finalize_aligned_state_probes(probes, ridge_relative_gain):
    result = {}
    motion = held_group_vector_metrics(
        probes["component_motion_prediction"],
        probes["component_motion_target"],
        probes["component_motion_weight"],
        probes["component_motion_group"],
    )
    active_motion = held_group_vector_metrics(
        probes["component_motion_prediction"],
        probes["component_motion_target"],
        probes["component_motion_active_weight"],
        probes["component_motion_group"],
    )
    result.update({f"component_motion_readout_{name}": value for name, value in motion.items()})
    result.update({
        f"component_motion_active_readout_{name}": value
        for name, value in active_motion.items()
    })
    result["component_motion_probe_relative_gain"] = ridge_relative_gain(
        probes["component_motion_feature"],
        probes["component_motion_target"],
        probes["component_motion_weight"],
        probes["component_motion_group"],
    )
    result["component_motion_active_probe_relative_gain"] = ridge_relative_gain(
        probes["component_motion_feature"],
        probes["component_motion_target"],
        probes["component_motion_active_weight"],
        probes["component_motion_group"],
    )
    result["component_motion_active_fraction"] = float(
        (probes["component_motion_active_weight"] > 1e-4).sum()
        / (probes["component_motion_weight"] > 1e-4).sum().clamp_min(1)
    )
    visibility = held_group_binary_metrics(
        probes["visibility_state_prediction"],
        probes["visibility_state_target"],
        probes["visibility_state_weight"],
        probes["visibility_state_group"],
    )
    result.update({f"explicit_visibility_{name}": value for name, value in visibility.items()})
    presence_weight = probes["presence_state_weight"].float()
    presence_target = probes["presence_state_target"].float()
    known = presence_weight > 1e-4
    result["presence_known_positive_count"] = float(
        (known & (presence_target >= 0.5)).sum()
    )
    result["presence_known_negative_count"] = float(
        (known & (presence_target < 0.5)).sum()
    )
    result["presence_supervision_identifiable"] = float(
        result["presence_known_positive_count"] > 0
        and result["presence_known_negative_count"] > 0
    )
    return result
