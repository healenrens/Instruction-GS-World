"""Paper-facing paired and action-aware statistics for goal evaluation."""
from __future__ import annotations

import math

import torch


ACTION_CHANNELS = {
    "all": slice(None),
    "center_0_3": slice(0, 3),
    "rgb_3_6": slice(3, 6),
}


_T95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _t95_critical(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("confidence interval requires positive degrees of freedom")
    if degrees_of_freedom <= len(_T95):
        return _T95[degrees_of_freedom - 1]
    if degrees_of_freedom <= 40:
        return 2.042
    if degrees_of_freedom <= 60:
        return 2.021
    if degrees_of_freedom <= 120:
        return 2.000
    return 1.980


def _cluster_means(
    values: torch.Tensor,
    cluster_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            values[cluster_ids == cluster].mean()
            for cluster in torch.unique(cluster_ids, sorted=True)
        ]
    )


def _cluster_interval(values: torch.Tensor) -> dict[str, float | bool | int]:
    if len(values) < 2:
        raise ValueError("cluster interval requires at least two clusters")
    mean = float(values.mean())
    standard_error = float(
        values.std(unbiased=True) / math.sqrt(len(values))
    )
    critical = _t95_critical(len(values) - 1)
    radius = critical * standard_error
    return {
        "clusters": len(values),
        "mean": mean,
        "cluster_standard_error": standard_error,
        "critical_value": critical,
        "ci95_lower": mean - radius,
        "ci95_upper": mean + radius,
        "positive_ci95_lower": mean - radius > 0.0,
    }


def paired_comparison(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    improvement = reference - prediction
    mean = float(improvement.mean())
    standard_error = float(
        improvement.std(unbiased=False) / math.sqrt(max(len(improvement), 1))
    )
    return {
        "absolute_improvement": mean,
        "relative_improvement": float(
            mean / reference.mean().clamp_min(1e-8)
        ),
        "prediction_win_fraction": float(
            (improvement > 0.0).float().mean()
        ),
        "paired_standard_error": standard_error,
        "positive_2se_margin": mean > 2.0 * standard_error,
    }


def clustered_paired_comparison(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    cluster_ids: torch.Tensor,
) -> dict[str, float | bool | int]:
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("clustered comparison requires aligned sample vectors")
    if cluster_ids.shape != prediction.shape:
        raise ValueError("cluster ids must align with sample errors")
    improvements = reference - prediction
    improvement = _cluster_means(improvements, cluster_ids)
    reference_mean = _cluster_means(reference, cluster_ids).mean()
    interval = _cluster_interval(improvement)
    return {
        "clusters": interval["clusters"],
        "absolute_improvement": interval["mean"],
        "relative_improvement": float(
            interval["mean"] / reference_mean.clamp_min(1e-8)
        ),
        "cluster_win_fraction": float(
            (improvement > 0.0).float().mean()
        ),
        "cluster_standard_error": interval["cluster_standard_error"],
        "critical_value": interval["critical_value"],
        "ci95_lower": interval["ci95_lower"],
        "ci95_upper": interval["ci95_upper"],
        "positive_ci95_lower": interval["positive_ci95_lower"],
    }


def clustered_relative_margin_test(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    cluster_ids: torch.Tensor,
    relative_margin: float,
) -> dict[str, float | bool | int]:
    if not 0.0 <= relative_margin < 1.0:
        raise ValueError("relative margin must be in [0,1)")
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("relative margin test requires aligned sample vectors")
    if cluster_ids.shape != prediction.shape:
        raise ValueError("cluster ids must align with sample errors")
    excess = reference - prediction - relative_margin * reference
    interval = _cluster_interval(_cluster_means(excess, cluster_ids))
    return {
        "clusters": interval["clusters"],
        "relative_margin": relative_margin,
        "mean_excess_over_margin": interval["mean"],
        "cluster_standard_error": interval["cluster_standard_error"],
        "critical_value": interval["critical_value"],
        "ci95_lower": interval["ci95_lower"],
        "ci95_upper": interval["ci95_upper"],
        "margin_ci95_passed": interval["positive_ci95_lower"],
    }


def clustered_relative_noninferiority_test(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    cluster_ids: torch.Tensor,
    relative_margin: float,
) -> dict[str, float | bool | int]:
    """Test whether prediction is less than margin worse than reference."""
    if relative_margin < 0.0:
        raise ValueError("relative noninferiority margin cannot be negative")
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("noninferiority test requires aligned sample vectors")
    if cluster_ids.shape != prediction.shape:
        raise ValueError("cluster ids must align with sample errors")
    surplus = (1.0 + relative_margin) * reference - prediction
    interval = _cluster_interval(_cluster_means(surplus, cluster_ids))
    return {
        "clusters": interval["clusters"],
        "relative_margin": relative_margin,
        "mean_noninferiority_surplus": interval["mean"],
        "cluster_standard_error": interval["cluster_standard_error"],
        "critical_value": interval["critical_value"],
        "ci95_lower": interval["ci95_lower"],
        "ci95_upper": interval["ci95_upper"],
        "noninferior_ci95": interval["positive_ci95_lower"],
    }


def goal_action_weight(
    history_activity: torch.Tensor,
    target_activity: torch.Tensor,
    action_shape: torch.Size,
    floor: float,
    power: float,
    enabled: bool,
) -> torch.Tensor:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("action activity floor must be in [0,1]")
    if power <= 0.0:
        raise ValueError("action activity power must be positive")
    if enabled:
        confidence = (
            history_activity[:, -1, None] * target_activity
        ).float().clamp(0.0, 1.0).pow(power)
        weight = floor + (1.0 - floor) * confidence
    else:
        weight = torch.ones_like(target_activity, dtype=torch.float32)
    if weight.shape != action_shape[:-1]:
        raise ValueError("action activity weight and action shape differ")
    return weight


def weighted_action_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("predicted and target actions must align")
    if prediction.shape[-1] < 6:
        raise ValueError("canonical action evaluation requires six channels")
    if weight.shape != prediction.shape[:-1]:
        raise ValueError("action weights must omit only the channel dimension")
    result = {}
    for name, channel in ACTION_CHANNELS.items():
        error = (
            prediction[..., channel].float()
            - target[..., channel].float()
        ).square().mean(dim=-1)
        result[name] = (error * weight).flatten(1).sum(dim=1) / (
            weight.flatten(1).sum(dim=1).clamp_min(1e-6)
        )
    if prediction.shape[-1] > 6:
        residual = (
            prediction[..., 6:].float() - target[..., 6:].float()
        ).square().mean(dim=-1)
        result["residual_6_plus"] = (
            (residual * weight).flatten(1).sum(dim=1)
            / weight.flatten(1).sum(dim=1).clamp_min(1e-6)
        )
    return result
