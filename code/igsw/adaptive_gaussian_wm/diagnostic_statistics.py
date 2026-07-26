"""Sufficient-statistic reduction for distributed training diagnostics."""
from __future__ import annotations

import math

import torch


_CORRELATION_PREFIX = "__diagnostic_correlation__"
_RATIO_PREFIX = "__diagnostic_ratio__"


def correlation_moments(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"diagnostic correlation {name} has mismatched shapes")
    prefix = f"{_CORRELATION_PREFIX}{name}__"
    return {
        f"{prefix}count": left.new_tensor(float(left.numel())),
        f"{prefix}left": left.sum(),
        f"{prefix}right": right.sum(),
        f"{prefix}left_square": left.square().sum(),
        f"{prefix}right_square": right.square().sum(),
        f"{prefix}product": (left * right).sum(),
    }


def ratio_moments(
    name: str,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prefix = f"{_RATIO_PREFIX}{name}__"
    return {
        f"{prefix}numerator": numerator.detach().float().sum(),
        f"{prefix}denominator": denominator.detach().float().sum(),
    }


def _split_raw_metric(name: str, prefix: str) -> tuple[str, str]:
    payload = name[len(prefix) :]
    if "__" not in payload:
        raise ValueError(f"malformed diagnostic metric: {name}")
    return payload.rsplit("__", 1)


def finalize_diagnostic_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Resolve reduced sufficient statistics into W&B-ready scalars."""
    result = dict(metrics)
    correlations: dict[str, dict[str, float]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for key in tuple(result):
        if key.startswith(_CORRELATION_PREFIX):
            group, field = _split_raw_metric(key, _CORRELATION_PREFIX)
            correlations.setdefault(group, {})[field] = result.pop(key)
        elif key.startswith(_RATIO_PREFIX):
            group, field = _split_raw_metric(key, _RATIO_PREFIX)
            ratios.setdefault(group, {})[field] = result.pop(key)

    for name, values in correlations.items():
        required = {
            "count",
            "left",
            "right",
            "left_square",
            "right_square",
            "product",
        }
        missing = required.difference(values)
        if missing:
            raise ValueError(f"diagnostic correlation {name} is missing {missing}")
        count = values["count"]
        left_mean = values["left"] / max(count, 1.0)
        right_mean = values["right"] / max(count, 1.0)
        left_variance = max(
            values["left_square"] / max(count, 1.0) - left_mean**2,
            0.0,
        )
        right_variance = max(
            values["right_square"] / max(count, 1.0) - right_mean**2,
            0.0,
        )
        valid = count > 1.0 and left_variance > 1e-12 and right_variance > 1e-12
        covariance = values["product"] / max(count, 1.0) - left_mean * right_mean
        result[name] = (
            covariance / math.sqrt(left_variance * right_variance) if valid else 0.0
        )
        result[f"{name}_valid"] = float(valid)

    for name, values in ratios.items():
        if set(values) != {"numerator", "denominator"}:
            raise ValueError(f"diagnostic ratio {name} is incomplete")
        denominator = values["denominator"]
        valid = abs(denominator) > 1e-12
        result[name] = values["numerator"] / denominator if valid else 0.0
        result[f"{name}_valid"] = float(valid)
    return result
