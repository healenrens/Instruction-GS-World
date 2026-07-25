"""Per-sample metrics for the deployable latent-action Prior gate."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .rgb_supervision import rgb_reconstruction_loss


def weighted_sample_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    expanded = weight.to(value.dtype)
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    numerator = (value * expanded).flatten(1).sum(dim=1)
    denominator = expanded.expand_as(value).flatten(1).sum(dim=1)
    return numerator / denominator.clamp_min(1e-6)


def feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return weighted_sample_mean(
        (prediction - target).square().mean(dim=-1),
        valid,
    )


def latent_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    return weighted_sample_mean(
        (
            F.normalize(prediction, dim=-1)
            - F.normalize(target, dim=-1)
        ).square(),
        activity,
    )


def rgb_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    return torch.stack(
        [
            rgb_reconstruction_loss(
                prediction[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                ssim_weight,
            )[0]
            for index in range(len(prediction))
        ]
    )


def paired_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    improvement = reference - candidate
    mean = improvement.mean()
    standard_error = improvement.std(unbiased=False) / math.sqrt(
        max(len(improvement), 1)
    )
    return {
        "absolute_improvement": float(mean),
        "relative_improvement": float(
            mean / reference.mean().clamp_min(1e-8)
        ),
        "candidate_win_fraction": float((improvement > 0.0).float().mean()),
        "paired_standard_error": float(standard_error),
        "positive_2se_margin": bool(mean > 2.0 * standard_error),
    }
