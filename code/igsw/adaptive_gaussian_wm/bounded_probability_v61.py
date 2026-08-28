"""Numerically bounded probability operations used by v61 objectives."""

from __future__ import annotations

import torch


def normalize_probability_mass(
    mass: torch.Tensor,
    dim: int,
    prior_mass: float,
) -> torch.Tensor:
    """Normalize non-negative mass with a fixed total uniform prior."""
    value = mass.float()
    count = value.shape[dim]
    prior = value.new_tensor(prior_mass / count)
    return (value + prior) / (
        value.sum(dim=dim, keepdim=True) + value.new_tensor(prior_mass)
    )


def smooth_binary_probability(
    probability: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Keep Bernoulli probabilities away from singular log boundaries."""
    value = probability.float().clamp(0.0, 1.0)
    return value * (1.0 - 2.0 * floor) + floor


def binary_probability_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    probability = smooth_binary_probability(prediction, floor)
    target = target.float()
    return -target * probability.log() - (1.0 - target) * torch.log1p(-probability)
