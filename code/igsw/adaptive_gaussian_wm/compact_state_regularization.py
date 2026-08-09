"""Cross-rank anti-collapse statistics over supported compact features."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import (
    gather_batch_with_grad,
    gather_batch_without_grad,
)


def feature_statistics_regularizer(
    feature: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if valid.shape != feature.shape[:-1]:
        raise ValueError("feature regularizer validity shape differs")
    global_feature = gather_batch_with_grad(feature).float()
    global_valid = gather_batch_without_grad(valid).bool()
    if not bool(torch.isfinite(global_feature).all()):
        raise ValueError("feature regularizer received non-finite features")
    supported = global_feature.reshape(-1, feature.shape[-1])[
        global_valid.reshape(-1)
    ]
    if supported.shape[0] < 2:
        # Covariance is undefined; retain a finite maximum variance penalty.
        zero = global_feature.sum() * 0.0
        return zero + 0.5, zero, zero + 0.2
    standard_deviation = supported.std(dim=0, unbiased=False)
    variance = F.relu(0.5 - standard_deviation).mean()
    sampled = supported[:, ::4]
    sampled = sampled - sampled.mean(dim=0, keepdim=True)
    sampled_deviation = sampled.std(dim=0, unbiased=False).clamp_min(0.1)
    normalized = sampled / sampled_deviation
    correlation = normalized.T @ normalized / sampled.shape[0]
    diagonal = correlation.diagonal()
    off_diagonal = correlation - torch.diag_embed(diagonal)
    covariance = off_diagonal.square().mean()
    spectral_sum = diagonal.sum()
    spectral_square_sum = correlation.square().sum()
    effective_rank = spectral_sum.square() / spectral_square_sum.clamp_min(1e-12)
    usable_rank = float(min(sampled.shape[0] - 1, sampled.shape[1]))
    rank_fraction = effective_rank / max(1.0, usable_rank)
    rank = F.relu(rank_fraction.new_tensor(0.2) - rank_fraction)
    return variance, covariance, rank


def action_statistics_regularizer(
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_actions = gather_batch_with_grad(actions).float().flatten(0, -2)
    standard_deviation = global_actions.std(dim=0, unbiased=False)
    variance = F.relu(0.05 - standard_deviation).mean()
    centered = global_actions - global_actions.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off_diagonal.square().mean()


def transient_owner_regularizer(
    owner: torch.Tensor,
    association_confidence: torch.Tensor,
    presence: torch.Tensor,
) -> torch.Tensor:
    evidence = (1.0 - association_confidence.float()).detach()
    logits = torch.logit(owner[..., -1].float().clamp(1e-5, 1.0 - 1e-5))
    loss = F.binary_cross_entropy_with_logits(logits, evidence, reduction="none")
    weight = presence.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
