"""Cross-rank anti-collapse statistics over supported compact features."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import (
    gather_batch_with_grad,
    gather_batch_without_grad,
)


def per_sample_effective_rank_fraction(
    feature: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if feature.ndim != 3:
        raise ValueError("per-sample rank expects feature shape [B,R,D]")
    if valid.shape != feature.shape[:2]:
        raise ValueError("per-sample rank validity shape differs")
    sampled = feature.float()[..., ::4]
    weight = valid.float()
    support = weight.sum(dim=1)
    mean = torch.einsum("br,brd->bd", weight, sampled)
    mean = mean / support.clamp_min(1.0)[:, None]
    centered = (sampled - mean[:, None]) * weight.sqrt()[..., None]
    covariance = centered.transpose(1, 2) @ centered
    spectral_sum = centered.square().sum(dim=(1, 2))
    spectral_square_sum = covariance.square().sum(dim=(1, 2))
    effective_rank = spectral_sum.square() / spectral_square_sum.clamp_min(1e-12)
    usable_rank = torch.minimum(
        (support - 1.0).clamp_min(1.0),
        support.new_full((), float(sampled.shape[-1])),
    )
    rank_fraction = effective_rank / usable_rank
    return torch.where(support >= 2.0, rank_fraction, torch.zeros_like(rank_fraction))


def _per_sample_rank_loss(
    feature: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    rank_fraction = per_sample_effective_rank_fraction(feature, valid)
    support = valid.float().sum(dim=1)
    available = (support >= 2.0).to(rank_fraction.dtype)
    deficit = F.relu(rank_fraction.new_tensor(0.2) - rank_fraction)
    return (deficit * available).sum() / available.sum().clamp_min(1.0)


def owner_conditioned_residual(
    feature: torch.Tensor,
    valid: torch.Tensor,
    owner: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if owner.shape[:2] != feature.shape[:2]:
        raise ValueError("owner and feature region axes differ")
    if owner.shape[-1] < 2:
        raise ValueError("owner tensor requires environment and transient owners")
    environment_owner = owner[..., :-1].detach().float()
    environment_mass = environment_owner.sum(dim=-1)
    weight = environment_owner * valid.float()[..., None]
    owner_mass = weight.sum(dim=1)
    owner_mean = torch.einsum("bro,brd->bod", weight, feature.float())
    owner_mean = owner_mean / owner_mass.clamp_min(1e-6)[..., None]
    normalized_owner = environment_owner / environment_mass.clamp_min(1e-6)[
        ..., None
    ]
    conditional_mean = torch.einsum("bro,bod->brd", normalized_owner, owner_mean)
    residual = feature.float() - conditional_mean
    residual_valid = valid & (environment_mass > 0.25)
    return residual, residual_valid


def feature_statistics_regularizer(
    feature: torch.Tensor,
    valid: torch.Tensor,
    owner: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if feature.ndim != 3:
        raise ValueError("feature regularizer expects feature shape [B,R,D]")
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
    rank = _per_sample_rank_loss(feature, valid)
    if owner is not None:
        residual, residual_valid = owner_conditioned_residual(
            feature, valid, owner
        )
        rank = rank + 0.5 * _per_sample_rank_loss(residual, residual_valid)
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
