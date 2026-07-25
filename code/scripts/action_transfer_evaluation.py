"""Cross-episode matched-effect action transfer helpers."""
from __future__ import annotations

import math

import torch


TRANSFER_VARIANTS = (
    "matched_effect_action",
    "matched_effect_residual",
    "random_residual",
)
TRANSFER_COMPARISONS = (
    (
        "matched_effect_over_shuffled",
        "matched_effect_action",
        "shuffled_action",
    ),
    (
        "matched_residual_over_random",
        "matched_effect_residual",
        "random_residual",
    ),
    (
        "matched_residual_over_canonical",
        "matched_effect_residual",
        "canonical_only",
    ),
)


def canonical_only(actions: torch.Tensor, canonical_dim: int) -> torch.Tensor:
    if not 0 < canonical_dim < actions.shape[-1]:
        raise ValueError("transfer evaluation requires canonical and residual actions")
    return torch.cat(
        (
            actions[..., :canonical_dim],
            torch.zeros_like(actions[..., canonical_dim:]),
        ),
        dim=-1,
    )


def component_action_variants(
    posterior: torch.Tensor,
    canonical_dim: int,
) -> dict[str, torch.Tensor]:
    canonical = canonical_only(posterior, canonical_dim)
    return {
        "canonical_only": canonical,
        "residual_only": posterior - canonical,
    }


def nearest_effect_donors(
    action_bank: torch.Tensor,
    clusters: torch.Tensor,
    shuffled_index: torch.Tensor,
    canonical_dim: int,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if action_bank.ndim != 4 or len(action_bank) < 2:
        raise ValueError("action bank must be [N,Q,S,D] with at least two samples")
    if clusters.shape != (len(action_bank),):
        raise ValueError("cluster ids must align with the action bank")
    if shuffled_index.shape != clusters.shape:
        raise ValueError("shuffled indices must align with the action bank")
    canonical = action_bank[..., :canonical_dim].float().flatten(1)
    scale = canonical.std(dim=0, unbiased=False).clamp_min(1e-3)
    normalized = (canonical - canonical.mean(dim=0)) / scale
    normalization = math.sqrt(normalized.shape[1])
    matched_indices = []
    matched_distances = []
    for start in range(0, len(normalized), chunk_size):
        end = min(start + chunk_size, len(normalized))
        distance = torch.cdist(normalized[start:end], normalized) / normalization
        same_cluster = clusters[start:end, None] == clusters[None, :]
        distance.masked_fill_(same_cluster, torch.inf)
        values, indices = distance.min(dim=1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("every sample needs a donor from another episode")
        matched_indices.append(indices)
        matched_distances.append(values)
    matched_index = torch.cat(matched_indices)
    matched_distance = torch.cat(matched_distances)
    shuffled_distance = (
        normalized - normalized[shuffled_index]
    ).square().sum(dim=1).sqrt() / normalization
    return matched_index, {
        "matched_canonical_distance_mean": float(matched_distance.mean()),
        "shuffled_canonical_distance_mean": float(shuffled_distance.mean()),
        "matched_distance_ratio": float(
            matched_distance.mean() / shuffled_distance.mean().clamp_min(1e-8)
        ),
        "matched_same_episode_collisions": int(
            (clusters == clusters[matched_index]).sum()
        ),
        "shuffled_same_episode_collisions": int(
            (clusters == clusters[shuffled_index]).sum()
        ),
    }


def transfer_action_variants(
    target: torch.Tensor,
    matched: torch.Tensor,
    shuffled: torch.Tensor,
    canonical_dim: int,
) -> dict[str, torch.Tensor]:
    if target.shape != matched.shape or target.shape != shuffled.shape:
        raise ValueError("target and donor actions must share one layout")
    target_canonical = canonical_only(target, canonical_dim)
    matched_residual = matched - canonical_only(matched, canonical_dim)
    shuffled_residual = shuffled - canonical_only(shuffled, canonical_dim)
    return {
        "matched_effect_action": matched,
        "matched_effect_residual": target_canonical + matched_residual,
        "random_residual": target_canonical + shuffled_residual,
    }
