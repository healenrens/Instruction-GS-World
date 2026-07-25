"""Evaluation-only metrics for controlled feature-video experiments."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_feature_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    weight = valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((left * right).sum() / denominator)


def slot_purity(output: dict, labels: torch.Tensor) -> float:
    """Majority-label purity; labels are evaluation metadata, never model inputs."""
    token_assignment = output["history_token_states"][-1].assignment
    slot_assignment = output["history_slot_states"][-1].assignment
    patch_to_slot = torch.einsum(
        "bmn,bmk->bnk",
        token_assignment,
        slot_assignment,
    )
    purities = []
    for batch_index in range(labels.shape[0]):
        valid = labels[batch_index] >= 0
        if not bool(valid.any()):
            continue
        sample_labels = labels[batch_index, valid]
        classes = int(sample_labels.max()) + 1
        one_hot = F.one_hot(sample_labels, num_classes=classes).float()
        weights = patch_to_slot[batch_index, valid]
        counts = torch.einsum("nk,nc->kc", weights, one_hot)
        purities.append(
            counts.max(dim=-1).values.sum() / counts.sum().clamp_min(1e-6)
        )
    if not purities:
        return 0.0
    return float(torch.stack(purities).mean())


def slot_clustering_scores(output: dict, labels: torch.Tensor) -> dict[str, float]:
    """Soft purity/completeness; object labels are evaluation-only metadata."""
    token_assignment = output["history_token_states"][-1].assignment
    slot_assignment = output["history_slot_states"][-1].assignment
    patch_to_slot = torch.einsum(
        "bmn,bmk->bnk",
        token_assignment,
        slot_assignment,
    )
    purities = []
    completenesses = []
    for batch_index in range(labels.shape[0]):
        valid = labels[batch_index] >= 0
        if not bool(valid.any()):
            continue
        sample_labels = labels[batch_index, valid]
        classes = int(sample_labels.max()) + 1
        one_hot = F.one_hot(sample_labels, num_classes=classes).float()
        counts = torch.einsum(
            "nk,nc->kc",
            patch_to_slot[batch_index, valid],
            one_hot,
        )
        total = counts.sum().clamp_min(1e-6)
        purities.append(counts.max(dim=-1).values.sum() / total)
        completenesses.append(counts.max(dim=0).values.sum() / total)
    if not purities:
        return {"purity": 0.0, "completeness": 0.0, "f1": 0.0}
    purity = torch.stack(purities).mean()
    completeness = torch.stack(completenesses).mean()
    f1 = 2.0 * purity * completeness / (purity + completeness).clamp_min(1e-6)
    return {
        "purity": float(purity),
        "completeness": float(completeness),
        "f1": float(f1),
    }


def paired_max_difference(value: torch.Tensor) -> float:
    pair_count = value.shape[0] // 2
    left = value[: pair_count * 2 : 2]
    right = value[1 : pair_count * 2 : 2]
    return float((left - right).abs().max())
