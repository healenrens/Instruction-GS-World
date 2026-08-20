"""External relation objective with frozen-DINO semantic anchoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .object_state_target_v52 import object_state_target_terms, weighted_mean


def semantic_alignment_terms(
    semantic_identity: torch.Tensor,
    teacher,
) -> dict[str, torch.Tensor]:
    prediction = F.normalize(semantic_identity.float(), dim=-1, eps=1e-6)
    target = F.normalize(teacher.track_identity.float(), dim=-1, eps=1e-6)
    cosine = (prediction * target[:, None]).sum(dim=-1)
    weight = teacher.visibility.float() * teacher.object_confidence[:, None]
    alignment = weighted_mean(1.0 - cosine, weight)
    visible_cosine = weighted_mean(cosine, weight)

    mean_prediction = F.normalize(
        (prediction * teacher.visibility[..., None].float()).sum(dim=1)
        / teacher.visibility.float().sum(dim=1).clamp_min(1.0)[..., None],
        dim=-1,
        eps=1e-6,
    )
    pair_similarity = torch.einsum("bpd,bqd->bpq", mean_prediction, mean_prediction)
    object_pair = (
        teacher.object_confidence[:, :, None]
        * teacher.object_confidence[:, None]
    )
    same_weight = teacher.same_confidence * object_pair
    different_weight = teacher.different_confidence * object_pair
    semantic_same = weighted_mean(1.0 - pair_similarity, same_weight)
    semantic_different = weighted_mean(
        F.relu(pair_similarity - 0.20), different_weight
    )
    return {
        "semantic_alignment": alignment,
        "semantic_alignment_cosine": visible_cosine,
        "semantic_same_relation": semantic_same,
        "semantic_different_relation": semantic_different,
    }


def relation_semantic_object_state_terms(
    prediction,
    semantic_identity: torch.Tensor,
    teacher,
    evidence,
    config,
) -> dict[str, torch.Tensor]:
    base = object_state_target_terms(prediction, teacher, evidence, config)
    semantic = semantic_alignment_terms(semantic_identity, teacher)
    semantic_total = (
        semantic["semantic_alignment"]
        + 0.25 * semantic["semantic_same_relation"]
        + 0.25 * semantic["semantic_different_relation"]
    )
    target_total = (
        base["target_total"] + config.semantic_alignment_weight * semantic_total
    )
    terms = {
        **base,
        **semantic,
        "semantic_total": semantic_total,
        "target_total_without_semantic": base["target_total"],
        "target_total": target_total,
    }
    if not all(bool(torch.isfinite(value)) for value in terms.values()):
        raise RuntimeError("v54 relation-semantic objective contains non-finite terms")
    return terms
