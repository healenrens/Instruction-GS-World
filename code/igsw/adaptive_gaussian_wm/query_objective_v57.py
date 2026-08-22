"""Held-out relation objective for v57 single-query object binding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .point_track_teacher import sample_patch_field


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def track_support_probability(state, evidence, grid_hw):
    values = state.support[..., None]
    return sample_patch_field(values, evidence.coordinates, grid_hw)[..., 0]


def _heldout_relation_loss(track_support, teacher):
    visible = teacher.track_visibility.float()
    predicted = (track_support * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
    same_weight = teacher.same_target.float() * teacher.heldout_track_mask.float()
    different_weight = teacher.different_target.float() * teacher.heldout_track_mask.float()
    same = _weighted_mean(-predicted.clamp_min(1e-6).log(), same_weight)
    different = _weighted_mean(-torch.log1p(-predicted.clamp_max(1.0 - 1e-6)), different_weight)
    return same + different, predicted, same, different


def _support_overlap(first, second, valid):
    first = first * valid.float()
    second = second * valid.float()
    intersection = torch.minimum(first, second).sum(dim=(-2, -1))
    union = torch.maximum(first, second).sum(dim=(-2, -1)).clamp_min(1e-6)
    return intersection / union


def query_object_binding_terms(
    primary,
    alternate,
    negative,
    features,
    evidence,
    teacher,
    grid_hw,
    config,
):
    primary_tracks = track_support_probability(primary, evidence, grid_hw)
    relation, predicted, relation_same, relation_different = _heldout_relation_loss(
        primary_tracks, teacher
    )
    alternate_weight = teacher.alternate_valid.float()
    query_weight = teacher.query_valid.float()
    identity_agreement = 1.0 - (primary.identity * alternate.identity).sum(dim=-1)
    identity_agreement = _weighted_mean(identity_agreement, alternate_weight)
    support_agreement = 1.0 - _support_overlap(
        primary.support, alternate.support, features.valid
    )
    support_agreement = _weighted_mean(support_agreement, alternate_weight)

    negative_weight = teacher.negative_valid.float()
    negative_similarity = (primary.identity * negative.identity).sum(dim=-1)
    identity_separation = _weighted_mean(
        F.relu(negative_similarity - config.identity_negative_margin), negative_weight
    )
    negative_overlap = _support_overlap(primary.support, negative.support, features.valid)
    support_separation = _weighted_mean(
        F.relu(negative_overlap - config.support_overlap_margin), negative_weight
    )

    semantic_delta = primary.pooled_semantic[:, 1:] - primary.pooled_semantic[:, :-1]
    visible_pair = primary.visibility[:, 1:] * primary.visibility[:, :-1]
    visible_pair = visible_pair * query_weight[:, None]
    semantic_consistency = _weighted_mean(
        semantic_delta.square().mean(dim=-1), visible_pair
    )
    trace = primary.covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    compactness = _weighted_mean(
        trace, primary.visibility * query_weight[:, None]
    )
    total = (
        config.heldout_track_weight * relation
        + config.seed_identity_weight * identity_agreement
        + config.seed_support_weight * support_agreement
        + config.query_separation_weight * (identity_separation + support_separation)
        + config.semantic_consistency_weight * semantic_consistency
        + config.compactness_weight * compactness
    )
    terms = {
        "total": total,
        "heldout_relation": relation,
        "heldout_same": relation_same,
        "heldout_different": relation_different,
        "seed_identity": identity_agreement,
        "seed_support": support_agreement,
        "query_identity_separation": identity_separation,
        "query_support_separation": support_separation,
        "semantic_consistency": semantic_consistency,
        "compactness": compactness,
        "heldout_support_positive": _weighted_mean(
            predicted, teacher.same_target * teacher.heldout_track_mask.float()
        ),
        "heldout_support_negative": _weighted_mean(
            predicted, teacher.different_target * teacher.heldout_track_mask.float()
        ),
    }
    nonfinite = [name for name, value in terms.items() if not bool(torch.isfinite(value))]
    if nonfinite:
        raise RuntimeError("v57 objective contains non-finite terms: " + ", ".join(nonfinite))
    return terms
