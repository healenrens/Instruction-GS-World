"""Externally masked objective for persistent state of one queried entity."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .query_objective_v57 import (
    _heldout_relation_loss,
    _support_overlap,
    _weighted_mean,
    track_support_probability,
)


def _balanced_visibility_loss(logits, target, known):
    known = known.float()
    positive = known * target
    negative = known * (1.0 - target)
    positive_count = positive.sum()
    negative_count = negative.sum()
    class_count = (positive_count > 0).float() + (negative_count > 0).float()
    positive_weight = known.sum() / positive_count.clamp_min(1.0) / class_count.clamp_min(1.0)
    negative_weight = known.sum() / negative_count.clamp_min(1.0) / class_count.clamp_min(1.0)
    weight = positive * positive_weight + negative * negative_weight
    return _weighted_mean(
        F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none"),
        weight,
    )


def _visibility_metrics(probability, target, known):
    known = known.float()
    positive = known * target
    negative = known * (1.0 - target)
    predicted = probability >= 0.5
    true_positive = (predicted.float() * positive).sum()
    true_negative = ((~predicted).float() * negative).sum()
    false_positive = (predicted.float() * negative).sum()
    recall = true_positive / positive.sum().clamp_min(1.0)
    specificity = true_negative / negative.sum().clamp_min(1.0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
    present_classes = (positive.sum() > 0).float() + (negative.sum() > 0).float()
    balanced = (
        recall * (positive.sum() > 0).float()
        + specificity * (negative.sum() > 0).float()
    ) / present_classes.clamp_min(1.0)
    brier = _weighted_mean((probability - target).square(), known)
    target_rate = positive.sum() / known.sum().clamp_min(1.0)
    constant_brier = _weighted_mean((target_rate - target).square(), known)
    predicted_rate = _weighted_mean(probability, known)
    rate_error = (predicted_rate - target_rate).abs() / target_rate.clamp_min(1e-3)
    return {
        "visibility_balanced_accuracy": balanced,
        "visibility_f1": f1,
        "visibility_recall": recall,
        "visibility_occluded_recall": specificity,
        "visibility_brier": brier,
        "visibility_brier_gain_over_constant": constant_brier - brier,
        "visibility_predicted_rate": predicted_rate,
        "visibility_target_rate": target_rate,
        "visibility_rate_relative_error": rate_error,
    }


def _reappearance_metrics(state, teacher):
    visible = teacher.visibility_target.bool() & teacher.lifecycle_known
    occluded = teacher.occluded_candidate
    seen_visible = visible.cumsum(dim=1) > visible.long()
    seen_occluded = occluded.cumsum(dim=1) > 0
    event = visible & seen_visible & seen_occluded
    correct = (state.identity_sequence * state.identity[:, None]).sum(dim=-1)
    shuffled_identity = state.identity.roll(1, dims=0)
    shuffled = (state.identity_sequence * shuffled_identity[:, None]).sum(dim=-1)
    return {
        "identity_reappearance_cosine": _weighted_mean(correct, event.float()),
        "identity_reappearance_shuffled": _weighted_mean(shuffled, event.float()),
        "identity_reappearance_margin": _weighted_mean(correct - shuffled, event.float()),
        "identity_reappearance_event_fraction": event.float().mean(),
    }


def query_persistent_state_terms(
    primary,
    alternate,
    negative,
    motion_prediction,
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
    identity_agreement = _weighted_mean(
        1.0 - (primary.identity * alternate.identity).sum(dim=-1), alternate_weight
    )
    support_agreement = _weighted_mean(
        1.0 - _support_overlap(primary.support, alternate.support, features.valid),
        alternate_weight,
    )
    negative_weight = teacher.negative_valid.float()
    identity_separation = _weighted_mean(
        F.relu(
            (primary.identity * negative.identity).sum(dim=-1)
            - config.identity_negative_margin
        ),
        negative_weight,
    )
    support_separation = _weighted_mean(
        F.relu(
            _support_overlap(primary.support, negative.support, features.valid)
            - config.support_overlap_margin
        ),
        negative_weight,
    )

    known = teacher.lifecycle_known.float()
    visible = teacher.visibility_target.float() * known
    visibility = _balanced_visibility_loss(
        primary.visibility_logits, teacher.visibility_target, teacher.lifecycle_known
    )
    identity_cosine = (
        primary.identity_sequence * primary.identity[:, None]
    ).sum(dim=-1)
    identity_persistence = _weighted_mean(1.0 - identity_cosine, known)
    if primary.pooled_semantic.shape[1] > 1:
        visible_pair = visible[:, 1:] * visible[:, :-1] * query_weight[:, None]
        semantic_delta = primary.pooled_semantic[:, 1:] - primary.pooled_semantic[:, :-1]
        semantic_consistency = _weighted_mean(
            semantic_delta.square().mean(dim=-1), visible_pair
        )
        center_delta = primary.center[:, 1:] - primary.center[:, :-1]
        pair_seconds = teacher.delta_seconds[:, 1:] - teacher.delta_seconds[:, :-1]
        center_velocity = center_delta / pair_seconds[..., None].clamp_min(1e-4)
        geometry_motion = _weighted_mean(
            F.smooth_l1_loss(
                center_velocity, teacher.motion_target[:, 1:], reduction="none"
            ).mean(dim=-1),
            teacher.motion_valid[:, 1:].float(),
        )
    else:
        semantic_consistency = primary.pooled_semantic.sum() * 0.0
        geometry_motion = primary.center.sum() * 0.0
    trace = primary.covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    compactness = _weighted_mean(trace, visible * query_weight[:, None])
    dynamic_motion = _weighted_mean(
        F.smooth_l1_loss(
            motion_prediction.float(), teacher.motion_target.float(), reduction="none"
        ).mean(dim=-1),
        teacher.motion_valid.float(),
    )
    zero_motion = _weighted_mean(
        F.smooth_l1_loss(
            torch.zeros_like(teacher.motion_target),
            teacher.motion_target.float(),
            reduction="none",
        ).mean(dim=-1),
        teacher.motion_valid.float(),
    )
    dynamic_motion_gain = (zero_motion - dynamic_motion) / zero_motion.clamp_min(1e-6)

    total = (
        config.heldout_track_weight * relation
        + config.seed_identity_weight * identity_agreement
        + config.seed_support_weight * support_agreement
        + config.query_separation_weight * (identity_separation + support_separation)
        + config.visibility_weight * visibility
        + config.identity_persistence_weight * identity_persistence
        + config.semantic_consistency_weight * semantic_consistency
        + config.compactness_weight * compactness
        + config.dynamic_motion_weight * dynamic_motion
        + config.geometry_motion_weight * geometry_motion
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
        "visibility_supervision": visibility,
        "identity_persistence": identity_persistence,
        "semantic_consistency": semantic_consistency,
        "compactness": compactness,
        "dynamic_motion": dynamic_motion,
        "dynamic_motion_zero_baseline": zero_motion,
        "dynamic_motion_relative_gain": dynamic_motion_gain,
        "geometry_motion": geometry_motion,
        "heldout_support_positive": _weighted_mean(
            predicted, teacher.same_target * teacher.heldout_track_mask.float()
        ),
        "heldout_support_negative": _weighted_mean(
            predicted, teacher.different_target * teacher.heldout_track_mask.float()
        ),
        "identity_known_cosine": _weighted_mean(identity_cosine, known),
        "dynamic_temporal_std": primary.dynamic.float().std(dim=1, correction=0).mean(),
        "geometry_temporal_std": primary.center.float().std(dim=1, correction=0).mean(),
        "visibility_unknown_activation": _weighted_mean(
            primary.visibility, teacher.unknown.float()
        ),
        **_visibility_metrics(
            primary.visibility, teacher.visibility_target, teacher.lifecycle_known
        ),
        **_reappearance_metrics(primary, teacher),
    }
    nonfinite = [name for name, value in terms.items() if not bool(torch.isfinite(value))]
    if nonfinite:
        raise RuntimeError("v58 objective contains non-finite terms: " + ", ".join(nonfinite))
    return terms
