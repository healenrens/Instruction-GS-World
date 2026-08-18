"""Factorized target kernel for relation-supervised Object State learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .trajectory_relation_teacher import TrajectoryRelationTeacher


@dataclass(frozen=True)
class ObjectStatePredictions:
    assignment: torch.Tensor
    identity: torch.Tensor
    motion: torch.Tensor
    center: torch.Tensor
    visibility: torch.Tensor
    presence: torch.Tensor
    decoder_assignment: torch.Tensor


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def binary_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float().clamp(0.0, 1.0)
    return -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())


def visible_track_mean(value: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    weight = visible.float()
    while weight.ndim < value.ndim:
        weight = weight[..., None]
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _assignment_terms(prediction, teacher, config):
    assignment = prediction.assignment.float().clamp_min(1e-6)
    visible = teacher.visibility.float()
    mean_owner = visible_track_mean(assignment, visible)
    temporal_similarity = (assignment * mean_owner[:, None]).sum(dim=-1)
    object_weight = teacher.object_confidence[:, None] * visible
    track_cycle = weighted_mean(1.0 - temporal_similarity, object_weight)

    object_mean = mean_owner[..., : config.object_slots]
    relation_similarity = torch.einsum("bpk,bqk->bpq", object_mean, object_mean)
    object_pair = (
        teacher.object_confidence[:, :, None]
        * teacher.object_confidence[:, None]
    )
    same_weight = teacher.same_confidence * object_pair
    different_weight = teacher.different_confidence * object_pair
    same = weighted_mean(1.0 - relation_similarity, same_weight)
    different = weighted_mean(relation_similarity, different_weight)
    relation = same + different

    object_probability = assignment[..., : config.object_slots].sum(dim=-1)
    scene_probability = assignment[..., config.object_slots]
    transient_probability = assignment[..., config.object_slots + 1]
    scene_weight = teacher.scene_confidence[:, None] * visible
    transient_weight = teacher.transient_confidence[:, None] * visible
    owner_evidence = weighted_mean(-object_probability.log(), object_weight)
    owner_evidence = owner_evidence + weighted_mean(-scene_probability.log(), scene_weight)
    owner_evidence = owner_evidence + weighted_mean(
        -transient_probability.log(), transient_weight
    )

    entropy = -(assignment * assignment.log()).sum(dim=-1)
    known_owner = object_weight + scene_weight + transient_weight
    assignment_entropy = weighted_mean(entropy, known_owner)

    root_mass = torch.einsum(
        "btpk,btp->bk", assignment[..., : config.object_slots], object_weight
    )
    active = 1.0 - torch.exp(-root_mass)
    root_complexity = active.mean()
    root_share = root_mass / root_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    dominant = F.relu(root_share.amax(dim=-1) - config.dominant_root_limit).square().mean()
    return {
        "track_cycle": track_cycle,
        "relation": relation,
        "same_relation": same,
        "different_relation": different,
        "owner_evidence": owner_evidence,
        "assignment_entropy": assignment_entropy,
        "root_complexity": root_complexity,
        "dominant_root": dominant,
        "object_probability": weighted_mean(object_probability, visible),
        "scene_probability": weighted_mean(scene_probability, visible),
        "transient_probability": weighted_mean(transient_probability, visible),
        "effective_roots": active.sum(dim=-1).mean(),
        "maximum_root_share": root_share.amax(dim=-1).mean(),
    }


def _identity_term(prediction, teacher, config):
    identity = F.normalize(prediction.identity.float(), dim=-1, eps=1e-6)
    visible_object = teacher.visibility.float() * teacher.object_confidence[:, None]
    mean_identity = F.normalize(
        visible_track_mean(identity, teacher.visibility), dim=-1, eps=1e-6
    )
    temporal = weighted_mean(
        1.0 - (identity * mean_identity[:, None]).sum(dim=-1), visible_object
    )
    pair_similarity = torch.einsum("bpd,bqd->bpq", mean_identity, mean_identity)
    object_pair = (
        teacher.object_confidence[:, :, None]
        * teacher.object_confidence[:, None]
    )
    same = weighted_mean(
        1.0 - pair_similarity,
        teacher.same_confidence * object_pair,
    )
    negative = weighted_mean(
        F.relu(pair_similarity - config.identity_negative_margin),
        teacher.different_confidence * object_pair,
    )
    return temporal + same + negative, temporal, same, negative


def _motion_geometry_terms(prediction, teacher, evidence):
    relation = teacher.same_confidence.float()
    relation_strength = relation.sum(dim=-1)
    diagonal = torch.eye(
        relation.shape[-1], device=relation.device, dtype=relation.dtype
    )[None]
    pooling = relation + diagonal * relation_strength[..., None]

    coordinate_valid = teacher.visibility.float()
    coordinate_weight = pooling[:, None] * coordinate_valid[:, :, None]
    coordinate_normalizer = coordinate_weight.sum(dim=-1).clamp_min(1e-6)
    target_center = torch.einsum(
        "btpq,btqd->btpd", coordinate_weight, evidence.coordinates.float()
    )
    target_center = target_center / coordinate_normalizer[..., None]

    motion_valid = teacher.motion_valid.float()
    motion_pooling = pooling[:, None, :, :, None] * motion_valid[:, :, None]
    motion_normalizer = motion_pooling.sum(dim=3).clamp_min(1e-6)
    target_motion = torch.einsum(
        "btpqh,btqhd->btphd", motion_pooling, teacher.motion.float()
    )
    target_motion = target_motion / motion_normalizer[..., None]
    motion_error = F.smooth_l1_loss(
        prediction.motion.float(), target_motion, reduction="none"
    ).mean(dim=-1)
    relation_known = (relation_strength > 0.0).float()
    motion_weight = (motion_normalizer > 1e-6).float()
    motion_weight = motion_weight * teacher.object_confidence[:, None, :, None]
    motion_weight = motion_weight * relation_known[:, None, :, None]
    motion = weighted_mean(motion_error, motion_weight)
    center_error = F.smooth_l1_loss(
        prediction.center.float(), target_center, reduction="none"
    ).mean(dim=-1)
    center_weight = teacher.visibility.float() * teacher.object_confidence[:, None]
    center_weight = center_weight * relation_known[:, None]
    geometry = weighted_mean(center_error, center_weight)
    return motion, geometry


def _lifecycle_term(prediction, teacher):
    known = teacher.lifecycle_known.float() * teacher.object_confidence[:, None]
    visibility = weighted_mean(
        binary_cross_entropy(prediction.visibility, teacher.visibility), known
    )
    presence = weighted_mean(
        binary_cross_entropy(prediction.presence, teacher.presence), known
    )
    return visibility + presence, visibility, presence


def _decoder_support_term(prediction, teacher):
    target = prediction.assignment.detach().float()
    decoder = prediction.decoder_assignment.float().clamp_min(1e-6)
    cross_entropy = -(target * decoder.log()).sum(dim=-1)
    return weighted_mean(cross_entropy, teacher.visibility.float())


def object_state_target_terms(
    prediction: ObjectStatePredictions,
    teacher: TrajectoryRelationTeacher,
    evidence,
    config,
) -> dict[str, torch.Tensor]:
    assignments = _assignment_terms(prediction, teacher, config)
    identity, identity_temporal, identity_same, identity_negative = _identity_term(
        prediction, teacher, config
    )
    motion, geometry = _motion_geometry_terms(prediction, teacher, evidence)
    lifecycle, visibility, presence = _lifecycle_term(prediction, teacher)
    decoder_support = _decoder_support_term(prediction, teacher)
    total = (
        config.track_cycle_weight * assignments["track_cycle"]
        + config.relation_weight * assignments["relation"]
        + config.owner_evidence_weight * assignments["owner_evidence"]
        + config.identity_weight * identity
        + config.motion_weight * motion
        + config.lifecycle_weight * lifecycle
        + config.geometry_weight * geometry
        + config.decoder_support_weight * decoder_support
        + config.assignment_entropy_weight * assignments["assignment_entropy"]
        + config.root_complexity_weight * assignments["root_complexity"]
        + config.dominant_root_weight * assignments["dominant_root"]
    )
    terms = {
        "target_total": total,
        "identity": identity,
        "identity_temporal": identity_temporal,
        "identity_same": identity_same,
        "identity_negative": identity_negative,
        "motion": motion,
        "geometry": geometry,
        "lifecycle": lifecycle,
        "visibility": visibility,
        "presence": presence,
        "decoder_support": decoder_support,
        **assignments,
    }
    if not all(bool(torch.isfinite(value)) for value in terms.values()):
        raise RuntimeError("v52 Object State target contains non-finite terms")
    return terms
