"""Dense signed trajectory relations with positive-only object support."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .trajectory_relation_teacher import (
    TrajectoryRelationTeacher,
    _lifecycle,
    _motion_targets,
    _ramp,
)


def _trajectory_statistics(evidence, config):
    visible = evidence.visibility.float()
    visible_count = visible.sum(dim=1)
    persistence = visible_count / visible.shape[1]
    identity = (evidence.sampled_features.float() * visible[..., None]).sum(dim=1)
    identity = F.normalize(
        identity / visible_count.clamp_min(1.0)[..., None], dim=-1, eps=1e-6
    )
    cosine = torch.einsum("bpd,bqd->bpq", identity, identity)
    appearance = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

    coordinates = evidence.coordinates.float()
    relative = coordinates[:, :, :, None] - coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    jointly_visible = visible[:, :, :, None] * visible[:, :, None]
    joint_count = jointly_visible.sum(dim=1)
    mean_distance = (distance * jointly_visible).sum(dim=1)
    mean_distance = mean_distance / joint_count.clamp_min(1.0)
    distance_variance = (
        (distance - mean_distance[:, None]).square() * jointly_visible
    ).sum(dim=1)
    distance_variance = distance_variance / joint_count.clamp_min(1.0)
    rigidity = torch.exp(-distance_variance.sqrt() / config.group_distance_sigma)
    locality = torch.exp(
        -mean_distance.square() / (2.0 * config.group_locality_sigma**2)
    )

    pair_visible = evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    pair_joint = pair_visible[:, :, :, None] & pair_visible[:, :, None]
    flow = evidence.residual_flow.float()
    flow_delta = (flow[:, :, :, None] - flow[:, :, None]).norm(dim=-1)
    motion_delta = (flow_delta * pair_joint.float()).sum(dim=1)
    motion_delta = motion_delta / pair_joint.float().sum(dim=1).clamp_min(1.0)
    motion_coherence = torch.exp(-motion_delta / config.relation_motion_sigma)

    union = (
        visible[:, :, :, None].sum(dim=1)
        + visible[:, :, None].sum(dim=1)
        - jointly_visible.sum(dim=1)
    )
    covisibility = jointly_visible.sum(dim=1) / union.clamp_min(1.0)
    salience = evidence.motion_salience.float() * pair_visible.float()
    activity = salience.amax(dim=1)
    return (
        identity,
        persistence,
        appearance,
        rigidity,
        locality,
        motion_coherence,
        covisibility,
        activity,
    )


def _signed_relations(evidence, config):
    values = _trajectory_statistics(evidence, config)
    (
        identity,
        persistence,
        appearance,
        rigidity,
        locality,
        motion_coherence,
        covisibility,
        activity,
    ) = values
    persistent_pair = persistence[:, :, None] * persistence[:, None]
    same_score = (
        (
            appearance.clamp_min(1e-6)
            * rigidity.clamp_min(1e-6)
            * locality.clamp_min(1e-6)
            * motion_coherence.clamp_min(1e-6)
        )
        .sqrt()
        .sqrt()
    )
    same_score = same_score * covisibility * persistent_pair

    active_pair = torch.maximum(activity[:, :, None], activity[:, None])
    separation = torch.maximum(1.0 - rigidity, 1.0 - motion_coherence)
    different_score = separation * (0.25 + 0.75 * active_pair)
    different_score = different_score * (0.5 + 0.5 * (1.0 - appearance))
    different_score = different_score * covisibility * persistent_pair

    signed = same_score - different_score
    floor = config.relation_confidence_floor
    scale = max(1.0 - floor, 1e-6)
    same = ((signed - floor) / scale).clamp(0.0, 1.0)
    different = ((-signed - floor) / scale).clamp(0.0, 1.0)
    diagonal = torch.eye(signed.shape[-1], device=signed.device, dtype=torch.bool)[None]
    same = same.masked_fill(diagonal, 0.0)
    different = different.masked_fill(diagonal, 0.0)
    relation = same_score / (same_score + different_score).clamp_min(1e-6)
    relation = relation.masked_fill(diagonal, 0.0)
    return identity, persistence, activity, relation, same, different


def _positive_object_support(persistence, activity, same, config):
    moving = _ramp(activity, config.object_motion_floor) * persistence
    propagated = (same * moving[:, None]).amax(dim=-1)
    support = torch.maximum(moving, propagated) * persistence
    return support.clamp(0.0, 1.0)


def build_trajectory_relation_teacher_v56(evidence, config, frame_times):
    identity, persistence, activity, relation, same, different = _signed_relations(
        evidence, config
    )
    object_support = _positive_object_support(persistence, activity, same, config)
    zero = torch.zeros_like(object_support)
    visibility, presence, known, lifecycle_state = _lifecycle(evidence)
    motion, motion_valid = _motion_targets(evidence, frame_times, config)
    tensors = (
        identity,
        persistence,
        relation,
        same,
        different,
        object_support,
        motion,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("v56 trajectory teacher produced non-finite evidence")
    return TrajectoryRelationTeacher(
        track_identity=identity.detach(),
        persistence=persistence.detach(),
        object_confidence=object_support.detach(),
        scene_confidence=zero.detach(),
        transient_confidence=zero.detach(),
        same_confidence=same.detach(),
        different_confidence=different.detach(),
        visibility=visibility.detach(),
        presence=presence.detach(),
        lifecycle_known=known.detach(),
        lifecycle_state=lifecycle_state.detach(),
        motion=motion.detach(),
        motion_valid=motion_valid.detach(),
        relation_score=relation.detach(),
    )
