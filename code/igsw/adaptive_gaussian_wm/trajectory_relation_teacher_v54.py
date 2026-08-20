"""Persistent motion-seeded relation evidence for v54 Object State."""

from __future__ import annotations

import torch

from .trajectory_relation_teacher import (
    TrajectoryRelationTeacher,
    _lifecycle,
    _motion_targets,
    _relation_confidence,
    _ramp,
)


def _owner_evidence(evidence, persistence, same, config):
    pair_visible = evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    salience = evidence.motion_salience.float() * pair_visible.float()
    motion = salience.amax(dim=1)
    moving = _ramp(motion, config.object_motion_floor)
    persistent = _ramp(persistence, config.object_persistence_floor)
    persistent_motion = moving * persistent
    propagated_motion = (same * persistent_motion[:, None]).amax(dim=-1)
    relation_support = same.amax(dim=-1)

    object_confidence = torch.maximum(persistent_motion, propagated_motion)
    object_confidence = torch.maximum(
        object_confidence,
        relation_support * persistent_motion.amax(dim=-1, keepdim=True),
    ) * persistence
    static = (1.0 - moving) * persistent
    scene_confidence = (
        static * (1.0 - relation_support) * config.scene_evidence_scale
    )
    short_lived = (persistence > 0.0).float() * (
        1.0 - _ramp(persistence, config.transient_visible_fraction)
    )
    unsupported_short_motion = moving * (1.0 - persistent)
    transient_confidence = torch.maximum(short_lived, unsupported_short_motion)
    transient_confidence = transient_confidence * (1.0 - object_confidence)
    return object_confidence, scene_confidence, transient_confidence


def build_trajectory_relation_teacher_v54(evidence, config, frame_times):
    identity, persistence, relation, same, different = _relation_confidence(
        evidence, config
    )
    object_confidence, scene_confidence, transient_confidence = _owner_evidence(
        evidence, persistence, same, config
    )
    visibility, presence, known, lifecycle_state = _lifecycle(evidence)
    motion, motion_valid = _motion_targets(evidence, frame_times, config)
    tensors = (
        identity, persistence, relation, same, different, object_confidence,
        scene_confidence, transient_confidence, motion,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("v54 trajectory relation teacher produced non-finite evidence")
    return TrajectoryRelationTeacher(
        track_identity=identity.detach(),
        persistence=persistence.detach(),
        object_confidence=object_confidence.detach(),
        scene_confidence=scene_confidence.detach(),
        transient_confidence=transient_confidence.detach(),
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
