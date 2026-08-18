"""Training-only trajectory relations without pseudo object identities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .point_track_teacher import PointTrackEvidence
from .trajectory_lifecycle import LIFECYCLE_OCCLUDED, LIFECYCLE_VISIBLE


@dataclass(frozen=True)
class TrajectoryRelationTeacher:
    track_identity: torch.Tensor
    persistence: torch.Tensor
    object_confidence: torch.Tensor
    scene_confidence: torch.Tensor
    transient_confidence: torch.Tensor
    same_confidence: torch.Tensor
    different_confidence: torch.Tensor
    visibility: torch.Tensor
    presence: torch.Tensor
    lifecycle_known: torch.Tensor
    lifecycle_state: torch.Tensor
    motion: torch.Tensor
    motion_valid: torch.Tensor
    relation_score: torch.Tensor


def _ramp(value: torch.Tensor, floor: float) -> torch.Tensor:
    return ((value - floor) / max(1.0 - floor, 1e-6)).clamp(0.0, 1.0)


def _track_statistics(evidence: PointTrackEvidence, config):
    visibility = evidence.visibility.float()
    visible_count = visibility.sum(dim=1)
    persistence = visible_count / visibility.shape[1]
    identity = (evidence.sampled_features.float() * visibility[..., None]).sum(dim=1)
    identity = F.normalize(
        identity / visible_count.clamp_min(1.0)[..., None], dim=-1, eps=1e-6
    )
    appearance = torch.einsum("bpd,bqd->bpq", identity, identity)
    appearance = ((appearance - config.group_appearance_floor) / (
        1.0 - config.group_appearance_floor
    )).clamp(0.0, 1.0)

    coordinates = evidence.coordinates.float()
    relative = coordinates[:, :, :, None] - coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    joint = visibility[:, :, :, None] * visibility[:, :, None]
    joint_count = joint.sum(dim=1).clamp_min(1.0)
    mean_distance = (distance * joint).sum(dim=1) / joint_count
    distance_deviation = (
        (distance - mean_distance[:, None]).square() * joint
    ).sum(dim=1) / joint_count
    rigidity = torch.exp(-distance_deviation.sqrt() / config.group_distance_sigma)
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
        visibility[:, :, :, None].sum(dim=1)
        + visibility[:, :, None].sum(dim=1)
        - joint.sum(dim=1)
    )
    covisibility = joint.sum(dim=1) / union.clamp_min(1.0)
    return identity, persistence, appearance, rigidity, locality, motion_coherence, covisibility


def _relation_confidence(evidence: PointTrackEvidence, config):
    values = _track_statistics(evidence, config)
    identity, persistence, appearance, rigidity, locality, motion, covisibility = values
    relation = appearance * rigidity * motion
    relation = relation * (0.25 + 0.75 * locality) * (0.5 + 0.5 * covisibility)
    independence = torch.maximum(1.0 - rigidity, 1.0 - motion)
    independence = independence * (0.5 + 0.5 * (1.0 - appearance))
    persistent_pair = persistence[:, :, None] * persistence[:, None]
    same = _ramp(relation, config.relation_same_floor) * persistent_pair
    different = _ramp(independence, config.relation_different_floor) * persistent_pair
    same = same * (relation >= independence).float()
    different = different * (independence > relation).float()
    diagonal = torch.eye(
        relation.shape[-1], device=relation.device, dtype=torch.bool
    )[None]
    same = same.masked_fill(diagonal, 0.0)
    different = different.masked_fill(diagonal, 0.0)
    return identity, persistence, relation, same, different


def _owner_evidence(
    evidence: PointTrackEvidence,
    persistence: torch.Tensor,
    same: torch.Tensor,
    config,
):
    pair_visible = evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    motion = (evidence.motion_salience.float() * pair_visible.float()).amax(dim=1)
    moving = _ramp(motion, config.object_motion_floor) * persistence
    propagated = (same * moving[:, None]).amax(dim=-1)
    object_confidence = torch.maximum(moving, propagated) * persistence
    static = (1.0 - moving) * persistence
    scene_confidence = static * (1.0 - propagated) * config.scene_evidence_scale
    transient = (persistence > 0.0).float() * (
        1.0 - _ramp(persistence, config.transient_visible_fraction)
    )
    transient = transient * (1.0 - object_confidence)
    return object_confidence, scene_confidence, transient


def _lifecycle(evidence: PointTrackEvidence):
    visible = evidence.visibility.bool()
    seen_before = visible.cumsum(dim=1) > 0
    seen_after = visible.flip(1).cumsum(dim=1).flip(1) > 0
    occluded = (~visible) & seen_before & seen_after
    known = visible | occluded
    state = torch.full_like(visible, -1, dtype=torch.long)
    state[visible] = LIFECYCLE_VISIBLE
    state[occluded] = LIFECYCLE_OCCLUDED
    presence = known.float()
    return visible.float(), presence, known, state


def _motion_targets(evidence: PointTrackEvidence, frame_times: torch.Tensor, config):
    batch, frames, points = evidence.visibility.shape
    horizons = config.dynamic_horizons
    target = evidence.coordinates.new_zeros(batch, frames, points, len(horizons), 2)
    valid = torch.zeros(
        batch, frames, points, len(horizons),
        device=evidence.visibility.device,
        dtype=torch.bool,
    )
    for horizon_index, horizon in enumerate(horizons):
        if horizon >= frames:
            continue
        pair_valid = evidence.visibility[:, :-horizon] & evidence.visibility[:, horizon:]
        dt = frame_times[:, horizon:] - frame_times[:, :-horizon]
        displacement = evidence.coordinates[:, horizon:] - evidence.coordinates[:, :-horizon]
        global_displacement = (
            displacement * pair_valid[..., None].float()
        ).sum(dim=2, keepdim=True)
        global_displacement = global_displacement / pair_valid.float().sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)[..., None]
        residual = (displacement - global_displacement) / dt[..., None, None].clamp_min(1e-4)
        target[:, :-horizon, :, horizon_index] = residual
        valid[:, :-horizon, :, horizon_index] = pair_valid
    return target, valid


def build_trajectory_relation_teacher(
    evidence: PointTrackEvidence,
    config,
    frame_times: torch.Tensor,
) -> TrajectoryRelationTeacher:
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
        raise RuntimeError("v52 trajectory relation teacher produced non-finite evidence")
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
