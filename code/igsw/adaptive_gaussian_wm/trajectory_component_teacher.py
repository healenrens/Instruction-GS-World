"""Training-only trajectory graph teacher independent of student assignments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .point_track_object_state import support_shape_from_assignment
from .point_track_teacher import PointTrackEvidence


LIFECYCLE_VISIBLE = 0
LIFECYCLE_OCCLUDED = 1
LIFECYCLE_UNKNOWN = 2
LIFECYCLE_ABSENT = 3


@dataclass(frozen=True)
class TrajectoryComponentTeacher:
    track_owner: torch.Tensor
    component_valid: torch.Tensor
    identity: torch.Tensor
    visibility: torch.Tensor
    presence: torch.Tensor
    lifecycle_known: torch.Tensor
    lifecycle_state: torch.Tensor
    center: torch.Tensor
    log_scale: torch.Tensor
    support_shape: torch.Tensor
    geometry_valid: torch.Tensor
    motion: torch.Tensor
    motion_valid: torch.Tensor
    graph_affinity: torch.Tensor
    track_motion: torch.Tensor


def _track_statistics(evidence: PointTrackEvidence, config):
    visibility = evidence.visibility.float()
    visible_count = visibility.sum(dim=1)
    appearance = (evidence.sampled_features.float() * visibility[..., None]).sum(dim=1)
    appearance = F.normalize(
        appearance / visible_count.clamp_min(1.0)[..., None], dim=-1, eps=1e-6
    )
    pair_visible = evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    flow_weight = pair_visible.float()
    flow = (evidence.residual_flow.float() * flow_weight[..., None]).sum(dim=1)
    flow = flow / flow_weight.sum(dim=1).clamp_min(1.0)[..., None]
    track_motion = (evidence.motion_salience.float() * flow_weight).sum(dim=1)
    track_motion = track_motion / flow_weight.sum(dim=1).clamp_min(1.0)

    appearance_similarity = torch.einsum("bpd,bqd->bpq", appearance, appearance)
    appearance_similarity = (
        (appearance_similarity - config.group_appearance_floor)
        / (1.0 - config.group_appearance_floor)
    ).clamp(0.0, 1.0)
    flow_distance = (flow[:, :, None] - flow[:, None]).square().sum(dim=-1)
    motion_similarity = torch.exp(
        -flow_distance / (2.0 * config.group_motion_sigma**2)
    )
    relative = evidence.coordinates[:, :, :, None] - evidence.coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    joint = visibility[:, :, :, None] * visibility[:, :, None]
    mean_distance = (distance * joint).sum(dim=1) / joint.sum(dim=1).clamp_min(1.0)
    distance_variance = (
        (distance - mean_distance[:, None]).square() * joint
    ).sum(dim=1) / joint.sum(dim=1).clamp_min(1.0)
    rigidity = torch.exp(-distance_variance.sqrt() / config.group_distance_sigma)
    union = (
        visibility[:, :, :, None].sum(dim=1)
        + visibility[:, :, None].sum(dim=1)
        - joint.sum(dim=1)
    )
    covisibility = joint.sum(dim=1) / union.clamp_min(1.0)
    affinity = appearance_similarity * torch.maximum(motion_similarity, rigidity)
    affinity = affinity * (0.5 + 0.5 * covisibility)
    return visible_count, track_motion, affinity


def _graph_components(
    affinity: torch.Tensor,
    candidate: torch.Tensor,
    motion: torch.Tensor,
    maximum: int,
) -> list[torch.Tensor]:
    points = len(candidate)
    adjacency = (affinity >= 0.45) & candidate[:, None] & candidate[None]
    adjacency.fill_diagonal_(True)
    labels = torch.arange(points, device=affinity.device)
    sentinel = torch.full((points, points), points, device=affinity.device, dtype=torch.long)
    for _ in range(points):
        propagated = torch.where(adjacency, labels[None], sentinel).amin(dim=1)
        changed = propagated != labels
        labels = torch.minimum(labels, propagated)
        if not bool(changed.any()):
            break
    components = []
    for label in labels[candidate].unique().tolist():
        members = (labels == int(label)) & candidate
        if int(members.sum()) >= 2:
            components.append(members)
    components.sort(key=lambda members: float(motion[members].sum()), reverse=True)
    return components[:maximum]


def _teacher_owners(evidence, config, visible_count, track_motion, affinity):
    batch, points = visible_count.shape
    owners = torch.zeros(
        batch,
        points,
        config.owner_count,
        device=visible_count.device,
        dtype=torch.float32,
    )
    component_valid = torch.zeros(
        batch, config.object_slots, device=visible_count.device, dtype=torch.bool
    )
    visible_fraction = visible_count / evidence.visibility.shape[1]
    for batch_index in range(batch):
        transient = (visible_fraction[batch_index] < 0.50) & (
            track_motion[batch_index] >= 0.15
        )
        candidate = (
            (visible_count[batch_index] >= 3)
            & (track_motion[batch_index] >= 0.15)
            & ~transient
        )
        components = _graph_components(
            affinity[batch_index],
            candidate,
            track_motion[batch_index],
            config.object_slots,
        )
        assigned = torch.zeros(points, device=owners.device, dtype=torch.bool)
        for component, members in enumerate(components):
            owners[batch_index, members, component] = 1.0
            component_valid[batch_index, component] = True
            assigned |= members
        transient = transient & ~assigned
        owners[batch_index, transient, config.object_slots + 1] = 1.0
        owners[batch_index, ~assigned & ~transient, config.object_slots] = 1.0
    return owners, component_valid


@torch.no_grad()
def build_trajectory_component_teacher(
    evidence: PointTrackEvidence,
    config,
) -> TrajectoryComponentTeacher:
    visible_count, track_motion, affinity = _track_statistics(evidence, config)
    owners, component_valid = _teacher_owners(
        evidence, config, visible_count, track_motion, affinity
    )
    object_owner = owners[..., : config.object_slots]
    visibility = evidence.visibility.float()
    component_size = object_owner.sum(dim=1).clamp_min(1.0)
    identity_weight = visibility[..., None] * object_owner[:, None]
    identity = torch.einsum(
        "btpk,btpd->bkd", identity_weight, evidence.sampled_features.float()
    )
    identity = F.normalize(
        identity / identity_weight.sum(dim=(1, 2)).clamp_min(1.0)[..., None],
        dim=-1,
        eps=1e-6,
    )
    visible_mass = torch.einsum("btp,bpk->btk", visibility, object_owner)
    visibility_target = visible_mass / component_size[:, None]
    component_visible = visible_mass > 0.0
    frames = visibility.shape[1]
    first = component_visible.float().argmax(dim=1)
    last = frames - 1 - component_visible.flip(1).float().argmax(dim=1)
    time = torch.arange(frames, device=visibility.device)[None, :, None]
    known = component_valid[:, None] & (time >= first[:, None]) & (time <= last[:, None])
    presence = known.float()
    lifecycle = torch.full_like(presence, LIFECYCLE_UNKNOWN, dtype=torch.long)
    lifecycle = torch.where(known & component_visible, LIFECYCLE_VISIBLE, lifecycle)
    lifecycle = torch.where(known & ~component_visible, LIFECYCLE_OCCLUDED, lifecycle)

    geometry_weights = visibility[..., None] * object_owner[:, None]
    geometry_mass = geometry_weights.sum(dim=2)
    normalized = geometry_weights.permute(0, 1, 3, 2)
    normalized = normalized / geometry_mass[..., None].clamp_min(1e-6)
    center = torch.einsum(
        "btkp,btpd->btkd", normalized, evidence.coordinates.float()
    )
    batch, _, slots, points = normalized.shape
    scale, shape = support_shape_from_assignment(
        normalized.reshape(batch * frames, slots, points),
        evidence.coordinates.reshape(batch * frames, points, 2),
        center.reshape(batch * frames, slots, 2),
    )
    pair_visible = (
        evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    ).float()
    motion_weight = pair_visible[..., None] * object_owner[:, None]
    motion_mass = motion_weight.sum(dim=2)
    motion = torch.einsum(
        "btpk,btpd->btkd", motion_weight, evidence.residual_flow.float()
    ) / motion_mass[..., None].clamp_min(1e-6)
    return TrajectoryComponentTeacher(
        track_owner=owners,
        component_valid=component_valid,
        identity=identity,
        visibility=visibility_target,
        presence=presence,
        lifecycle_known=known,
        lifecycle_state=lifecycle,
        center=center,
        log_scale=scale.reshape(batch, frames, slots),
        support_shape=shape.reshape(batch, frames, slots, 3),
        geometry_valid=geometry_mass > 0.0,
        motion=motion,
        motion_valid=motion_mass > 0.0,
        graph_affinity=affinity,
        track_motion=track_motion,
    )
