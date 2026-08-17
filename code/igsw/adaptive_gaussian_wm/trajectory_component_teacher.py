"""Training-only trajectory component teacher independent of student assignments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .point_track_object_state import support_shape_from_assignment
from .point_track_teacher import PointTrackEvidence
from .trajectory_lifecycle import (
    LIFECYCLE_ABSENT,
    LIFECYCLE_OCCLUDED,
    LIFECYCLE_UNKNOWN,
    LIFECYCLE_VISIBLE,
    component_lifecycle_targets,
)

__all__ = [
    "LIFECYCLE_ABSENT",
    "LIFECYCLE_OCCLUDED",
    "LIFECYCLE_UNKNOWN",
    "LIFECYCLE_VISIBLE",
    "TrajectoryComponentTeacher",
    "build_trajectory_component_teacher",
]


@dataclass(frozen=True)
class TrajectoryComponentTeacher:
    track_owner: torch.Tensor
    component_valid: torch.Tensor
    component_score: torch.Tensor
    component_track_weight: torch.Tensor
    identity: torch.Tensor
    visibility: torch.Tensor
    presence: torch.Tensor
    lifecycle_known: torch.Tensor
    lifecycle_state: torch.Tensor
    center: torch.Tensor
    log_scale: torch.Tensor
    support_shape: torch.Tensor
    geometry_valid: torch.Tensor
    relative_motion: torch.Tensor
    relative_motion_valid: torch.Tensor
    geometry_residual: torch.Tensor
    geometry_residual_valid: torch.Tensor
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
    track_motion = (evidence.motion_salience.float() * flow_weight).sum(dim=1)
    track_motion = track_motion / flow_weight.sum(dim=1).clamp_min(1.0)

    appearance_similarity = torch.einsum("bpd,bqd->bpq", appearance, appearance)
    appearance_similarity = (
        (appearance_similarity - config.group_appearance_floor)
        / (1.0 - config.group_appearance_floor)
    ).clamp(0.0, 1.0)
    relative = evidence.coordinates[:, :, :, None] - evidence.coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    joint = visibility[:, :, :, None] * visibility[:, :, None]
    mean_distance = (distance * joint).sum(dim=1) / joint.sum(dim=1).clamp_min(1.0)
    distance_variance = (
        (distance - mean_distance[:, None]).square() * joint
    ).sum(dim=1) / joint.sum(dim=1).clamp_min(1.0)
    rigidity = torch.exp(-distance_variance.sqrt() / config.group_distance_sigma)
    locality = torch.exp(
        -mean_distance.square() / (2.0 * config.group_locality_sigma**2)
    )
    union = (
        visibility[:, :, :, None].sum(dim=1)
        + visibility[:, :, None].sum(dim=1)
        - joint.sum(dim=1)
    )
    covisibility = joint.sum(dim=1) / union.clamp_min(1.0)
    affinity = appearance_similarity * rigidity
    affinity = affinity * (0.25 + 0.75 * locality) * (0.5 + 0.5 * covisibility)
    affinity.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    return visible_count, track_motion, affinity


def _graph_components(
    affinity: torch.Tensor,
    candidate: torch.Tensor,
    visible_fraction: torch.Tensor,
    config,
) -> list[tuple[torch.Tensor, float]]:
    points = len(candidate)
    adjacency = (
        (affinity >= config.component_affinity_threshold)
        & candidate[:, None]
        & candidate[None]
    )
    adjacency.fill_diagonal_(True)
    labels = torch.arange(points, device=affinity.device)
    sentinel = torch.full((points, points), points, device=affinity.device, dtype=torch.long)
    for _ in range(points):
        propagated = torch.where(adjacency, labels[None], sentinel).amin(dim=1)
        updated = torch.minimum(labels, propagated)
        if bool((updated == labels).all()):
            break
        labels = updated
    maximum_tracks = max(
        config.component_min_tracks,
        int(points * config.component_max_track_fraction),
    )
    components: list[tuple[torch.Tensor, float]] = []
    for label in labels[candidate].unique().tolist():
        members = (labels == int(label)) & candidate
        count = int(members.sum())
        if count < config.component_min_tracks or count > maximum_tracks:
            continue
        internal = affinity[members][:, members]
        off_diagonal = ~torch.eye(count, device=affinity.device, dtype=torch.bool)
        cohesion = internal[off_diagonal].mean() if count > 1 else internal.mean()
        persistence = visible_fraction[members].mean()
        compactness = internal.mean()
        score = float(0.50 * cohesion + 0.30 * persistence + 0.20 * compactness)
        components.append((members, score))
    components.sort(key=lambda item: item[1], reverse=True)
    return components[: config.object_slots]


def _teacher_owners(evidence, config, visible_count, affinity):
    batch, points = visible_count.shape
    owners = torch.zeros(
        batch, points, config.owner_count, device=visible_count.device, dtype=torch.float32
    )
    component_valid = torch.zeros(
        batch, config.object_slots, device=visible_count.device, dtype=torch.bool
    )
    component_score = torch.zeros(
        batch, config.object_slots, device=visible_count.device, dtype=torch.float32
    )
    visible_fraction = visible_count / evidence.visibility.shape[1]
    for batch_index in range(batch):
        persistent = (
            (visible_count[batch_index] >= config.component_min_visible_frames)
            & (visible_fraction[batch_index] >= config.component_min_visible_fraction)
        )
        transient = (visible_count[batch_index] > 0) & ~persistent
        components = _graph_components(
            affinity[batch_index], persistent, visible_fraction[batch_index], config
        )
        assigned = torch.zeros(points, device=owners.device, dtype=torch.bool)
        for component, (members, score) in enumerate(components):
            owners[batch_index, members, component] = 1.0
            component_valid[batch_index, component] = True
            component_score[batch_index, component] = score
            assigned |= members
        transient &= ~assigned
        owners[batch_index, transient, config.object_slots + 1] = 1.0
        owners[batch_index, ~assigned & ~transient, config.object_slots] = 1.0
    return owners, component_valid, component_score


def _component_geometry(evidence, object_owner):
    visibility = evidence.visibility.float()
    weights = visibility[..., None] * object_owner[:, None]
    mass = weights.sum(dim=2)
    normalized = weights.permute(0, 1, 3, 2)
    normalized = normalized / mass[..., None].clamp_min(1e-6)
    center = torch.einsum("btkp,btpd->btkd", normalized, evidence.coordinates.float())
    batch, frames, slots, points = normalized.shape
    scale, shape = support_shape_from_assignment(
        normalized.reshape(batch * frames, slots, points),
        evidence.coordinates.reshape(batch * frames, points, 2),
        center.reshape(batch * frames, slots, 2),
    )
    return (
        center,
        scale.reshape(batch, frames, slots),
        shape.reshape(batch, frames, slots, 3),
        mass > 0.0,
    )


def _multi_horizon_targets(
    evidence,
    object_owner,
    scene_owner,
    center,
    log_scale,
    support_shape,
    geometry_valid,
    frame_times,
    horizons,
):
    batch, frames, slots = center.shape[:3]
    horizon_count = len(horizons)
    motion = center.new_zeros(batch, frames, slots, horizon_count, 2)
    motion_valid = torch.zeros(
        batch, frames, slots, horizon_count, device=center.device, dtype=torch.bool
    )
    geometry = center.new_zeros(batch, frames, slots, horizon_count, 6)
    geometry_residual_valid = torch.zeros_like(motion_valid)
    for horizon_index, horizon in enumerate(horizons):
        if horizon >= frames:
            continue
        pair = evidence.visibility[:, :-horizon] & evidence.visibility[:, horizon:]
        displacement = (
            evidence.coordinates[:, horizon:].float()
            - evidence.coordinates[:, :-horizon].float()
        )
        pair_weight = pair.float()
        scene_weight = pair_weight * scene_owner[:, None]
        scene_mass = scene_weight.sum(dim=2)
        all_mass = pair_weight.sum(dim=2)
        scene_displacement = (displacement * scene_weight[..., None]).sum(dim=2)
        all_displacement = (displacement * pair_weight[..., None]).sum(dim=2)
        global_displacement = torch.where(
            (scene_mass > 0.0)[..., None],
            scene_displacement / scene_mass.clamp_min(1.0)[..., None],
            all_displacement / all_mass.clamp_min(1.0)[..., None],
        )
        residual = displacement - global_displacement[:, :, None]
        component_weight = pair_weight[..., None] * object_owner[:, None]
        component_mass = component_weight.sum(dim=2)
        delta_time = (frame_times[:, horizon:] - frame_times[:, :-horizon]).clamp_min(1e-4)
        component_motion = torch.einsum(
            "btpk,btpd->btkd", component_weight, residual
        ) / component_mass[..., None].clamp_min(1e-6)
        component_motion = component_motion / delta_time[..., None, None]
        motion[:, :-horizon, :, horizon_index] = component_motion
        valid = component_mass > 0.0
        motion_valid[:, :-horizon, :, horizon_index] = valid
        center_delta = (
            center[:, horizon:] - center[:, :-horizon] - global_displacement[:, :, None]
        ) / delta_time[..., None, None]
        scale_delta = (
            log_scale[:, horizon:] - log_scale[:, :-horizon]
        ) / delta_time[..., None]
        shape_delta = (
            support_shape[:, horizon:] - support_shape[:, :-horizon]
        ) / delta_time[..., None, None]
        geometry[:, :-horizon, :, horizon_index] = torch.cat(
            (center_delta, scale_delta[..., None], shape_delta), dim=-1
        )
        geometry_residual_valid[:, :-horizon, :, horizon_index] = (
            geometry_valid[:, :-horizon] & geometry_valid[:, horizon:] & valid
        )
    return motion, motion_valid, geometry, geometry_residual_valid


@torch.no_grad()
def build_trajectory_component_teacher(
    evidence: PointTrackEvidence,
    config,
    frame_times: torch.Tensor,
) -> TrajectoryComponentTeacher:
    visible_count, track_motion, affinity = _track_statistics(evidence, config)
    owners, component_valid, component_score = _teacher_owners(
        evidence, config, visible_count, affinity
    )
    object_owner = owners[..., : config.object_slots]
    component_size = object_owner.sum(dim=1).clamp_min(1.0)
    component_track_weight = object_owner / component_size[:, None]
    identity_weight = evidence.visibility.float()[..., None] * object_owner[:, None]
    identity = torch.einsum(
        "btpk,btpd->bkd", identity_weight, evidence.sampled_features.float()
    )
    identity = F.normalize(
        identity / identity_weight.sum(dim=(1, 2)).clamp_min(1.0)[..., None],
        dim=-1,
        eps=1e-6,
    )
    center, log_scale, support_shape, geometry_valid = _component_geometry(
        evidence, object_owner
    )
    lifecycle = component_lifecycle_targets(
        evidence.visibility, object_owner, component_valid, config
    )
    motion, motion_valid, geometry_residual, geometry_residual_valid = (
        _multi_horizon_targets(
            evidence,
            object_owner,
            owners[..., config.object_slots],
            center,
            log_scale,
            support_shape,
            geometry_valid,
            frame_times,
            config.dynamic_horizons,
        )
    )
    return TrajectoryComponentTeacher(
        track_owner=owners,
        component_valid=component_valid,
        component_score=component_score,
        component_track_weight=component_track_weight,
        identity=identity,
        visibility=lifecycle.visibility,
        presence=lifecycle.presence,
        lifecycle_known=lifecycle.known,
        lifecycle_state=lifecycle.state,
        center=center,
        log_scale=log_scale,
        support_shape=support_shape,
        geometry_valid=geometry_valid,
        relative_motion=motion,
        relative_motion_valid=motion_valid,
        geometry_residual=geometry_residual,
        geometry_residual_valid=geometry_residual_valid,
        graph_affinity=affinity,
        track_motion=track_motion,
    )
