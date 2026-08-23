"""Training-only lifecycle and motion targets for persistent query state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .point_track_teacher import PointTrackEvidence
from .query_object_teacher_v57 import (
    QueryObjectTeacher,
    build_query_object_teacher_v57,
    observed_evidence_prefix,
    query_teacher_contract_metrics,
)
from .trajectory_relation_teacher import TrajectoryRelationTeacher


@dataclass(frozen=True)
class QueryPersistentTeacher:
    binding: QueryObjectTeacher
    track_coordinate: torch.Tensor
    visibility_target: torch.Tensor
    lifecycle_known: torch.Tensor
    occluded_candidate: torch.Tensor
    unknown: torch.Tensor
    motion_target: torch.Tensor
    motion_valid: torch.Tensor
    delta_seconds: torch.Tensor
    query_time: torch.Tensor

    def __getattr__(self, name):
        binding = object.__getattribute__(self, "binding")
        if name in binding.__dataclass_fields__:
            return getattr(binding, name)
        raise AttributeError(name)


def _gather_track(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(len(values), device=values.device)
    return values[batch, :, indices]


def _enclosed_occlusion(visibility: torch.Tensor) -> torch.Tensor:
    visible = visibility.bool()
    empty = torch.zeros_like(visible[:, :1])
    seen_before = torch.cat(
        (empty, visible[:, :-1].cumsum(dim=1) > 0), dim=1
    )
    seen_after = torch.cat(
        (visible[:, 1:].flip(1).cumsum(dim=1).flip(1) > 0, empty), dim=1
    )
    return (~visible) & seen_before & seen_after


def build_query_persistent_teacher_v58(
    evidence: PointTrackEvidence,
    relation: TrajectoryRelationTeacher,
    config,
    frame_times: torch.Tensor,
    observed_frames: int,
) -> QueryPersistentTeacher:
    observed = observed_evidence_prefix(evidence, observed_frames)
    observed_visibility = observed.visibility.bool()
    enclosed_occlusion = _enclosed_occlusion(observed_visibility)
    preferred_query = enclosed_occlusion.any(dim=1)
    binding = build_query_object_teacher_v57(
        evidence,
        relation,
        config,
        observed_frames=observed_frames,
        preferred_query=preferred_query,
    )
    visibility = _gather_track(observed.visibility, binding.query_index).bool()
    coordinates = _gather_track(observed.coordinates, binding.query_index).float()
    query_times = evidence.query_times.to(binding.query_index.device)
    query_time = query_times[binding.query_index]
    occluded = _enclosed_occlusion(visibility) & binding.query_valid[:, None]
    known = (visibility | occluded) & binding.query_valid[:, None]
    unknown = binding.query_valid[:, None] & ~known

    motion = coordinates.new_zeros(len(coordinates), observed_frames, 2)
    motion_valid = torch.zeros_like(visibility)
    if observed_frames > 1:
        residual = _gather_track(
            observed.residual_flow, binding.query_index
        ).float()
        pair_visible = visibility[:, 1:] & visibility[:, :-1]
        delta_seconds = frame_times[:, :observed_frames] - frame_times[:, :1]
        pair_seconds = delta_seconds[:, 1:] - delta_seconds[:, :-1]
        motion[:, 1:] = residual / pair_seconds[..., None].clamp_min(1e-4)
        motion_valid[:, 1:] = pair_visible & binding.query_valid[:, None]
    else:
        delta_seconds = frame_times[:, :1] - frame_times[:, :1]
    return QueryPersistentTeacher(
        binding=binding,
        track_coordinate=coordinates.detach(),
        visibility_target=visibility.float().detach(),
        lifecycle_known=known.detach(),
        occluded_candidate=occluded.detach(),
        unknown=unknown.detach(),
        motion_target=motion.detach(),
        motion_valid=motion_valid.detach(),
        delta_seconds=delta_seconds.detach(),
        query_time=query_time.detach(),
    )


def persistent_teacher_contract_metrics(teacher: QueryPersistentTeacher):
    binding = query_teacher_contract_metrics(teacher.binding)
    known = teacher.lifecycle_known.float()
    visible = teacher.visibility_target * known
    occluded = teacher.occluded_candidate.float()
    return {
        **binding,
        "lifecycle_known_fraction": known.mean(),
        "lifecycle_visible_fraction": visible.mean(),
        "lifecycle_occluded_candidate_fraction": occluded.mean(),
        "lifecycle_unknown_fraction": teacher.unknown.float().mean(),
        "motion_valid_fraction": teacher.motion_valid.float().mean(),
    }
