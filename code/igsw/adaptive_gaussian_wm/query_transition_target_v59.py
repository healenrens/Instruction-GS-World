"""Training-only object targets built from related tracks and frozen DINO features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .query_object_teacher_v57 import QueryObjectTeacher


@dataclass(frozen=True)
class ObjectTargetSequence:
    semantic: torch.Tensor
    geometry: torch.Tensor
    visibility: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class QueryTransitionTarget:
    source_semantic: torch.Tensor
    source_geometry: torch.Tensor
    source_visibility: torch.Tensor
    future_semantic: torch.Tensor
    future_geometry: torch.Tensor
    future_visibility: torch.Tensor
    delta_seconds: torch.Tensor
    pair_valid: torch.Tensor
    motion_active: torch.Tensor


def _query_relation(values: torch.Tensor, query_index: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(len(values), device=values.device)
    return values[batch, query_index]


def _object_target_sequence(evidence, relation, binding: QueryObjectTeacher):
    coordinates = evidence.coordinates.float()
    visible = evidence.visibility.float()
    same = _query_relation(relation.same_confidence.float(), binding.query_index)
    query = F.one_hot(binding.query_index, coordinates.shape[2]).float()
    membership = torch.maximum(same, query) * binding.query_valid[:, None].float()

    object_weight = visible * membership[:, None]
    object_mass = object_weight.sum(dim=-1)
    normalized = object_weight / object_mass[..., None].clamp_min(1e-6)
    semantic = torch.einsum(
        "btp,btpd->btd", normalized, evidence.sampled_features.float()
    )
    semantic = F.normalize(semantic, dim=-1, eps=1e-6)
    center = torch.einsum("btp,btpd->btd", normalized, coordinates)
    offset = coordinates - center[:, :, None]
    covariance = torch.einsum("btp,btpi,btpj->btij", normalized, offset, offset)

    scene_weight = visible / visible.sum(dim=-1, keepdim=True).clamp_min(1.0)
    scene_center = torch.einsum("btp,btpd->btd", scene_weight, coordinates)
    relative_center = center - scene_center
    geometry = torch.stack(
        (
            relative_center[..., 0],
            relative_center[..., 1],
            covariance[..., 0, 0],
            covariance[..., 1, 1],
            covariance[..., 0, 1],
        ),
        dim=-1,
    )
    visibility = object_mass / membership.sum(dim=-1)[:, None].clamp_min(1e-6)
    valid = (object_mass > 0.0) & binding.query_valid[:, None]
    tensors = semantic, geometry, visibility
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("v59 object transition target contains non-finite values")
    return ObjectTargetSequence(
        semantic=semantic.detach(),
        geometry=geometry.detach(),
        visibility=visibility.detach(),
        valid=valid.detach(),
    )


def build_query_transition_target_v59(
    evidence,
    relation,
    binding: QueryObjectTeacher,
    frame_times: torch.Tensor,
    observed_frames: int,
    config,
) -> QueryTransitionTarget:
    sequence = _object_target_sequence(evidence, relation, binding)
    source_index = observed_frames - 1
    horizons = torch.tensor(
        config.dynamic_horizons, device=frame_times.device, dtype=torch.long
    )
    future_indices = source_index + horizons
    if int(future_indices.max()) >= sequence.semantic.shape[1]:
        raise ValueError("v59 dynamic horizon exceeds teacher future frames")

    source_semantic = sequence.semantic[:, source_index]
    source_geometry = sequence.geometry[:, source_index]
    source_visibility = sequence.visibility[:, source_index]
    future_semantic = sequence.semantic[:, future_indices]
    future_geometry = sequence.geometry[:, future_indices]
    future_visibility = sequence.visibility[:, future_indices]
    delta_seconds = frame_times[:, future_indices] - frame_times[:, source_index, None]
    if bool((delta_seconds <= 0.0).any()):
        raise ValueError("v59 transition target requires positive time gaps")

    pair_valid = (
        sequence.valid[:, source_index, None] & sequence.valid[:, future_indices]
    )
    center_change = (future_geometry[..., :2] - source_geometry[:, None, :2]).norm(
        dim=-1
    )
    shape_change = (future_geometry[..., 2:] - source_geometry[:, None, 2:]).norm(
        dim=-1
    )
    semantic_change = 1.0 - torch.einsum("bd,bkd->bk", source_semantic, future_semantic)
    visibility_change = (future_visibility - source_visibility[:, None]).abs()
    motion_active = pair_valid & (
        (center_change >= config.motion_center_threshold)
        | (shape_change >= config.motion_shape_threshold)
        | (semantic_change >= config.motion_semantic_threshold)
        | (visibility_change >= 0.5)
    )
    return QueryTransitionTarget(
        source_semantic=source_semantic,
        source_geometry=source_geometry,
        source_visibility=source_visibility,
        future_semantic=future_semantic,
        future_geometry=future_geometry,
        future_visibility=future_visibility,
        delta_seconds=delta_seconds.detach(),
        pair_valid=pair_valid.detach(),
        motion_active=motion_active.detach(),
    )
