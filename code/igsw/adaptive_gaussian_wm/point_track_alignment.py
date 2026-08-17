"""Permutation matching from independent trajectory components to student slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .point_track_teacher import PointTrackEvidence, sample_patch_field
from .trajectory_component_teacher import TrajectoryComponentTeacher


@dataclass(frozen=True)
class TeacherStudentMatch:
    sampled_student_assignment: torch.Tensor
    target_student_owner: torch.Tensor
    permutation: torch.Tensor
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


def _hungarian_minimize(cost: torch.Tensor) -> torch.Tensor:
    size = cost.shape[0]
    values = cost.detach().float().cpu().tolist()
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = p[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = values[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = torch.zeros(size, size, device=cost.device, dtype=torch.float32)
    for teacher_column in range(1, size + 1):
        student_row = p[teacher_column]
        assignment[student_row - 1, teacher_column - 1] = 1.0
    return assignment


def _match_components(
    sampled_student: torch.Tensor,
    teacher: TrajectoryComponentTeacher,
    visibility: torch.Tensor,
    object_slots: int,
) -> torch.Tensor:
    student = sampled_student[..., :object_slots].float()
    teacher_owner = teacher.track_owner[..., :object_slots].float()
    intersection = torch.einsum(
        "btps,bpc,btp->bsc",
        student.detach(),
        teacher_owner,
        visibility.float(),
    )
    teacher_mass = torch.einsum(
        "bpc,btp->bc", teacher_owner, visibility.float()
    ).clamp_min(1.0)
    student_mass = torch.einsum(
        "btps,btp->bs", student.detach(), visibility.float()
    ).clamp_min(1.0)
    coverage = intersection / teacher_mass[:, None]
    precision = intersection / student_mass[:, :, None]
    dice = 2.0 * intersection / (
        teacher_mass[:, None] + student_mass[:, :, None]
    )
    score = 0.50 * dice + 0.25 * coverage + 0.25 * precision
    invalid_cost = (~teacher.component_valid).float()[:, None]
    cost = -score + invalid_cost
    return torch.stack(
        [
            _hungarian_minimize(cost[index])
            for index in range(len(cost))
        ]
    )


def _match_static(permutation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("unsupported component teacher target rank")
    return torch.einsum("bsc,bc...->bs...", permutation, value)


def _match_temporal(permutation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 3:
        raise ValueError("unsupported temporal component teacher target rank")
    return torch.einsum("bsc,btc...->bts...", permutation, value)


def match_trajectory_teacher_to_student(
    state: dict[str, torch.Tensor],
    evidence: PointTrackEvidence,
    teacher: TrajectoryComponentTeacher,
    grid_hw: tuple[int, int],
    object_slots: int,
) -> TeacherStudentMatch:
    sampled = sample_patch_field(
        state["assignment"].float(), evidence.coordinates, grid_hw
    ).clamp_min(0.0)
    permutation = _match_components(
        sampled, teacher, evidence.visibility, object_slots
    )
    target_objects = torch.einsum(
        "bpc,bsc->bps",
        teacher.track_owner[..., :object_slots],
        permutation,
    )
    component_track_weight = torch.einsum(
        "bpc,bsc->bps", teacher.component_track_weight, permutation
    )
    target_owner = torch.cat(
        (
            target_objects,
            teacher.track_owner[..., object_slots : object_slots + 2],
        ),
        dim=-1,
    )
    return TeacherStudentMatch(
        sampled_student_assignment=sampled,
        target_student_owner=target_owner,
        permutation=permutation,
        component_valid=_match_static(permutation, teacher.component_valid.float()).bool(),
        component_score=_match_static(permutation, teacher.component_score),
        component_track_weight=component_track_weight,
        identity=_match_static(permutation, teacher.identity),
        visibility=_match_temporal(permutation, teacher.visibility),
        presence=_match_temporal(permutation, teacher.presence),
        lifecycle_known=_match_temporal(permutation, teacher.lifecycle_known.float()).bool(),
        lifecycle_state=_match_temporal(permutation, teacher.lifecycle_state.float()).long(),
        center=_match_temporal(permutation, teacher.center),
        log_scale=_match_temporal(permutation, teacher.log_scale),
        support_shape=_match_temporal(permutation, teacher.support_shape),
        geometry_valid=_match_temporal(permutation, teacher.geometry_valid.float()).bool(),
        relative_motion=_match_temporal(permutation, teacher.relative_motion),
        relative_motion_valid=_match_temporal(
            permutation, teacher.relative_motion_valid.float()
        ).bool(),
        geometry_residual=_match_temporal(permutation, teacher.geometry_residual),
        geometry_residual_valid=_match_temporal(
            permutation, teacher.geometry_residual_valid.float()
        ).bool(),
    )
