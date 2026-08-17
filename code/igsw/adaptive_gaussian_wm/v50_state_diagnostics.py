"""Held-video diagnostics for the v50 RGB-only Object State student."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .point_track_teacher import PointTrackEvidence
from .trajectory_component_teacher import (
    LIFECYCLE_ABSENT,
    LIFECYCLE_OCCLUDED,
    LIFECYCLE_VISIBLE,
)


@dataclass(frozen=True)
class DiagnosticBatch:
    weighted: dict[str, tuple[float, float]]
    totals: dict[str, float]
    probes: dict[str, torch.Tensor]


class EvaluationAggregate:
    def __init__(self):
        self.weighted_sum: dict[str, float] = {}
        self.weight: dict[str, float] = {}
        self.totals: dict[str, float] = {}
        self.probes: dict[str, list[torch.Tensor]] = {}
        self.causal_max = 0.0
        self.order_sum = {
            "identity_order_change": 0.0,
            "dynamic_order_change": 0.0,
        }
        self.order_count = 0
        self.items = 0

    def add_mean(self, name: str, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        self.weighted_sum[name] = self.weighted_sum.get(name, 0.0) + value * weight
        self.weight[name] = self.weight.get(name, 0.0) + weight

    def add_total(self, name: str, value: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + value

    def means(self) -> dict[str, float]:
        return {
            name: value / self.weight[name] for name, value in self.weighted_sum.items()
        }


def _weighted(value: torch.Tensor, weight: torch.Tensor) -> tuple[float, float]:
    count = float(weight.float().sum())
    if count == 0.0:
        return 0.0, 0.0
    mean = float((value.float() * weight.float()).sum() / count)
    return mean, count


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-6)


def _track_correspondence(assignment, visibility):
    pair = (visibility[:, 1:] & visibility[:, :-1]).float()
    correct = _weighted(_cosine(assignment[:, :-1], assignment[:, 1:]), pair)
    shuffled = _weighted(
        _cosine(assignment[:, :-1], assignment[:, 1:].roll(1, dims=2)), pair
    )
    return correct, shuffled


def _track_reappearance(assignment, visibility):
    batch, frames, points = visibility.shape
    last = torch.zeros(batch, points, assignment.shape[-1], device=assignment.device)
    seen = torch.zeros(batch, points, device=assignment.device, dtype=torch.bool)
    gap = torch.zeros_like(seen)
    correct_sum = assignment.new_zeros((), dtype=torch.float32)
    shuffled_sum = assignment.new_zeros((), dtype=torch.float32)
    events = assignment.new_zeros((), dtype=torch.float32)
    for frame in range(frames):
        current = visibility[:, frame]
        reappeared = current & seen & gap
        if bool(reappeared.any()):
            after = assignment[:, frame]
            correct_sum += (_cosine(last, after) * reappeared).sum()
            shuffled_sum += (_cosine(last, after.roll(1, dims=1)) * reappeared).sum()
            events += reappeared.float().sum()
        last = torch.where(current[..., None], assignment[:, frame], last)
        gap = torch.where(current, torch.zeros_like(gap), gap | seen)
        seen = seen | current
    count = float(events)
    if count == 0.0:
        return (0.0, 0.0), (0.0, 0.0), 0.0
    return (
        (float(correct_sum / events), count),
        (float(shuffled_sum / events), count),
        count,
    )


def _component_reappearance(model, state, match):
    prediction = F.normalize(
        model.identity_readout(state["identity"].float()), dim=-1, eps=1e-6
    )
    target = match.identity[:, None]
    visible = match.visibility * match.component_valid[:, None].float()
    occluded_before = (match.lifecycle_state == LIFECYCLE_OCCLUDED).float().cumsum(
        dim=1
    ) > 0
    weight = visible * occluded_before.float()
    correct = _weighted(_cosine(prediction, target), weight)
    shuffled = _weighted(_cosine(prediction, target.roll(1, dims=2)), weight)
    return correct, shuffled, float(weight.sum())


def _coverage_metrics(model, evidence, teacher, match):
    objects = match.sampled_student_assignment[..., : model.config.object_slots]
    student_object = objects.sum(dim=-1)
    target_object = match.target_student_owner[..., : model.config.object_slots].sum(-1)
    visible = evidence.visibility.float()
    per_track_student = (student_object * visible).sum(dim=1) / visible.sum(
        dim=1
    ).clamp_min(1.0)
    track_valid = visible.sum(dim=1) > 0
    static = track_valid & (teacher.track_motion < 0.05)
    moving = track_valid & (teacher.track_motion >= 0.15)
    weighted = {
        "student_object_track_fraction": _weighted(student_object, visible),
        "teacher_object_track_fraction": _weighted(
            target_object[:, None].expand_as(student_object), visible
        ),
        "static_student_object_fraction": _weighted(per_track_student, static.float()),
        "static_teacher_object_fraction": _weighted(target_object, static.float()),
        "moving_student_object_fraction": _weighted(per_track_student, moving.float()),
        "moving_teacher_object_fraction": _weighted(target_object, moving.float()),
    }
    slot_mass = (objects * visible[..., None]).sum(dim=(1, 2))
    share = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    effective = torch.exp(-(share * share.clamp_min(1e-8).log()).sum(dim=-1))
    active = (share >= 0.01).float().sum(dim=-1)
    batch_weight = torch.ones_like(effective)
    weighted.update(
        slot_effective_count=_weighted(effective, batch_weight),
        slot_active_count=_weighted(active, batch_weight),
        slot_max_fraction=_weighted(share.max(dim=-1).values, batch_weight),
    )
    return weighted


def _state_change_metrics(state, match):
    visible_pair = (match.geometry_valid[:, 1:] & match.geometry_valid[:, :-1]).float()
    identity_drift = 1.0 - _cosine(state["identity"][:, :-1], state["identity"][:, 1:])
    dynamic_change = 1.0 - _cosine(state["dynamic"][:, :-1], state["dynamic"][:, 1:])
    return {
        "identity_temporal_drift": _weighted(identity_drift, visible_pair),
        "dynamic_temporal_change": _weighted(dynamic_change, visible_pair),
    }


def deletion_locality(model, features, evidence, output, amp_context):
    state, match = output["state"], output["match"]
    inside_sum = outside_sum = valid_items = 0.0
    for batch_index in range(len(state["identity"])):
        frame_motion = torch.cat(
            (
                evidence.motion_salience[batch_index],
                torch.zeros_like(evidence.motion_salience[batch_index, :1]),
            ),
            dim=0,
        ).sum(dim=1)
        frame = int(frame_motion.argmax())
        teacher_owner = match.target_student_owner[
            batch_index, :, : model.config.object_slots
        ]
        track_motion = evidence.motion_salience[batch_index].mean(dim=0)
        slot = int(torch.einsum("p,pk->k", track_motion, teacher_owner).argmax())
        frame_state = {
            name: value[batch_index : batch_index + 1, frame]
            for name, value in state.items()
            if name not in ("assignment", "mass")
        }
        coordinates = features.coordinates[batch_index : batch_index + 1, frame]
        valid = features.valid[batch_index : batch_index + 1, frame]
        with amp_context():
            reference, _ = model.decoder(frame_state, coordinates, valid)
            object_valid = torch.ones(
                1,
                model.config.object_slots,
                device=reference.device,
                dtype=torch.bool,
            )
            object_valid[:, slot] = False
            deleted, _ = model.decoder(frame_state, coordinates, valid, object_valid)
        change = 1.0 - _cosine(reference, deleted)
        assigned = (teacher_owner[:, slot] > 0.5) & evidence.visibility[
            batch_index, frame
        ]
        if not bool(assigned.any()):
            continue
        track_positions = evidence.coordinates[batch_index, frame, assigned]
        patch_positions = features.coordinates[batch_index, frame]
        distance = (
            (patch_positions[:, None] - track_positions[None]).norm(dim=-1).amin(dim=1)
        )
        inside = (distance <= 0.18) & valid[0]
        outside = (distance >= 0.30) & valid[0]
        if not bool(inside.any()) or not bool(outside.any()):
            continue
        inside_sum += float(change[0, inside].mean())
        outside_sum += float(change[0, outside].mean())
        valid_items += 1.0
    return inside_sum, outside_sum, valid_items


def causal_prefix_difference(model, features, frame_times, full_state) -> float:
    prefix = max(2, features.patches.shape[1] // 2)
    _, prefix_state = model.encode_student(
        features.patches[:, :prefix],
        features.coordinates[:, :prefix],
        features.valid[:, :prefix],
        frame_times[:, :prefix],
    )
    names = (
        "identity",
        "dynamic",
        "center",
        "log_scale",
        "support_shape",
        "presence",
        "visibility",
        "scene",
        "transient",
        "assignment",
        "mass",
    )
    return max(
        float((prefix_state[name] - full_state[name][:, :prefix]).abs().max())
        for name in names
    )


def order_sensitivity(model, features, frame_times, full_state):
    _, reversed_state = model.encode_student(
        features.patches.flip(1),
        features.coordinates.flip(1),
        features.valid.flip(1),
        frame_times,
    )
    return {
        "identity_order_change": float(
            (
                1.0
                - _cosine(
                    full_state["identity"][:, -1], reversed_state["identity"][:, -1]
                )
            ).mean()
        ),
        "dynamic_order_change": float(
            (
                1.0
                - _cosine(
                    full_state["dynamic"][:, -1], reversed_state["dynamic"][:, -1]
                )
            ).mean()
        ),
    }


def collect_batch_diagnostics(
    model, features, evidence: PointTrackEvidence, output, amp_context
):
    state, teacher, match = output["state"], output["teacher"], output["match"]
    assignment = match.sampled_student_assignment.float()
    correct, shuffled = _track_correspondence(assignment, evidence.visibility)
    reappear_correct, reappear_shuffled, track_events = _track_reappearance(
        assignment, evidence.visibility
    )
    component_correct, component_shuffled, component_events = _component_reappearance(
        model, state, match
    )
    weighted = {
        "track_assignment_correct_cosine": correct,
        "track_assignment_shuffled_cosine": shuffled,
        "track_reappearance_correct_cosine": reappear_correct,
        "track_reappearance_shuffled_cosine": reappear_shuffled,
        "component_reappearance_correct_cosine": component_correct,
        "component_reappearance_shuffled_cosine": component_shuffled,
        **_coverage_metrics(model, evidence, teacher, match),
        **_state_change_metrics(state, match),
    }
    lifecycle = match.lifecycle_state
    totals = {
        "track_reappearance_events": track_events,
        "component_reappearance_events": component_events,
        "lifecycle_visible": float((lifecycle == LIFECYCLE_VISIBLE).sum()),
        "lifecycle_occluded": float((lifecycle == LIFECYCLE_OCCLUDED).sum()),
        "lifecycle_absent": float((lifecycle == LIFECYCLE_ABSENT).sum()),
        "lifecycle_unknown": float((~match.lifecycle_known).sum()),
        "lifecycle_total": float(lifecycle.numel()),
    }
    inside, outside, valid_items = deletion_locality(
        model, features, evidence, output, amp_context
    )
    totals.update(
        deletion_inside_sum=inside,
        deletion_outside_sum=outside,
        deletion_valid_items=valid_items,
    )
    probes = {
        "identity": state["identity"][:, :-1]
        .reshape(-1, state["identity"].shape[-1])
        .detach()
        .cpu(),
        "dynamic": state["dynamic"][:, :-1]
        .reshape(-1, state["dynamic"].shape[-1])
        .detach()
        .cpu(),
        "motion": match.motion.reshape(-1, 2).detach().cpu(),
        "motion_weight": match.motion_valid.reshape(-1).float().detach().cpu(),
        "visibility": match.visibility[:, :-1].reshape(-1, 1).detach().cpu(),
        "visibility_weight": match.lifecycle_known[:, :-1]
        .reshape(-1)
        .float()
        .detach()
        .cpu(),
    }
    return DiagnosticBatch(weighted=weighted, totals=totals, probes=probes)


def ridge_relative_gain(features, target, weight) -> float:
    keep = weight > 1e-4
    x, y = features[keep].float(), target[keep].float()
    if len(x) < 16:
        return -1.0
    order = torch.arange(len(x))
    train, test = order % 2 == 0, order % 2 == 1
    mean = x[train].mean(0)
    scale = x[train].std(0, unbiased=False).clamp_min(1e-4)
    normalized = (x - mean) / scale
    normalized = torch.cat((normalized, torch.ones(len(normalized), 1)), dim=1)
    xtx = normalized[train].T @ normalized[train]
    ridge = 1e-2 * torch.eye(len(xtx))
    coefficient = torch.linalg.solve(xtx + ridge, normalized[train].T @ y[train])
    prediction = normalized[test] @ coefficient
    mse = (prediction - y[test]).square().mean()
    baseline = (y[test] - y[train].mean(0)).square().mean().clamp_min(1e-8)
    return float((baseline - mse) / baseline)
