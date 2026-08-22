"""Held-video diagnostics for the v56 RGB-only Object State student."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from .object_state_target_v52 import visible_track_mean
from .verified_relation_objective_v56 import (
    verified_relation_object_state_terms,
)
from .v56_state_probe_metrics import (
    aligned_state_probe_tensors,
    finalize_aligned_state_probes,
)


@dataclass(frozen=True)
class V56DiagnosticBatch:
    weighted: dict[str, tuple[float, float]]
    totals: dict[str, float]
    probes: dict[str, torch.Tensor]


class V56EvaluationAggregate:
    def __init__(self) -> None:
        self.weighted_sum: dict[str, float] = {}
        self.weight: dict[str, float] = {}
        self.totals: dict[str, float] = {}
        self.probes: dict[str, list[torch.Tensor]] = {}
        self.items = 0
        self.causal_max = 0.0

    def add_mean(self, name: str, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        self.weighted_sum[name] = self.weighted_sum.get(name, 0.0) + value * weight
        self.weight[name] = self.weight.get(name, 0.0) + weight

    def add_total(self, name: str, value: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + value

    def add_batch(self, diagnostics: V56DiagnosticBatch) -> None:
        for name, (value, weight) in diagnostics.weighted.items():
            self.add_mean(name, value, weight)
        for name, value in diagnostics.totals.items():
            self.add_total(name, value)
        for name, value in diagnostics.probes.items():
            self.probes.setdefault(name, []).append(value)

    def means(self) -> dict[str, float]:
        return {
            name: value / self.weight[name]
            for name, value in self.weighted_sum.items()
        }


def _weighted(value: torch.Tensor, weight: torch.Tensor) -> tuple[float, float]:
    weight = weight.float()
    count = float(weight.sum())
    if count == 0.0:
        return 0.0, 0.0
    return float((value.float() * weight).sum() / count), count


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-6)


def conditional_object_assignment(assignment, object_slots):
    objects = assignment[..., :object_slots].float()
    probability = objects.sum(dim=-1, keepdim=True)
    return objects / probability.clamp_min(1e-6), probability[..., 0]


def visible_mask(teacher):
    return teacher.visibility.float() >= 0.5


def _relation_metrics(conditional, teacher):
    track = visible_track_mean(conditional, teacher.visibility)
    track = track / track.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    similarity = torch.einsum("bpk,bqk->bpq", track, track)
    support = teacher.object_confidence
    pair = support[:, :, None] * support[:, None]
    same_weight = teacher.same_confidence * pair
    different_weight = teacher.different_confidence * pair
    same = _weighted(similarity, same_weight)
    different = _weighted(similarity, different_weight)
    relation_degree = (
        teacher.same_confidence + teacher.different_confidence
    ).amax(dim=-1)
    root_weight = support * relation_degree
    root_mass = (track * root_weight[..., None]).sum(dim=1)
    share = root_mass / root_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    effective = (-(share * share.clamp_min(1e-6).log()).sum(dim=-1)).exp()
    batch_weight = torch.ones_like(effective)
    return track, {
        "same_relation_root_similarity": same,
        "different_relation_root_similarity": different,
        "effective_roots": _weighted(effective, batch_weight),
        "maximum_root_share": _weighted(share.amax(dim=-1), batch_weight),
        "supported_roots": _weighted(
            (root_mass > 0.05).float().sum(dim=-1), batch_weight
        ),
    }


def _track_correspondence(conditional, teacher):
    visible = visible_mask(teacher)
    weight = (
        visible[:, 1:]
        & visible[:, :-1]
    ).float() * teacher.object_confidence[:, None]
    correct = _weighted(_cosine(conditional[:, :-1], conditional[:, 1:]), weight)
    shuffled = _weighted(
        _cosine(conditional[:, :-1], conditional[:, 1:].roll(1, dims=2)),
        weight,
    )
    return correct, shuffled


def _reappearance(values, teacher):
    visibility = visible_mask(teacher)
    batch, frames, points = visibility.shape
    last = torch.zeros(batch, points, values.shape[-1], device=values.device)
    seen = torch.zeros(batch, points, device=values.device, dtype=torch.bool)
    gap = torch.zeros_like(seen)
    correct = values.new_zeros((), dtype=torch.float32)
    shuffled = values.new_zeros((), dtype=torch.float32)
    events = values.new_zeros((), dtype=torch.float32)
    for frame in range(frames):
        current = visibility[:, frame]
        reappeared = current & seen & gap & (teacher.object_confidence > 0.0)
        after = values[:, frame]
        correct += (_cosine(last, after) * reappeared).sum()
        shuffled += (_cosine(last, after.roll(1, dims=1)) * reappeared).sum()
        events += reappeared.float().sum()
        last = torch.where(current[..., None], after, last)
        gap = torch.where(current, torch.zeros_like(gap), gap | seen)
        seen = seen | current
    count = float(events)
    if count == 0.0:
        return (0.0, 0.0), (0.0, 0.0), 0.0
    return (float(correct / events), count), (float(shuffled / events), count), count


def _permute_track_dimension(value: torch.Tensor, midpoint: int) -> torch.Tensor:
    result = value.clone()
    result[:, midpoint:] = result[:, midpoint:].roll(1, dims=2)
    return result


def track_shuffle_objective_delta(model, output, evidence) -> float:
    prediction = output["prediction"]
    midpoint = prediction.assignment.shape[1] // 2
    shuffled = replace(
        prediction,
        assignment=_permute_track_dimension(prediction.assignment, midpoint),
        decoder_assignment=_permute_track_dimension(
            prediction.decoder_assignment, midpoint
        ),
        identity=_permute_track_dimension(prediction.identity, midpoint),
        motion=_permute_track_dimension(prediction.motion, midpoint),
        center=_permute_track_dimension(prediction.center, midpoint),
        visibility=_permute_track_dimension(prediction.visibility, midpoint),
        presence=_permute_track_dimension(prediction.presence, midpoint),
    )
    semantic = _permute_track_dimension(output["semantic_identity"], midpoint)
    corrupted = verified_relation_object_state_terms(
        shuffled, semantic, output["teacher"], evidence, model.config
    )
    reference = output["parts"]["target_target_total"]
    return float(corrupted["target_total"] - reference)


def teacher_deletion_locality(model, features, output, amp_context):
    state, teacher = output["state"], output["teacher"]
    prediction = output["prediction"]
    conditional, _ = conditional_object_assignment(
        prediction.assignment, model.config.object_slots
    )
    canonical = visible_track_mean(conditional, teacher.visibility)
    visible = visible_mask(teacher)
    relation_degree = (
        teacher.same_confidence + teacher.different_confidence
    ).amax(dim=-1)
    score = teacher.object_confidence * relation_degree
    inside_sum = outside_sum = events = 0.0
    for batch_index in range(len(state["identity"])):
        track = int(score[batch_index].argmax())
        if float(score[batch_index, track]) == 0.0:
            continue
        visible_frames = torch.where(visible[batch_index, :, track])[0]
        if len(visible_frames) == 0:
            continue
        frame = int(visible_frames[len(visible_frames) // 2])
        slot = int(canonical[batch_index, track].argmax())
        component = teacher.same_confidence[batch_index, track] > 0.0
        component[track] = True
        component &= visible[batch_index, frame]
        track_positions = output["teacher_evidence"].coordinates[
            batch_index, frame, component
        ]
        if len(track_positions) == 0:
            continue
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
            deleted, _ = model.decoder(
                frame_state, coordinates, valid, object_valid
            )
        change = 1.0 - _cosine(reference, deleted)
        patch_positions = features.coordinates[batch_index, frame]
        distance = (
            patch_positions[:, None] - track_positions[None]
        ).norm(dim=-1).amin(dim=1)
        inside = (distance <= 0.18) & valid[0]
        outside = (distance >= 0.30) & valid[0]
        if not bool(inside.any()) or not bool(outside.any()):
            continue
        inside_sum += float(change[0, inside].mean())
        outside_sum += float(change[0, outside].mean())
        events += 1.0
    return inside_sum, outside_sum, events


def collect_v56_diagnostics(
    model,
    features,
    evidence,
    output,
    amp_context,
    groups,
    motion_active_threshold,
):
    output = dict(output)
    output["teacher_evidence"] = evidence
    teacher = output["teacher"]
    prediction = output["prediction"]
    conditional, object_probability = conditional_object_assignment(
        prediction.assignment, model.config.object_slots
    )
    track, relation = _relation_metrics(conditional, teacher)
    correspondence, correspondence_shuffled = _track_correspondence(
        conditional, teacher
    )
    assignment_reappearance = _reappearance(conditional, teacher)
    identity_reappearance = _reappearance(
        F.normalize(prediction.identity.float(), dim=-1, eps=1e-6), teacher
    )
    deletion_inside, deletion_outside, deletion_events = teacher_deletion_locality(
        model, features, output, amp_context
    )
    weighted = {
        **relation,
        "track_correspondence_correct": correspondence,
        "track_correspondence_shuffled": correspondence_shuffled,
        "assignment_reappearance_correct": assignment_reappearance[0],
        "assignment_reappearance_shuffled": assignment_reappearance[1],
        "identity_reappearance_correct": identity_reappearance[0],
        "identity_reappearance_shuffled": identity_reappearance[1],
        "object_routing_probability": _weighted(
            object_probability,
            teacher.visibility.float() * teacher.object_confidence[:, None],
        ),
    }
    state = output["state"]
    mapped_dynamic = torch.einsum(
        "btpk,btkd->btpd", conditional, state["dynamic"].float()
    )
    group = groups[:, None, None].expand_as(teacher.visibility)
    horizons = len(model.config.dynamic_horizons)
    probes = {
        "raw_track_motion_target": teacher.motion.reshape(-1, 2).detach().cpu(),
        "raw_track_motion_weight": (
            teacher.motion_valid.float() * teacher.object_confidence[:, None, :, None]
        ).reshape(-1).detach().cpu(),
        "raw_track_motion_group": group[..., None]
        .expand(-1, -1, -1, horizons)
        .reshape(-1)
        .detach()
        .cpu(),
        "dynamic_visibility_feature": mapped_dynamic.reshape(
            -1, mapped_dynamic.shape[-1]
        ).detach().cpu(),
        "dynamic_visibility_target": teacher.visibility.reshape(-1, 1)
        .float()
        .detach()
        .cpu(),
        "dynamic_visibility_weight": (
            teacher.lifecycle_known.float() * teacher.object_confidence[:, None]
        ).reshape(-1).detach().cpu(),
        "dynamic_visibility_group": group.reshape(-1).detach().cpu(),
    }
    probes.update(
        aligned_state_probe_tensors(
            mapped_dynamic,
            output,
            groups,
            motion_active_threshold,
        )
    )
    totals = {
        "assignment_reappearance_events": assignment_reappearance[2],
        "identity_reappearance_events": identity_reappearance[2],
        "deletion_inside_sum": deletion_inside,
        "deletion_outside_sum": deletion_outside,
        "deletion_events": deletion_events,
        "track_shuffle_objective_delta_sum": track_shuffle_objective_delta(
            model, output, evidence
        ) * len(groups),
        "track_shuffle_objective_delta_count": float(len(groups)),
    }
    return V56DiagnosticBatch(weighted, totals, probes)


def finalize_v56_metrics(aggregate, ridge_relative_gain):
    metrics = aggregate.means()
    metrics.update(aggregate.totals)
    metrics["evaluated_items"] = float(aggregate.items)
    metrics["causal_prefix_max_difference"] = aggregate.causal_max
    for prefix in (
        "track_correspondence",
        "assignment_reappearance",
        "identity_reappearance",
    ):
        metrics[f"{prefix}_margin"] = (
            metrics.get(f"{prefix}_correct", 0.0)
            - metrics.get(f"{prefix}_shuffled", 0.0)
        )
    metrics["relation_root_margin"] = (
        metrics["same_relation_root_similarity"]
        - metrics["different_relation_root_similarity"]
    )
    metrics["track_shuffle_objective_delta"] = metrics[
        "track_shuffle_objective_delta_sum"
    ] / max(metrics["track_shuffle_objective_delta_count"], 1.0)
    metrics["deletion_inside_change"] = metrics["deletion_inside_sum"] / max(
        metrics["deletion_events"], 1.0
    )
    metrics["deletion_outside_change"] = metrics["deletion_outside_sum"] / max(
        metrics["deletion_events"], 1.0
    )
    metrics["deletion_locality_ratio"] = metrics["deletion_inside_change"] / max(
        metrics["deletion_outside_change"], 1e-8
    )
    probes = {name: torch.cat(values) for name, values in aggregate.probes.items()}
    metrics["raw_track_motion_probe_relative_gain"] = ridge_relative_gain(
        probes["component_motion_feature"], probes["raw_track_motion_target"],
        probes["raw_track_motion_weight"], probes["raw_track_motion_group"]
    )
    metrics["dynamic_visibility_probe_relative_gain"] = ridge_relative_gain(
        probes["dynamic_visibility_feature"], probes["dynamic_visibility_target"],
        probes["dynamic_visibility_weight"], probes["dynamic_visibility_group"]
    )
    metrics.update(finalize_aligned_state_probes(probes, ridge_relative_gain))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("v56 evaluation produced non-finite metrics")
    return metrics
