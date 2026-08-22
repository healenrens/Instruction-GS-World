"""Aggregation and admission checks for v57 real-video query coverage."""

from __future__ import annotations

from collections import defaultdict

import torch

from .point_track_teacher import PointTrackEvidence
from .query_object_teacher_v57 import QueryObjectTeacher, query_teacher_contract_metrics


def future_track_shuffle(evidence: PointTrackEvidence, observed_frames: int):
    """Break future track identity while preserving every observed tensor."""

    coordinates = evidence.coordinates.clone()
    visibility = evidence.visibility.clone()
    sampled = evidence.sampled_features.clone()
    coordinates[:, observed_frames:] = coordinates[:, observed_frames:].roll(1, dims=2)
    visibility[:, observed_frames:] = visibility[:, observed_frames:].roll(1, dims=2)
    sampled[:, observed_frames:] = sampled[:, observed_frames:].roll(1, dims=2)
    flow = coordinates[:, 1:] - coordinates[:, :-1]
    pair_visible = visibility[:, 1:] & visibility[:, :-1]
    weight = pair_visible.float()
    global_flow = (flow * weight[..., None]).sum(dim=2, keepdim=True)
    global_flow = global_flow / weight.sum(dim=2, keepdim=True).clamp_min(1.0)[..., None]
    residual = flow - global_flow
    speed = residual.norm(dim=-1) * weight
    scale = torch.quantile(speed, 0.75, dim=2, keepdim=True).clamp_min(0.01)
    salience = (speed / scale).clamp(0.0, 1.0) * weight
    return PointTrackEvidence(
        coordinates=coordinates,
        visibility=visibility,
        residual_flow=residual,
        motion_salience=salience,
        query_times=evidence.query_times,
        sampled_features=sampled,
    )


def teacher_target_difference(first, second) -> float:
    differences = (
        (first.same_target.float() - second.same_target.float()).abs().mean(),
        (first.different_target.float() - second.different_target.float()).abs().mean(),
    )
    return float(torch.stack(differences).sum().detach())


def select_query_teacher(teacher: QueryObjectTeacher, index: int) -> QueryObjectTeacher:
    values = {}
    for name in teacher.__dataclass_fields__:
        value = getattr(teacher, name)
        values[name] = value[index : index + 1]
    return QueryObjectTeacher(**values)


def coverage_row(source: str, history: int, teacher, shuffled_teacher) -> dict:
    metrics = query_teacher_contract_metrics(teacher)
    return {
        "source": source,
        "history_frames": int(history),
        **{name: float(value.detach()) for name, value in metrics.items()},
        "teacher_future_swap_difference": teacher_target_difference(
            teacher, shuffled_teacher
        ),
        "samples": int(len(teacher.query_valid)),
    }


def _mean_rows(rows: list[dict]) -> dict:
    numeric = [name for name in rows[0] if name not in {"source", "history_frames"}]
    return {name: sum(float(row[name]) for row in rows) / len(rows) for name in numeric}


def summarize_query_coverage(
    rows: list[dict], config, expected_sources, expected_histories
) -> dict:
    if not rows:
        raise ValueError("v57 coverage audit has no rows")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], int(row["history_frames"]))].append(row)
    conditions = {
        f"{source}/h{history}": _mean_rows(selected)
        for (source, history), selected in sorted(grouped.items())
    }
    expected = {
        f"{source}/h{history}"
        for source in expected_sources
        for history in expected_histories
    }
    missing = sorted(expected.difference(conditions))
    unexpected = sorted(set(conditions).difference(expected))
    aggregate = _mean_rows(rows)
    checks = {
        "all_expected_conditions_are_present": not missing and not unexpected,
        "prompt_and_heldout_are_disjoint": all(
            value["prompt_heldout_overlap"] == 0.0 for value in conditions.values()
        ),
        "every_condition_has_query_coverage": all(
            value["query_valid_fraction"]
            >= config.minimum_condition_query_fraction
            for value in conditions.values()
        ),
        "every_condition_has_trainable_examples": all(
            value["trainable_sample_fraction"]
            >= config.minimum_condition_trainable_fraction
            for value in conditions.values()
        ),
        "aggregate_trainable_coverage_is_sufficient": (
            aggregate["trainable_sample_fraction"]
            >= config.minimum_aggregate_trainable_fraction
        ),
        "teacher_targets_use_future_tracks": (
            aggregate["teacher_future_swap_difference"] > 1e-6
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "aggregate": aggregate,
        "conditions": conditions,
        "expected_condition_count": len(expected),
        "condition_count": len(conditions),
        "missing_conditions": missing,
        "unexpected_conditions": unexpected,
        "row_count": len(rows),
    }
