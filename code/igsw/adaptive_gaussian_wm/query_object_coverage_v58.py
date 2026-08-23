"""Coverage aggregation for persistent query-state supervision."""

from __future__ import annotations

from collections import defaultdict

from .query_object_coverage_v57 import future_track_shuffle as future_track_shuffle
from .query_object_teacher_v57 import QueryObjectTeacher
from .query_object_teacher_v58 import (
    QueryPersistentTeacher,
    persistent_teacher_contract_metrics,
)

__all__ = (
    "coverage_row",
    "future_track_shuffle",
    "select_query_teacher",
    "summarize_query_coverage",
    "teacher_target_difference",
)


def _select_binding(teacher: QueryObjectTeacher, index: int) -> QueryObjectTeacher:
    return QueryObjectTeacher(
        **{
            name: getattr(teacher, name)[index : index + 1]
            for name in teacher.__dataclass_fields__
        }
    )


def select_query_teacher(
    teacher: QueryPersistentTeacher, index: int
) -> QueryPersistentTeacher:
    return QueryPersistentTeacher(
        binding=_select_binding(teacher.binding, index),
        **{
            name: getattr(teacher, name)[index : index + 1]
            for name in teacher.__dataclass_fields__
            if name != "binding"
        },
    )


def teacher_target_difference(first, second) -> float:
    differences = (
        (first.same_target.float() - second.same_target.float()).abs().mean(),
        (first.different_target.float() - second.different_target.float()).abs().mean(),
    )
    return float(sum(differences).detach())


def coverage_row(source: str, history: int, teacher, shuffled_teacher) -> dict:
    metrics = persistent_teacher_contract_metrics(teacher)
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


def summarize_query_coverage(rows, config, expected_sources, expected_histories):
    if not rows:
        raise ValueError("v58 coverage audit has no rows")
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
            value["query_valid_fraction"] >= config.minimum_condition_query_fraction
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
        "aggregate_contains_occlusion_candidates": (
            aggregate["lifecycle_occluded_candidate_fraction"]
            >= config.minimum_aggregate_occluded_fraction
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
