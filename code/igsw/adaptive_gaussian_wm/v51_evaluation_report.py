"""Promotion metrics for component-balanced v51 Object State evaluation."""

from __future__ import annotations

import math

import torch

from .v51_state_diagnostics import EvaluationAggregate, ridge_relative_gain


def relative_gain(correct: float, shuffled: float) -> float:
    return (correct - shuffled) / max(1.0 - shuffled, 1e-6)


def finalize_metrics(aggregate: EvaluationAggregate) -> dict[str, float]:
    metrics = aggregate.means()
    metrics.update(aggregate.totals)
    defaults = {
        "moving_student_object_fraction": -1.0,
        "moving_teacher_object_fraction": -1.0,
        "static_student_object_fraction": -1.0,
        "static_teacher_object_fraction": -1.0,
        "teacher_component_coverage": 0.0,
        "teacher_component_covered_fraction": 0.0,
        "multi_component_slot_max_fraction": 1.0,
    }
    for name, value in defaults.items():
        metrics.setdefault(name, value)
    metrics["evaluated_items"] = float(aggregate.items)
    metrics["causal_prefix_max_difference"] = aggregate.causal_max
    if aggregate.order_count:
        for name, value in aggregate.order_sum.items():
            metrics[name] = value / aggregate.order_count
    for prefix in (
        "track_assignment",
        "track_reappearance",
        "component_reappearance",
    ):
        correct = metrics.get(f"{prefix}_correct_cosine", 0.0)
        shuffled = metrics.get(f"{prefix}_shuffled_cosine", 0.0)
        metrics[f"{prefix}_absolute_margin"] = correct - shuffled
        metrics[f"{prefix}_gain_over_shuffled"] = relative_gain(correct, shuffled)
    deletion_items = metrics.get("deletion_valid_items", 0.0)
    metrics["deletion_inside_change"] = metrics.get(
        "deletion_inside_sum", 0.0
    ) / max(deletion_items, 1.0)
    metrics["deletion_outside_change"] = metrics.get(
        "deletion_outside_sum", 0.0
    ) / max(deletion_items, 1.0)
    metrics["deletion_locality_ratio"] = metrics["deletion_inside_change"] / max(
        metrics["deletion_outside_change"], 1e-8
    )
    lifecycle_total = metrics.get("lifecycle_total", 0.0)
    lifecycle_known = (
        metrics.get("lifecycle_visible", 0.0)
        + metrics.get("lifecycle_occluded", 0.0)
        + metrics.get("lifecycle_absent", 0.0)
    )
    metrics["lifecycle_unknown_fraction"] = metrics.get(
        "lifecycle_unknown", 0.0
    ) / max(lifecycle_total, 1.0)
    metrics["lifecycle_occluded_given_known"] = metrics.get(
        "lifecycle_occluded", 0.0
    ) / max(lifecycle_known, 1.0)
    metrics["lifecycle_absent_given_known"] = metrics.get(
        "lifecycle_absent", 0.0
    ) / max(lifecycle_known, 1.0)
    probes = {name: torch.cat(values) for name, values in aggregate.probes.items()}
    motion_identity = probes["motion_identity"]
    motion_dynamic = probes["motion_dynamic"]
    visibility_identity = probes["visibility_identity"]
    visibility_dynamic = probes["visibility_dynamic"]
    for representation, feature in (
        ("identity", motion_identity),
        ("dynamic", motion_dynamic),
        ("full", torch.cat((motion_identity, motion_dynamic), dim=-1)),
    ):
        metrics[f"motion_probe_{representation}_relative_gain"] = ridge_relative_gain(
            feature,
            probes["motion"],
            probes["motion_weight"],
            probes["motion_group"],
        )
    for representation, feature in (
        ("identity", visibility_identity),
        ("dynamic", visibility_dynamic),
        ("full", torch.cat((visibility_identity, visibility_dynamic), dim=-1)),
    ):
        metrics[f"visibility_probe_{representation}_relative_gain"] = ridge_relative_gain(
            feature,
            probes["visibility"],
            probes["visibility_weight"],
            probes["visibility_group"],
        )
    metrics["dynamic_motion_probe_advantage"] = (
        metrics["motion_probe_dynamic_relative_gain"]
        - metrics["motion_probe_identity_relative_gain"]
    )
    metrics["dynamic_to_identity_temporal_change"] = metrics.get(
        "dynamic_temporal_change", 0.0
    ) / max(metrics.get("identity_temporal_drift", 0.0), 1e-6)
    metrics["same_endpoint_order_dynamic_advantage"] = metrics.get(
        "dynamic_same_endpoint_order_change", 0.0
    ) - metrics.get("identity_same_endpoint_order_change", 0.0)
    return metrics


def condition_checks(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "all_metrics_finite": all(math.isfinite(value) for value in metrics.values()),
        "causal_rgb_student": metrics["causal_prefix_max_difference"] < 1e-6,
        "teacher_owner_gain": metrics["teacher_owner_gain_over_shuffled"] >= 0.05,
        "visible_identity_gain": metrics["teacher_identity_gain_over_shuffled"] >= 0.05,
        "track_correspondence_absolute": metrics[
            "track_assignment_absolute_margin"
        ] >= 0.03,
        "track_reappearance_available": metrics["track_reappearance_events"] >= 8,
        "track_reappearance_identity": metrics[
            "track_reappearance_absolute_margin"
        ] >= 0.02,
        "component_reappearance_available": metrics[
            "component_reappearance_events"
        ] >= 8,
        "component_reappearance_identity": metrics[
            "component_reappearance_absolute_margin"
        ] > 0.0,
        "component_coverage": metrics["teacher_component_coverage"] >= 0.50,
        "component_set_not_concentrated": metrics[
            "multi_component_slot_max_fraction"
        ] <= 0.75,
        "static_tracks_have_teacher_objects": metrics[
            "static_teacher_object_fraction"
        ] >= 0.10,
        "static_tracks_reach_student_objects": metrics[
            "static_student_object_fraction"
        ] >= 0.35,
        "lifecycle_has_occlusion": metrics[
            "lifecycle_occluded_given_known"
        ] > 0.0,
        "lifecycle_has_absence": metrics["lifecycle_absent_given_known"] > 0.0,
        "deletion_examples_available": metrics["deletion_valid_items"] >= 8,
        "deletion_is_local": metrics["deletion_locality_ratio"] >= 1.25,
        "dynamic_state_decodes_motion": metrics[
            "motion_probe_dynamic_relative_gain"
        ] >= 0.05,
        "dynamic_more_motion_specific_than_identity": metrics[
            "dynamic_motion_probe_advantage"
        ] > 0.0,
        "state_decodes_visibility": metrics[
            "visibility_probe_full_relative_gain"
        ] >= 0.05,
        "same_endpoint_order_sensitive": metrics[
            "dynamic_same_endpoint_order_change"
        ] >= 0.01
        and metrics["same_endpoint_order_dynamic_advantage"] > 0.0,
        "moving_tracks_use_object_path": metrics[
            "moving_student_object_fraction"
        ] >= 0.50,
    }


def aggregate_decision(results, splits, lengths):
    conditions = [
        results[f"{split}/H{length}"] for split in splits for length in lengths
    ]
    all_checks = [condition["checks"] for condition in conditions]
    track_events = sum(
        condition["metrics"]["track_reappearance_events"]
        for condition in conditions
    )
    component_events = sum(
        condition["metrics"]["component_reappearance_events"]
        for condition in conditions
    )
    track_margin = sum(
        condition["metrics"]["track_reappearance_absolute_margin"]
        * condition["metrics"]["track_reappearance_events"]
        for condition in conditions
    ) / max(track_events, 1.0)
    short = [
        results[f"{split}/H{min(lengths)}"]["metrics"] for split in splits
    ]
    long = [
        results[f"{split}/H{max(lengths)}"]["metrics"] for split in splits
    ]
    short_margin = sum(item["track_assignment_absolute_margin"] for item in short) / len(short)
    long_margin = sum(item["track_assignment_absolute_margin"] for item in long) / len(long)
    aggregate = {
        "condition_count": float(len(conditions)),
        "track_reappearance_events": track_events,
        "component_reappearance_events": component_events,
        "track_reappearance_absolute_margin": track_margin,
        "minimum_track_correspondence_absolute_margin": min(
            condition["metrics"]["track_assignment_absolute_margin"]
            for condition in conditions
        ),
        "minimum_component_coverage": min(
            condition["metrics"]["teacher_component_coverage"]
            for condition in conditions
        ),
        "maximum_multi_component_slot_fraction": max(
            condition["metrics"]["multi_component_slot_max_fraction"]
            for condition in conditions
        ),
        "minimum_static_teacher_object_fraction": min(
            condition["metrics"]["static_teacher_object_fraction"]
            for condition in conditions
        ),
        "minimum_lifecycle_absent_fraction": min(
            condition["metrics"]["lifecycle_absent_given_known"]
            for condition in conditions
        ),
        "minimum_dynamic_motion_probe": min(
            condition["metrics"]["motion_probe_dynamic_relative_gain"]
            for condition in conditions
        ),
        "minimum_visibility_probe": min(
            condition["metrics"]["visibility_probe_full_relative_gain"]
            for condition in conditions
        ),
        "long_minus_short_track_absolute_margin": long_margin - short_margin,
    }
    checks = {
        "matrix_complete": len(conditions) == len(splits) * len(lengths),
        "all_finite": all(check["all_metrics_finite"] for check in all_checks),
        "causal_all_conditions": all(
            check["causal_rgb_student"] for check in all_checks
        ),
        "static_and_moving_teacher_all_conditions": all(
            check["static_tracks_have_teacher_objects"]
            and check["static_tracks_reach_student_objects"]
            and check["moving_tracks_use_object_path"]
            for check in all_checks
        ),
        "teacher_identity_all_conditions": all(
            check["teacher_owner_gain"] and check["visible_identity_gain"]
            for check in all_checks
        ),
        "component_set_all_conditions": all(
            check["component_coverage"] and check["component_set_not_concentrated"]
            for check in all_checks
        ),
        "track_correspondence_all_conditions": all(
            check["track_correspondence_absolute"] for check in all_checks
        ),
        "track_reappearance_supported": track_events >= 64 and track_margin >= 0.02,
        "component_reappearance_supported": component_events >= 64,
        "lifecycle_supported_all_conditions": all(
            check["lifecycle_has_occlusion"] and check["lifecycle_has_absence"]
            for check in all_checks
        ),
        "deletion_locality_all_conditions": all(
            check["deletion_examples_available"] and check["deletion_is_local"]
            for check in all_checks
        ),
        "motion_disentanglement_all_conditions": all(
            check["dynamic_state_decodes_motion"]
            and check["dynamic_more_motion_specific_than_identity"]
            for check in all_checks
        ),
        "visibility_decodable_all_conditions": all(
            check["state_decodes_visibility"] for check in all_checks
        ),
        "pure_order_sensitive_all_conditions": all(
            check["same_endpoint_order_sensitive"] for check in all_checks
        ),
        "long_history_not_worse": aggregate[
            "long_minus_short_track_absolute_margin"
        ] >= -0.01,
    }
    return aggregate, checks, all(checks.values())
