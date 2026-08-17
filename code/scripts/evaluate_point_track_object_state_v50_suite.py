"""Comprehensive held-video evaluation for the v50 Object State checkpoint."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (
    FrozenDinoVideoRuntime,
)  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_dataset import (
    PointTrackObjectVideoDataset,
)  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_teacher import (
    FrozenPointTrackerRuntime,
)  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_world_model import (
    PointTrackObjectWorldModel,
)  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v50_checkpointing import (
    validate_checkpoint_header,
)  # noqa: E402
from igsw.adaptive_gaussian_wm.v50_config import (
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    PointTrackObjectStateConfig,
)  # noqa: E402
from igsw.adaptive_gaussian_wm.v50_state_diagnostics import (  # noqa: E402
    EvaluationAggregate,
    causal_prefix_difference,
    collect_batch_diagnostics,
    order_sensitivity,
    ridge_relative_gain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--evaluator_revision", default="")
    parser.add_argument("--expected_step", type=int, default=22000)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--splits", default="heldseed,heldtask")
    parser.add_argument("--chunk_lengths", default="8,16,24,32")
    parser.add_argument("--temporal_stride", type=int, default=1)
    parser.add_argument("--items", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--tracker_sequence_batch", type=int, default=1)
    parser.add_argument("--causal_items", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--wandb_mode", choices=("disabled", "online", "offline"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument(
        "--wandb_name", default="point_track_object_state_v50_step22000_comprehensive"
    )
    parser.add_argument("--wandb_group", default="point-track-object-state-v50-eval")
    parser.add_argument("--wandb_dir", default="")
    parser.add_argument("--require_promotion", action="store_true")
    return parser.parse_args()


def _relative_gain(correct: float, shuffled: float) -> float:
    return (correct - shuffled) / max(1.0 - shuffled, 1e-6)


def _finalize(aggregate: EvaluationAggregate) -> dict[str, float]:
    metrics = aggregate.means()
    metrics.update(aggregate.totals)
    for name in (
        "moving_student_object_fraction",
        "moving_teacher_object_fraction",
        "static_student_object_fraction",
        "static_teacher_object_fraction",
    ):
        metrics.setdefault(name, -1.0)
    metrics["evaluated_items"] = float(aggregate.items)
    metrics["causal_prefix_max_difference"] = aggregate.causal_max
    if aggregate.order_count:
        for name, value in aggregate.order_sum.items():
            metrics[name] = value / aggregate.order_count
    for prefix in ("track_assignment", "track_reappearance", "component_reappearance"):
        correct = metrics.get(f"{prefix}_correct_cosine", 0.0)
        shuffled = metrics.get(f"{prefix}_shuffled_cosine", 0.0)
        metrics[f"{prefix}_gain_over_shuffled"] = _relative_gain(correct, shuffled)
    deletion_items = metrics.get("deletion_valid_items", 0.0)
    metrics["deletion_inside_change"] = metrics.get("deletion_inside_sum", 0.0) / max(
        deletion_items, 1.0
    )
    metrics["deletion_outside_change"] = metrics.get("deletion_outside_sum", 0.0) / max(
        deletion_items, 1.0
    )
    metrics["deletion_locality_ratio"] = metrics["deletion_inside_change"] / max(
        metrics["deletion_outside_change"], 1e-8
    )
    lifecycle_total = metrics.get("lifecycle_total", 0.0)
    lifecycle_known = (
        metrics.get("lifecycle_visible", 0.0)
        + metrics.get("lifecycle_occluded", 0.0)
        + metrics.get("lifecycle_absent", 0.0)
    )
    metrics["lifecycle_unknown_fraction"] = metrics.get("lifecycle_unknown", 0.0) / max(
        lifecycle_total, 1.0
    )
    metrics["lifecycle_occluded_given_known"] = metrics.get(
        "lifecycle_occluded", 0.0
    ) / max(lifecycle_known, 1.0)
    metrics["lifecycle_absent_given_known"] = metrics.get(
        "lifecycle_absent", 0.0
    ) / max(lifecycle_known, 1.0)
    probes = {name: torch.cat(values) for name, values in aggregate.probes.items()}
    identity, dynamic = probes["identity"], probes["dynamic"]
    full = torch.cat((identity, dynamic), dim=-1)
    for representation, feature in (
        ("identity", identity),
        ("dynamic", dynamic),
        ("full", full),
    ):
        metrics[f"motion_probe_{representation}_relative_gain"] = ridge_relative_gain(
            feature, probes["motion"], probes["motion_weight"]
        )
        metrics[f"visibility_probe_{representation}_relative_gain"] = (
            ridge_relative_gain(
                feature, probes["visibility"], probes["visibility_weight"]
            )
        )
    metrics["dynamic_motion_probe_advantage"] = (
        metrics["motion_probe_dynamic_relative_gain"]
        - metrics["motion_probe_identity_relative_gain"]
    )
    metrics["dynamic_to_identity_temporal_change"] = metrics.get(
        "dynamic_temporal_change", 0.0
    ) / max(metrics.get("identity_temporal_drift", 0.0), 1e-6)
    return metrics


def _checks(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "all_metrics_finite": all(math.isfinite(value) for value in metrics.values()),
        "causal_rgb_student": metrics["causal_prefix_max_difference"] < 1e-6,
        "teacher_owner_gain": metrics["teacher_owner_gain_over_shuffled"] >= 0.05,
        "visible_identity_gain": metrics["teacher_identity_gain_over_shuffled"] >= 0.05,
        "external_track_correspondence": metrics["track_assignment_gain_over_shuffled"]
        >= 0.05,
        "track_reappearance_available": metrics["track_reappearance_events"] >= 8,
        "track_reappearance_identity": metrics["track_reappearance_gain_over_shuffled"]
        >= 0.05,
        "component_reappearance_available": metrics["component_reappearance_events"]
        >= 8,
        "component_reappearance_identity": metrics[
            "component_reappearance_gain_over_shuffled"
        ]
        > 0.0,
        "deletion_examples_available": metrics["deletion_valid_items"] >= 8,
        "deletion_is_local": metrics["deletion_locality_ratio"] >= 1.25,
        "dynamic_state_decodes_motion": metrics["motion_probe_dynamic_relative_gain"]
        >= 0.05,
        "dynamic_more_motion_specific_than_identity": metrics[
            "dynamic_motion_probe_advantage"
        ]
        > 0.0,
        "state_decodes_visibility": metrics["visibility_probe_full_relative_gain"]
        >= 0.05,
        "moving_tracks_use_object_path": metrics["moving_student_object_fraction"]
        >= 0.50,
        "student_teacher_object_coverage_agree": abs(
            metrics["student_object_track_fraction"]
            - metrics["teacher_object_track_fraction"]
        )
        <= 0.15,
    }


def evaluate_condition(args, model, loader, dino, tracker, device, amp_context):
    aggregate = EvaluationAggregate()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            features = dino(batch)
            evidence = tracker(batch, features.patches, features.grid_hw)
            with amp_context():
                output = model(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                    evidence,
                    features.grid_hw,
                )
            batch_items = len(features.patches)
            aggregate.items += batch_items
            for name, value in output["parts"].items():
                aggregate.add_mean(name, float(value), batch_items)
            diagnostics = collect_batch_diagnostics(
                model, features, evidence, output, amp_context
            )
            for name, (value, weight) in diagnostics.weighted.items():
                aggregate.add_mean(name, value, weight)
            for name, value in diagnostics.totals.items():
                aggregate.add_total(name, value)
            for name, value in diagnostics.probes.items():
                aggregate.probes.setdefault(name, []).append(value)
            if aggregate.order_count < args.causal_items:
                aggregate.causal_max = max(
                    aggregate.causal_max,
                    causal_prefix_difference(
                        model, features, batch["frame_times"], output["state"]
                    ),
                )
                order = order_sensitivity(
                    model, features, batch["frame_times"], output["state"]
                )
                for name, value in order.items():
                    aggregate.order_sum[name] += value * batch_items
                aggregate.order_count += batch_items
    metrics = _finalize(aggregate)
    return metrics, _checks(metrics)


def _aggregate_decision(results, splits, lengths):
    conditions = [
        results[f"{split}/H{length}"] for split in splits for length in lengths
    ]
    all_checks = [condition["checks"] for condition in conditions]
    track_events = sum(
        condition["metrics"]["track_reappearance_events"] for condition in conditions
    )
    component_events = sum(
        condition["metrics"]["component_reappearance_events"]
        for condition in conditions
    )
    track_correct = sum(
        condition["metrics"]["track_reappearance_correct_cosine"]
        * condition["metrics"]["track_reappearance_events"]
        for condition in conditions
    ) / max(track_events, 1.0)
    track_shuffled = sum(
        condition["metrics"]["track_reappearance_shuffled_cosine"]
        * condition["metrics"]["track_reappearance_events"]
        for condition in conditions
    ) / max(track_events, 1.0)
    h8 = [
        results[f"{split}/H{length}"]["metrics"]
        for split in splits
        for length in lengths
        if length == min(lengths)
    ]
    hmax = [
        results[f"{split}/H{length}"]["metrics"]
        for split in splits
        for length in lengths
        if length == max(lengths)
    ]
    short_gain = sum(item["track_assignment_gain_over_shuffled"] for item in h8) / len(
        h8
    )
    long_gain = sum(item["track_assignment_gain_over_shuffled"] for item in hmax) / len(
        hmax
    )
    aggregate = {
        "condition_count": float(len(conditions)),
        "track_reappearance_events": track_events,
        "component_reappearance_events": component_events,
        "track_reappearance_gain_over_shuffled": _relative_gain(
            track_correct, track_shuffled
        ),
        "minimum_teacher_owner_gain": min(
            condition["metrics"]["teacher_owner_gain_over_shuffled"]
            for condition in conditions
        ),
        "minimum_visible_identity_gain": min(
            condition["metrics"]["teacher_identity_gain_over_shuffled"]
            for condition in conditions
        ),
        "minimum_track_correspondence_gain": min(
            condition["metrics"]["track_assignment_gain_over_shuffled"]
            for condition in conditions
        ),
        "minimum_deletion_locality_ratio": min(
            condition["metrics"]["deletion_locality_ratio"] for condition in conditions
        ),
        "minimum_dynamic_motion_probe": min(
            condition["metrics"]["motion_probe_dynamic_relative_gain"]
            for condition in conditions
        ),
        "minimum_visibility_probe": min(
            condition["metrics"]["visibility_probe_full_relative_gain"]
            for condition in conditions
        ),
        "long_minus_short_track_gain": long_gain - short_gain,
    }
    checks = {
        "matrix_complete": len(conditions) == len(splits) * len(lengths),
        "all_finite": all(check["all_metrics_finite"] for check in all_checks),
        "causal_all_conditions": all(
            check["causal_rgb_student"] for check in all_checks
        ),
        "teacher_consistency_all_conditions": all(
            check["teacher_owner_gain"] and check["visible_identity_gain"]
            for check in all_checks
        ),
        "track_correspondence_all_conditions": all(
            check["external_track_correspondence"] for check in all_checks
        ),
        "track_reappearance_supported": track_events >= 64
        and aggregate["track_reappearance_gain_over_shuffled"] >= 0.05,
        "component_reappearance_supported": component_events >= 64,
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
        "long_history_not_worse": aggregate["long_minus_short_track_gain"] >= -0.02,
    }
    return aggregate, checks, all(checks.values())


def _init_wandb(args, checkpoint_step):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir or not args.wandb_project:
        raise ValueError("v50 suite W&B requires directory and project")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        job_type="v50-object-state-comprehensive-held-evaluation",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={**vars(args), "checkpoint_step": checkpoint_step},
    )
    run.define_metric("evaluation/condition_index")
    run.define_metric("metric/*", step_metric="evaluation/condition_index")
    run.define_metric("check/*", step_metric="evaluation/condition_index")
    return run


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v50 comprehensive evaluation requires CUDA")
    device = torch.device("cuda:0")
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_checkpoint_header(checkpoint)
    if checkpoint.get("stage") != "object_state":
        raise ValueError("v50 suite requires an Object State checkpoint")
    if checkpoint.get("git_commit") != args.source_revision:
        raise ValueError("v50 suite source revision differs from checkpoint")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError(
            f"v50 suite requires step {args.expected_step}, got {checkpoint.get('global_step')}"
        )
    config = PointTrackObjectStateConfig()
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("v50 suite config differs from checkpoint")
    model = PointTrackObjectWorldModel(config, "object_state").to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, args.tracker_sequence_batch
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    lengths = [int(value) for value in args.chunk_lengths.split(",")]
    run = _init_wandb(args, int(checkpoint["global_step"]))
    results = {}
    condition_index = 0
    for split in splits:
        for length in lengths:
            dataset = PointTrackObjectVideoDataset(
                args.data,
                split,
                str(length),
                str(args.temporal_stride),
                args.items,
                args.seed,
            )
            loader = DataLoader(
                dataset, batch_size=args.batch, shuffle=False, num_workers=0
            )
            metrics, checks = evaluate_condition(
                args, model, loader, dino, tracker, device, amp_context
            )
            key = f"{split}/H{length}"
            results[key] = {"metrics": metrics, "checks": checks}
            print(
                json.dumps(
                    {"condition": key, "metrics": metrics, "checks": checks},
                    sort_keys=True,
                ),
                flush=True,
            )
            if run is not None:
                payload = {
                    "evaluation/condition_index": condition_index,
                    "evaluation/history_length": length,
                    "evaluation/split_code": splits.index(split),
                    **{f"metric/{name}": value for name, value in metrics.items()},
                    **{f"check/{name}": int(value) for name, value in checks.items()},
                }
                run.log(payload, step=condition_index)
                for name, value in metrics.items():
                    run.summary[f"condition/{key}/{name}"] = value
                for name, value in checks.items():
                    run.summary[f"condition/{key}/check/{name}"] = int(value)
            condition_index += 1
    aggregate, checks, promotion = _aggregate_decision(results, splits, lengths)
    report = {
        "status": "completed",
        "promotion_ready": promotion,
        "evaluation_scope": "held_video_teacher_anchored_diagnostics_not_semantic_annotation",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "training_git_commit": args.source_revision,
        "evaluator_git_commit": args.evaluator_revision,
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data),
        "splits": splits,
        "chunk_lengths": lengths,
        "temporal_stride": args.temporal_stride,
        "conditions": results,
        "aggregate": aggregate,
        "checks": checks,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if run is not None:
        run.summary.update(
            {
                "evaluation/status": report["status"],
                "evaluation/promotion_ready": int(promotion),
                "evaluation/checkpoint_step": int(checkpoint["global_step"]),
                "evaluation/report": output,
                **{f"aggregate/{name}": value for name, value in aggregate.items()},
                **{f"gate/{name}": int(value) for name, value in checks.items()},
            }
        )
        run.finish()
    print(
        json.dumps(
            {"report": output, "promotion_ready": promotion, "checks": checks},
            sort_keys=True,
        ),
        flush=True,
    )
    if args.require_promotion and not promotion:
        raise RuntimeError("v50 Object State promotion gates did not pass")


if __name__ == "__main__":
    main()
