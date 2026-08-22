#!/usr/bin/env python3
"""Evaluate v56 on deterministic source-balanced robot-video clips."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v51_state_diagnostics import (  # noqa: E402
    causal_prefix_difference,
    ridge_relative_gain,
)
from igsw.adaptive_gaussian_wm.v56_checkpointing import validate_header  # noqa: E402
from igsw.adaptive_gaussian_wm.v56_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    VerifiedRelationObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v56_evaluation_metrics import (  # noqa: E402
    V56EvaluationAggregate,
    collect_v56_diagnostics,
    finalize_v56_metrics,
)
from igsw.adaptive_gaussian_wm.verified_relation_object_state_v56 import (  # noqa: E402
    VerifiedRelationObjectStateModel,
)


class FixedEvaluationDataset(Dataset):
    def __init__(self, dataset, indices, chunk_length):
        self.dataset = dataset
        self.indices = tuple(indices)
        self.chunk_length = int(chunk_length)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[(self.indices[index], self.chunk_length)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluator_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--expected_step", type=int, default=20_000)
    parser.add_argument("--chunk_lengths", default="4,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--items_per_source", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--causal_items", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="online")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="verified_relation_object_state_v56_eval")
    parser.add_argument("--wandb_group", default="verified-relation-object-state-v56-eval")
    parser.add_argument("--wandb_dir", default="")
    parser.add_argument("--require_training_evidence_gate", action="store_true")
    return parser.parse_args()


def init_wandb(args, checkpoint):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir:
        raise ValueError("v56 evaluation W&B directory is missing")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        job_type="v56-source-balanced-training-evidence-evaluation",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            **vars(args),
            "training_revision": checkpoint["git_commit"],
            "checkpoint_step": checkpoint["global_step"],
        },
    )


def evaluate_condition(args, model, loader, dino, tracker, device, amp_context):
    aggregate = V56EvaluationAggregate()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            features = dino(batch)
            evidence = tracker(batch, features.patches, features.grid_hw)
            indices = torch.arange(len(features.patches), device=device)
            with amp_context():
                output = model(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                    evidence,
                    indices,
                    features.grid_hw,
                )
            batch_items = len(features.patches)
            aggregate.items += batch_items
            for name, value in output["parts"].items():
                if torch.is_tensor(value) and value.numel() == 1:
                    aggregate.add_mean(name, float(value), batch_items)
            diagnostics = collect_v56_diagnostics(
                model,
                features,
                evidence,
                output,
                amp_context,
                batch["requested_sequence_index"],
            )
            aggregate.add_batch(diagnostics)
            aggregate.add_mean(
                "decode_replacement_fraction",
                float(batch["decode_replaced"].float().mean()),
                batch_items,
            )
            if aggregate.items <= args.causal_items:
                with amp_context():
                    difference = causal_prefix_difference(
                        model, features, batch["frame_times"], output["state"]
                    )
                aggregate.causal_max = max(aggregate.causal_max, difference)
    return finalize_v56_metrics(aggregate, ridge_relative_gain)


def runtime_preflight(args, model, dataset, dino, tracker, device, amp_context):
    length = int(args.chunk_lengths.split(",")[0])
    indices = dataset.balanced_source_evaluation_indices(0, 2)
    loader = DataLoader(
        FixedEvaluationDataset(dataset, indices, length),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        batch = move_to_device(next(iter(loader)), device)
        features = dino(batch)
        evidence = tracker(batch, features.patches, features.grid_hw)
        teacher_indices = torch.arange(len(features.patches), device=device)
        with amp_context():
            output = model(
                features.patches,
                features.coordinates,
                features.valid,
                batch["frame_times"],
                evidence,
                teacher_indices,
                features.grid_hw,
            )
        collect_v56_diagnostics(
            model,
            features,
            evidence,
            output,
            amp_context,
            batch["requested_sequence_index"],
        )
        with amp_context():
            causal = causal_prefix_difference(
                model, features, batch["frame_times"], output["state"]
            )
    report = {
        "status": "passed",
        "items": len(features.patches),
        "history_length": length,
        "causal_prefix_max_difference": causal,
    }
    print(json.dumps({"runtime_preflight": report}, sort_keys=True), flush=True)
    return report


def condition_checks(metrics):
    return {
        "causal_rgb_only_student": metrics["causal_prefix_max_difference"] < 1e-6,
        "relation_groups_separated": metrics["relation_root_margin"] >= 0.30,
        "factorized_roots": metrics["effective_roots"] >= 1.50,
        "no_dominant_root_collapse": metrics["maximum_root_share"] <= 0.80,
        "track_correspondence": metrics["track_correspondence_margin"] >= 0.03,
        "track_shuffle_hurts_objective": metrics["track_shuffle_objective_delta"] >= 0.02,
        "decode_replacement_bounded": metrics["decode_replacement_fraction"] <= 0.05,
    }


def aggregate_checks(conditions):
    metrics = [condition["metrics"] for condition in conditions.values()]
    checks = [condition["checks"] for condition in conditions.values()]
    reappearance_events = sum(
        value["assignment_reappearance_events"] for value in metrics
    )
    identity_events = sum(value["identity_reappearance_events"] for value in metrics)
    deletion_events = sum(value["deletion_events"] for value in metrics)
    assignment_margin = sum(
        value["assignment_reappearance_margin"]
        * value["assignment_reappearance_events"]
        for value in metrics
    ) / max(reappearance_events, 1.0)
    identity_margin = sum(
        value["identity_reappearance_margin"] * value["identity_reappearance_events"]
        for value in metrics
    ) / max(identity_events, 1.0)
    deletion_inside = sum(value["deletion_inside_sum"] for value in metrics)
    deletion_outside = sum(value["deletion_outside_sum"] for value in metrics)
    deletion_ratio = deletion_inside / max(deletion_outside, 1e-8)
    aggregate = {
        "condition_count": len(conditions),
        "assignment_reappearance_events": reappearance_events,
        "assignment_reappearance_margin": assignment_margin,
        "identity_reappearance_events": identity_events,
        "identity_reappearance_margin": identity_margin,
        "deletion_events": deletion_events,
        "teacher_track_deletion_locality_ratio": deletion_ratio,
        "minimum_effective_roots": min(value["effective_roots"] for value in metrics),
        "maximum_root_share": max(value["maximum_root_share"] for value in metrics),
        "minimum_relation_root_margin": min(value["relation_root_margin"] for value in metrics),
        "minimum_motion_probe_gain": min(value["motion_probe_relative_gain"] for value in metrics),
        "minimum_visibility_probe_gain": min(value["visibility_probe_relative_gain"] for value in metrics),
        "maximum_decode_replacement_fraction": max(
            value["decode_replacement_fraction"] for value in metrics
        ),
    }
    gates = {
        "all_conditions_causal": all(value["causal_rgb_only_student"] for value in checks),
        "all_sources_factorized": all(
            value["factorized_roots"] and value["no_dominant_root_collapse"]
            for value in checks
        ),
        "all_sources_relation_separated": all(value["relation_groups_separated"] for value in checks),
        "all_sources_track_correspondence": all(value["track_correspondence"] for value in checks),
        "all_sources_track_shuffle_sensitive": all(value["track_shuffle_hurts_objective"] for value in checks),
        "assignment_reappearance_supported": reappearance_events >= 32 and assignment_margin >= 0.02,
        "identity_reappearance_supported": identity_events >= 32 and identity_margin >= 0.02,
        "teacher_track_deletion_local": deletion_events >= 32 and deletion_ratio >= 1.25,
        "motion_decodable": aggregate["minimum_motion_probe_gain"] >= 0.05,
        "visibility_decodable": aggregate["minimum_visibility_probe_gain"] >= 0.05,
        "decode_replacement_bounded": all(value["decode_replacement_bounded"] for value in checks),
    }
    return aggregate, gates


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v56 evaluation requires CUDA")
    checkpoint = torch.load(
        os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    validate_header(checkpoint)
    if int(checkpoint["global_step"]) != args.expected_step:
        raise ValueError(
            f"v56 evaluation expected step {args.expected_step}, got {checkpoint['global_step']}"
        )
    config = VerifiedRelationObjectStateConfig()
    config.validate()
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("v56 evaluation config differs from checkpoint")
    device = torch.device("cuda:0")
    model = VerifiedRelationObjectStateModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    lengths = tuple(int(value) for value in args.chunk_lengths.split(","))
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        0,
        args.seed,
    )
    runtime_preflight(
        args, model, dataset, dino, tracker, device, amp_context
    )
    run = init_wandb(args, checkpoint)
    conditions = {}
    condition_index = 0
    for source_index, source_name in enumerate(dataset.source_names):
        indices = dataset.balanced_source_evaluation_indices(
            source_index, args.items_per_source
        )
        for length in lengths:
            loader = DataLoader(
                FixedEvaluationDataset(dataset, indices, length),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.workers,
                persistent_workers=args.workers > 0,
            )
            metrics = evaluate_condition(
                args, model, loader, dino, tracker, device, amp_context
            )
            checks = condition_checks(metrics)
            key = f"{source_name}/H{length}"
            conditions[key] = {"metrics": metrics, "checks": checks}
            print(json.dumps({"condition": key, **conditions[key]}, sort_keys=True), flush=True)
            if run is not None:
                run.log(
                    {
                        "evaluation/condition_index": condition_index,
                        "evaluation/source_index": source_index,
                        "evaluation/history_length": length,
                        **{f"metric/{name}": value for name, value in metrics.items()},
                        **{f"check/{name}": int(value) for name, value in checks.items()},
                    },
                    step=condition_index,
                )
            condition_index += 1
    aggregate, gates = aggregate_checks(conditions)
    training_evidence_ready = all(gates.values())
    report = {
        "status": "completed",
        "training_evidence_gate_passed": training_evidence_ready,
        "deployment_promotion_ready": False,
        "promotion_blocker": "independent external object truth evaluation is required",
        "evaluation_scope": "source_balanced_training_teacher_diagnostics_not_independent_object_truth",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "training_git_commit": checkpoint["git_commit"],
        "evaluator_git_commit": args.evaluator_revision,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "data_index": os.path.abspath(args.data_index),
        "point_tracker_used_for_evaluation": True,
        "point_tracker_in_deployable_student": False,
        "conditions": conditions,
        "aggregate": aggregate,
        "gates": gates,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if run is not None:
        run.summary.update({
            "evaluation/status": report["status"],
            "evaluation/training_evidence_gate_passed": int(training_evidence_ready),
            "evaluation/deployment_promotion_ready": 0,
            "evaluation/report": output,
            **{f"aggregate/{name}": value for name, value in aggregate.items()},
            **{f"gate/{name}": int(value) for name, value in gates.items()},
        })
        for key, condition in conditions.items():
            for name, value in condition["metrics"].items():
                run.summary[f"condition/{key}/{name}"] = value
        run.finish()
    print(json.dumps({"report": output, "gates": gates}, sort_keys=True), flush=True)
    if args.require_training_evidence_gate and not training_evidence_ready:
        raise RuntimeError("v56 training-evidence evaluation gate failed")


if __name__ == "__main__":
    main()
