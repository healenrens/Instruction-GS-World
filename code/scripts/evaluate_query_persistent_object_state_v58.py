#!/usr/bin/env python3
"""Source-balanced held-teacher evaluation for v58 G2 promotion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import FrozenPointTrackerRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.query_object_teacher_v58 import (  # noqa: E402
    build_query_persistent_teacher_v58,
    observed_evidence_prefix,
    persistent_teacher_contract_metrics,
)
from igsw.adaptive_gaussian_wm.query_persistent_object_state_v58 import (  # noqa: E402
    QueryPersistentObjectStateModel,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v58_checkpointing import validate_header  # noqa: E402
from igsw.adaptive_gaussian_wm.v58_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    QueryPersistentObjectStateConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history_lengths", default="1,2,3,4")
    parser.add_argument("--teacher_future_frames", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--samples_per_condition", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dino_frame_batch", type=int, default=96)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="online")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="query-persistent-object-state-v58-eval")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def mean_records(records):
    keys = records[0].keys()
    return {name: sum(row[name] for row in records) / len(records) for name in keys}


@torch.no_grad()
def evaluate_condition(dataset, source_index, history, args, model, dino, tracker, device):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.samples_per_condition
    )
    rows = []
    total_frames = history + args.teacher_future_frames
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    for start in range(0, len(indices), args.batch):
        samples = [
            dataset[(index, total_frames)] for index in indices[start : start + args.batch]
        ]
        batch = move_batch(default_collate(samples), device)
        features = dino(batch)
        evidence = tracker(batch, features.patches, features.grid_hw)
        relation = build_trajectory_relation_teacher_v56(
            evidence, model.config, batch["frame_times"]
        )
        teacher = build_query_persistent_teacher_v58(
            evidence,
            relation,
            model.config,
            batch["frame_times"],
            observed_frames=history,
        )
        with amp_context():
            output = model(
                features.patches[:, :history],
                features.coordinates[:, :history],
                features.valid[:, :history],
                batch["frame_times"][:, :history],
                teacher,
                observed_evidence_prefix(evidence, history),
                features.grid_hw,
            )
        values = {
            **output["parts"],
            **persistent_teacher_contract_metrics(teacher),
            "query_support_perturbation": (
                output["primary"].support - output["negative"].support
            ).abs().mean(),
            "support_probability_mean": output["primary"].support.mean(),
        }
        rows.append({name: float(value.detach()) for name, value in values.items()})
    return mean_records(rows)


def condition_checks(metrics):
    binding = (
        metrics["heldout_support_positive"] >= 0.90
        and metrics["heldout_support_negative"] <= 0.10
    )
    visibility = metrics["lifecycle_occluded_candidate_fraction"] == 0.0 or (
        metrics["visibility_balanced_accuracy"] >= 0.55
        and metrics["visibility_f1"] >= 0.55
        and metrics["visibility_brier_gain_over_constant"] > 0.0
        and metrics["visibility_rate_relative_error"] <= 0.20
    )
    reappearance = (
        metrics["identity_reappearance_event_fraction"] == 0.0
        or metrics["identity_reappearance_margin"] >= 0.02
    )
    motion = (
        metrics["motion_valid_fraction"] == 0.0
        or metrics["dynamic_motion_relative_gain"] > 0.0
    )
    return {
        "binding": binding,
        "visibility": visibility,
        "reappearance": reappearance,
        "query_sensitive": metrics["query_support_perturbation"] > 1e-3,
        "motion_decodable": motion,
    }


def init_wandb(args, config):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir:
        raise ValueError("v58 evaluation W&B directory is required")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        job_type="v58-source-balanced-held-teacher-evaluation",
        config={**vars(args), **config.to_dict()},
    )


def main():
    args = parse_args()
    for name in ("data_index", "checkpoint", "dino_checkpoint", "tracker_checkpoint", "output"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    for name in ("data_index", "checkpoint", "dino_checkpoint", "tracker_checkpoint"):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v58 evaluation {name} is missing: {getattr(args, name)}")
    if not torch.cuda.is_available():
        raise RuntimeError("v58 evaluation requires CUDA")
    histories = tuple(int(value) for value in args.history_lengths.split(",") if value)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    validate_header(checkpoint)
    config = QueryPersistentObjectStateConfig(**checkpoint["config"])
    config.validate()
    if config.teacher_future_frames != args.teacher_future_frames:
        raise ValueError("v58 evaluation future-frame contract differs")
    chunks = ",".join(str(value + args.teacher_future_frames) for value in histories)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index, "train", chunks, args.temporal_step_ms, 0, args.seed
    )
    device = torch.device("cuda:0")
    model = QueryPersistentObjectStateModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    conditions = {}
    for source_index, source in enumerate(dataset.source_names):
        for history in histories:
            name = f"{source}/h{history}"
            conditions[name] = evaluate_condition(
                dataset, source_index, history, args, model, dino, tracker, device
            )
    checks = {name: condition_checks(value) for name, value in conditions.items()}
    aggregate = mean_records(list(conditions.values()))
    gate = {
        "all_conditions_bind": all(row["binding"] for row in checks.values()),
        "all_conditions_calibrate_visibility": all(
            row["visibility"] for row in checks.values()
        ),
        "all_observed_reappearance_is_stable": all(
            row["reappearance"] for row in checks.values()
        ),
        "all_conditions_are_query_sensitive": all(
            row["query_sensitive"] for row in checks.values()
        ),
        "all_motion_conditions_decode_motion": all(
            row["motion_decodable"] for row in checks.values()
        ),
        "occlusion_evidence_is_present": (
            aggregate["lifecycle_occluded_candidate_fraction"] > 0.0
        ),
    }
    report = {
        "status": "passed" if all(gate.values()) else "failed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint["global_step"],
        "git_commit": checkpoint["git_commit"],
        "data_index": args.data_index,
        "evaluation_scope": "held_training_teacher_not_independent_object_truth",
        "g2_promotion_only": True,
        "g3_independent_truth_required": True,
        "conditions": conditions,
        "condition_checks": checks,
        "aggregate": aggregate,
        "gate": gate,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run = init_wandb(args, config)
    if run is not None:
        for step, (condition, metrics) in enumerate(conditions.items()):
            run.log(
                {f"condition/{condition}/{name}": value for name, value in metrics.items()},
                step=step,
            )
        run.summary.update(
            {
                "gate/status": report["status"],
                **{f"gate/{name}": value for name, value in gate.items()},
                **{f"aggregate/{name}": value for name, value in aggregate.items()},
                "gate/report": args.output,
            }
        )
        run.finish()
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("v58 held-teacher G2 evaluation failed")


if __name__ == "__main__":
    main()
