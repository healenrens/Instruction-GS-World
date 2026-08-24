#!/usr/bin/env python3
"""Source-balanced held evaluation of v59 effect necessity and sufficiency."""

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

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.latent_object_transition_v59 import (  # noqa: E402
    QueryObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_objective_v59 import (  # noqa: E402
    object_transition_objective_v59,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.query_object_teacher_v57 import (  # noqa: E402
    build_query_object_teacher_v57,
)
from igsw.adaptive_gaussian_wm.query_transition_target_v59 import (  # noqa: E402
    build_query_transition_target_v59,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v59_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    ObjectTransitionConfig,
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
    parser.add_argument("--temporal_step_ms", default="100,200,400,800")
    parser.add_argument("--samples_per_condition", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dino_frame_batch", type=int, default=96)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument(
        "--wandb_mode", choices=("disabled", "online", "offline"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="object-transition-objective-v59-eval")
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
def evaluate_condition(
    dataset, source_index, history, args, model, dino, tracker, device
):
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
            dataset[(index, total_frames)]
            for index in indices[start : start + args.batch]
        ]
        batch = move_batch(default_collate(samples), device)
        features = dino(batch)
        evidence = tracker(batch, features.patches, features.grid_hw)
        relation = build_trajectory_relation_teacher_v56(
            evidence, model.config, batch["frame_times"]
        )
        binding = build_query_object_teacher_v57(
            evidence, relation, model.config, observed_frames=history
        )
        target = build_query_transition_target_v59(
            evidence,
            relation,
            binding,
            batch["frame_times"],
            history,
            model.config,
        )
        with amp_context():
            output = model(
                features.patches[:, :history],
                features.coordinates[:, :history],
                features.valid[:, :history],
                batch["frame_times"][:, :history],
                binding.query_coordinate,
                target,
            )
            _, parts = object_transition_objective_v59(output, target, model.config)
        rows.append({name: float(value.detach()) for name, value in parts.items()})
    metrics = mean_records(rows)
    for baseline in ("zero", "shuffled", "persistence"):
        numerator = (
            metrics[f"{baseline}_active_error"] - metrics["correct_active_error"]
        )
        metrics[f"aggregate_gain_over_{baseline}"] = numerator / max(
            metrics[f"{baseline}_active_error"], 1e-6
        )
    return metrics


def init_wandb(args, config, checkpoint):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir:
        raise ValueError("v59 evaluation W&B directory is required")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        job_type="v59-object-transition-held-evaluation",
        config={
            **vars(args),
            **config.to_dict(),
            "checkpoint_step": checkpoint["global_step"],
        },
    )


def main():
    args = parse_args()
    for name in (
        "data_index",
        "checkpoint",
        "dino_checkpoint",
        "tracker_checkpoint",
        "output",
    ):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    for name in ("data_index", "checkpoint", "dino_checkpoint", "tracker_checkpoint"):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v59 evaluation {name} is missing: {getattr(args, name)}")
    if not torch.cuda.is_available():
        raise RuntimeError("v59 evaluation requires CUDA")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v59 evaluation requires a version-59 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v59 evaluation checkpoint architecture differs")
    config = ObjectTransitionConfig(**checkpoint["config"])
    config.validate()
    histories = tuple(int(value) for value in args.history_lengths.split(",") if value)
    chunks = ",".join(str(value + args.teacher_future_frames) for value in histories)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index, "train", chunks, args.temporal_step_ms, 0, args.seed
    )
    device = torch.device("cuda:0")
    model = QueryObjectTransitionModel(config).to(device).eval()
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
    aggregate = mean_records(list(conditions.values()))
    for baseline in ("zero", "shuffled", "persistence"):
        numerator = (
            aggregate[f"{baseline}_active_error"] - aggregate["correct_active_error"]
        )
        aggregate[f"aggregate_gain_over_{baseline}"] = numerator / max(
            aggregate[f"{baseline}_active_error"], 1e-6
        )
    condition_checks = {
        name: {
            baseline: metrics[f"aggregate_gain_over_{baseline}"] >= 0.10
            for baseline in ("zero", "shuffled", "persistence")
        }
        for name, metrics in conditions.items()
    }
    gate = {
        "motion_active_evidence_present": aggregate["motion_active_fraction"] > 0.0,
        "aggregate_beats_zero_by_10pct": (
            aggregate["aggregate_gain_over_zero"] >= 0.10
        ),
        "aggregate_beats_shuffled_by_10pct": (
            aggregate["aggregate_gain_over_shuffled"] >= 0.10
        ),
        "aggregate_beats_persistence_by_10pct": (
            aggregate["aggregate_gain_over_persistence"] >= 0.10
        ),
    }
    report = {
        "status": "passed" if all(gate.values()) else "failed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint["global_step"],
        "teacher_dependent_objective_evaluation": True,
        "independent_object_validity_claim": False,
        "gate": gate,
        "aggregate": aggregate,
        "condition_checks": condition_checks,
        "conditions": conditions,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run = init_wandb(args, config, checkpoint)
    if run is not None:
        run.log({f"eval/aggregate/{name}": value for name, value in aggregate.items()})
        run.log({f"eval/gate/{name}": float(value) for name, value in gate.items()})
        run.summary.update(report)
        run.finish()
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
