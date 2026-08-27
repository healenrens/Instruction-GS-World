"""Held source-balanced evaluation for posterior carrier Dynamics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_carrier_dynamics_model_v61 import (  # noqa: E402
    ContinuousCarrierDynamicsModelV61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_world_model_v61 import (  # noqa: E402
    ContinuousCarrierObjectWorldModelV61,
)
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
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v61_config import (  # noqa: E402
    CHECKPOINT_VERSION,
    DYNAMICS_ARCHITECTURE,
    config_for_variant,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip2_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_lengths", default="4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--student_frame_batch", type=int, default=16)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--wandb_group", default="continuous-carrier-dynamics-v61-eval")
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def _batch(sample, device):
    values = {
        name: value[None] if torch.is_tensor(value) else value
        for name, value in sample.items()
    }
    return move_to_device(values, device)


def _mean(records):
    names = sorted({name for record in records for name in record})
    return {
        name: sum(record[name] for record in records if name in record)
        / sum(name in record for record in records)
        for name in names
    }


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v61 Dynamics evaluator requires checkpoint version 61")
    if checkpoint.get("architecture") != DYNAMICS_ARCHITECTURE:
        raise ValueError("v61 Dynamics evaluator architecture differs")
    variant = checkpoint["config"]["variant"]
    capacity = checkpoint["effect_capacity"]
    config = config_for_variant(variant)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        max_items=0,
        seed=checkpoint["args"]["seed"],
        group_partition="held",
        held_group_stride=args.held_group_stride,
    )
    if not dataset.source_audit_indices:
        raise ValueError("v61 Dynamics evaluator requires source audit indices")
    state_model = ContinuousCarrierObjectWorldModelV61(
        config,
        args.dino_checkpoint,
        args.siglip2_checkpoint,
        args.student_frame_batch,
    ).to(device)
    model = ContinuousCarrierDynamicsModelV61(state_model, capacity).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config={
                "checkpoint": args.checkpoint,
                "checkpoint_step": checkpoint["global_step"],
                "variant": variant,
                "effect_capacity": capacity,
                "chunk_lengths": args.chunk_lengths,
                "temporal_step_ms": args.temporal_step_ms,
                "group_partition": "held",
                "held_group_stride": args.held_group_stride,
            },
        )
    chunk_lengths = tuple(int(value) for value in args.chunk_lengths.split(","))
    records, source_records = [], {name: [] for name in dataset.source_names}
    condition = 0
    with torch.no_grad():
        for source_index, indices in enumerate(dataset.source_audit_indices):
            for base_index in indices:
                for chunk_length in chunk_lengths:
                    batch = _batch(dataset[(base_index, chunk_length)], device)
                    teacher_features = dino(batch)
                    evidence = tracker(
                        batch, teacher_features.patches, teacher_features.grid_hw
                    )
                    relation = build_trajectory_relation_teacher_v56(
                        evidence, config, batch["frame_times"]
                    )
                    with amp_context():
                        output = model(batch, evidence, relation)
                    metrics = {
                        name: float(value) for name, value in output["parts"].items()
                    }
                    metrics.update(
                        {
                            "source_index": float(source_index),
                            "chunk_length": float(chunk_length),
                            "temporal_step_seconds": float(
                                batch["temporal_step_seconds"].mean()
                            ),
                        }
                    )
                    records.append(metrics)
                    source_records[dataset.source_names[source_index]].append(metrics)
                    if run is not None:
                        run.log(
                            {
                                **{
                                    f"condition/{name}": value
                                    for name, value in metrics.items()
                                },
                                "condition/index": condition,
                            },
                            step=condition,
                        )
                    condition += 1
    macro = _mean(records)
    sources = {name: _mean(values) for name, values in source_records.items()}
    accepted = (
        macro["persistence_relative_gain"] >= 0.10
        and macro["zero_effect_relative_gain"] >= 0.10
        and macro["shuffled_effect_relative_gain"] >= 0.10
    )
    report = {
        "status": "completed",
        "accepted": accepted,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": DYNAMICS_ARCHITECTURE,
        "variant": variant,
        "effect_capacity": capacity,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": checkpoint["global_step"],
        "conditions": len(records),
        "group_partition": "held",
        "held_group_stride": args.held_group_stride,
        "acceptance": {
            "minimum_persistence_relative_gain": 0.10,
            "minimum_zero_effect_relative_gain": 0.10,
            "minimum_shuffled_effect_relative_gain": 0.10,
        },
        "macro": macro,
        "sources": sources,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if run is not None:
        run.log(
            {f"macro/{name}": value for name, value in macro.items()}, step=condition
        )
        for source, metrics in sources.items():
            run.summary.update(
                {f"source/{source}/{name}": value for name, value in metrics.items()}
            )
        run.summary.update(
            {
                "evaluation/status": "completed",
                "evaluation/accepted": accepted,
                "evaluation/conditions": len(records),
                "evaluation/report": os.path.abspath(args.output),
            }
        )
        run.finish()
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
