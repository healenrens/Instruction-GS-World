#!/usr/bin/env python3
"""Audit v58 query supervision on every real-video source and history length."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import default_collate

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
from igsw.adaptive_gaussian_wm.query_object_coverage_v58 import (  # noqa: E402
    coverage_row,
    future_track_shuffle,
    select_query_teacher,
    summarize_query_coverage,
)
from igsw.adaptive_gaussian_wm.query_object_teacher_v58 import (  # noqa: E402
    build_query_persistent_teacher_v58,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v56_data_contract import (  # noqa: E402
    audit_decode_frontier,
)
from igsw.adaptive_gaussian_wm.v58_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    QueryPersistentObjectStateConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--history_lengths", default="1,2,3,4")
    parser.add_argument("--teacher_future_frames", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400,800")
    parser.add_argument("--samples_per_condition", type=int, default=16)
    parser.add_argument("--audit_batch", type=int, default=4)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="online")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="query_object_coverage_v58")
    parser.add_argument("--wandb_group", default="query-object-v58-gates")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item)
    if not lengths or tuple(sorted(set(lengths))) != lengths or min(lengths) < 1:
        raise ValueError("v58 history lengths must be unique, increasing, and positive")
    return lengths


def move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def init_wandb(args, config):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir:
        raise ValueError("v58 coverage W&B directory is required")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={**vars(args), **config.to_dict()},
    )


def audit_condition(dataset, source_index, history, args, dino, tracker, config, device):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.samples_per_condition
    )
    rows = []
    total_frames = history + args.teacher_future_frames
    for start in range(0, len(indices), args.audit_batch):
        samples = [
            dataset[(index, total_frames)]
            for index in indices[start : start + args.audit_batch]
        ]
        batch = move_batch(default_collate(samples), device)
        features = dino(batch)
        evidence = tracker(batch, features.patches, features.grid_hw)
        relation = build_trajectory_relation_teacher_v56(
            evidence, config, batch["frame_times"]
        )
        teacher = build_query_persistent_teacher_v58(
            evidence, relation, config, batch["frame_times"], observed_frames=history
        )
        shuffled_evidence = future_track_shuffle(evidence, history)
        shuffled_relation = build_trajectory_relation_teacher_v56(
            shuffled_evidence, config, batch["frame_times"]
        )
        shuffled_teacher = build_query_persistent_teacher_v58(
            shuffled_evidence,
            shuffled_relation,
            config,
            batch["frame_times"],
            observed_frames=history,
        )
        for item, actual_source in enumerate(batch["source_index"].tolist()):
            rows.append(
                coverage_row(
                    dataset.source_names[int(actual_source)],
                    history,
                    select_query_teacher(teacher, item),
                    select_query_teacher(shuffled_teacher, item),
                )
            )
    return rows


def main():
    args = parse_args()
    path_names = ("data_index", "decode_report", "dino_checkpoint", "tracker_checkpoint", "output")
    for name in path_names:
        setattr(args, name, os.path.abspath(getattr(args, name)))
    for name in path_names[:-1]:
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v58 coverage {name} is missing: {getattr(args, name)}")
    if not torch.cuda.is_available():
        raise RuntimeError("v58 coverage audit requires CUDA")
    histories = parse_lengths(args.history_lengths)
    if args.teacher_future_frames < 1 or min(args.samples_per_condition, args.audit_batch) < 1:
        raise ValueError("v58 coverage dimensions must be positive")
    config = QueryPersistentObjectStateConfig(
        teacher_future_frames=args.teacher_future_frames
    )
    config.validate()
    total_lengths = ",".join(str(value + args.teacher_future_frames) for value in histories)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        chunk_lengths=total_lengths,
        temporal_step_ms=args.temporal_step_ms,
        max_items=0,
        seed=args.seed,
    )
    decode = audit_decode_frontier(
        args.decode_report, args.data_index, args.seed, dataset.source_names
    )
    if decode["status"] != "passed":
        raise RuntimeError(f"v58 decode frontier failed: {decode}")
    device = torch.device("cuda:0")
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    rows = []
    for source_index in range(len(dataset.source_names)):
        for history in histories:
            rows.extend(
                audit_condition(
                    dataset, source_index, history, args, dino, tracker, config, device
                )
            )
    coverage = summarize_query_coverage(
        rows, config, dataset.source_names, histories
    )
    report = {
        "status": coverage["status"],
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.source_revision,
        "data_index": args.data_index,
        "decode_report": args.decode_report,
        "dino_checkpoint": args.dino_checkpoint,
        "tracker_checkpoint": args.tracker_checkpoint,
        "tracker_bidirectional": config.tracker_bidirectional,
        "tracker_include_observed_current_anchor": (
            config.tracker_include_observed_current_anchor
        ),
        "history_lengths": args.history_lengths,
        "teacher_future_frames": args.teacher_future_frames,
        "temporal_step_ms": args.temporal_step_ms,
        "samples_per_condition": args.samples_per_condition,
        "source_names": list(dataset.source_names),
        "student_observation": "rgb_dino_prefix_and_current_query_only",
        "teacher_observation": "full_clip_tracks_and_dino_affinity",
        "dynamics_present": False,
        "latent_effect_present": False,
        "historical_checkpoint_used": False,
        "decode_frontier_audit": decode,
        "query_coverage": coverage,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run = init_wandb(args, config)
    if run is not None:
        for index, (condition, metrics) in enumerate(coverage["conditions"].items()):
            run.log(
                {f"coverage/{condition}/{name}": value for name, value in metrics.items()},
                step=index,
            )
        run.summary.update(
            {
                "gate/status": coverage["status"],
                **{f"aggregate/{name}": value for name, value in coverage["aggregate"].items()},
                **{f"check/{name}": value for name, value in coverage["checks"].items()},
                "gate/report": args.output,
            }
        )
        run.finish()
    print(json.dumps(report, sort_keys=True), flush=True)
    if coverage["status"] != "passed":
        raise RuntimeError("v58 real query coverage gate failed")


if __name__ == "__main__":
    main()
