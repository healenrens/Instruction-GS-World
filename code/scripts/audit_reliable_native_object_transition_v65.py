#!/usr/bin/env python3
"""Four-GPU six-source audit for reliable native object transitions v65."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.distributed_audit_v62 import (  # noqa: E402
    finish_distributed_audit_v62,
    gather_rank_payloads_v62,
    initialize_distributed_audit_v62,
    shard_indices_v62,
)
from igsw.adaptive_gaussian_wm.held_visual_transition_audit_v65 import (  # noqa: E402
    held_visual_transition_audit_v65,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.native_video_batch_v65 import (  # noqa: E402
    collate_native_video_batch_v65,
)
from igsw.adaptive_gaussian_wm.reliable_native_transition_runtime_v65 import (  # noqa: E402
    ReliableNativeTransitionAuditRuntimeV65,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v65_config import (  # noqa: E402
    CONTRACT,
    ReliableNativeTransitionConfigV65,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--items_per_source", type=int, default=32)
    parser.add_argument("--chunk_length", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--siglip_frame_batch", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument(
        "--wandb_group", default="object-transition-v65-native-reliable-audit"
    )
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


class MetricAccumulator:
    def __init__(self):
        self.numerators = {}
        self.denominators = {}

    def add(self, audit):
        for name, value in audit.metrics.items():
            weight = audit.valid[name].float()
            self.numerators[name] = self.numerators.get(name, 0.0) + float(
                (value.float() * weight).sum()
            )
            self.denominators[name] = self.denominators.get(name, 0.0) + float(
                weight.sum()
            )

    def merge(self, other):
        for name, value in other.numerators.items():
            self.numerators[name] = self.numerators.get(name, 0.0) + value
            self.denominators[name] = (
                self.denominators.get(name, 0.0) + other.denominators[name]
            )

    def means(self):
        return {
            name: value / max(self.denominators[name], 1.0)
            for name, value in self.numerators.items()
        }

    def valid_counts(self):
        return dict(self.denominators)

    def state_dict(self):
        return {
            "numerators": self.numerators,
            "denominators": self.denominators,
        }

    @classmethod
    def from_state_dict(cls, state):
        accumulator = cls()
        accumulator.numerators = dict(state["numerators"])
        accumulator.denominators = dict(state["denominators"])
        return accumulator


@torch.no_grad()
def evaluate_source(args, dataset, source_index, runtime, context):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    indices = shard_indices_v62(indices, context)
    accumulator = MetricAccumulator()
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        samples = [dataset[(index, args.chunk_length)] for index in selected]
        batch = collate_native_video_batch_v65(samples)
        device_batch = move_to_device(batch, context.device)
        bundle = runtime(device_batch)
        audit = held_visual_transition_audit_v65(
            bundle, device_batch["sequence_index"], runtime.config
        )
        accumulator.add(audit)
    return accumulator


def source_gate(metrics, config):
    checks = {
        "audit_valid_fraction": (
            metrics["audit_valid"] >= config.minimum_audit_valid_fraction
        ),
        "rolled_core_joint_coverage": metrics["rolled_core_valid"] >= 0.25,
        "shuffled_sample_joint_coverage": metrics["shuffled_sample_valid"] >= 0.25,
        "reliable_component": (
            metrics["component_reliability"]
            >= config.tracker_reliability_floor
        ),
        "future_image_beats_persistence": (
            metrics["visual_gain_over_persistence"] > 0.0
        ),
        "correct_core_beats_rolled_core": (
            metrics["visual_margin_rolled_core"] > 0.0
        ),
        "correct_sample_beats_shuffled_sample": (
            metrics["visual_margin_shuffled_sample"] > 0.0
        ),
    }
    return checks


def write_wandb(args, report):
    if args.wandb_mode == "disabled":
        return
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            "contract": report["contract"],
            "source_revision": args.source_revision,
            "chunk_length": args.chunk_length,
            "items_per_source": args.items_per_source,
            "tracker_grid_side": report["tracker_grid_side"],
            "native_tile_size": report["native_tile_size"],
            "transition_horizons": report["transition_horizons"],
        },
    )
    payload = {
        f"audit/overall/{name}": value for name, value in report["overall"].items()
    }
    for source, metrics in report["source_metrics"].items():
        payload.update(
            {f"audit/{source}/{name}": value for name, value in metrics.items()}
        )
    run.log(payload)
    run.summary.update(report)
    run.finish()


def main():
    args = parse_args()
    context = initialize_distributed_audit_v62()
    config = ReliableNativeTransitionConfigV65()
    config.validate()
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        str(args.chunk_length),
        "100",
        0,
        17,
        group_partition="held",
        held_group_stride=args.held_group_stride,
        preserve_native_rgb=True,
    )
    runtime = ReliableNativeTransitionAuditRuntimeV65(
        config,
        context.device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
    local = {
        name: evaluate_source(args, dataset, index, runtime, context)
        for index, name in enumerate(dataset.source_names)
    }
    payloads = gather_rank_payloads_v62(
        {name: accumulator.state_dict() for name, accumulator in local.items()},
        context,
    )
    if not context.is_main:
        finish_distributed_audit_v62()
        return
    sources = {name: MetricAccumulator() for name in dataset.source_names}
    for payload in payloads:
        for name, state in payload.items():
            sources[name].merge(MetricAccumulator.from_state_dict(state))
    overall = MetricAccumulator()
    for accumulator in sources.values():
        overall.merge(accumulator)
    source_metrics = {name: value.means() for name, value in sources.items()}
    source_valid_counts = {
        name: value.valid_counts() for name, value in sources.items()
    }
    source_checks = {
        name: source_gate(metrics, config)
        for name, metrics in source_metrics.items()
    }
    source_pass = {
        name: all(checks.values()) for name, checks in source_checks.items()
    }
    passing_sources = sum(source_pass.values())
    report = {
        "status": "completed",
        "contract": CONTRACT,
        "source_revision": args.source_revision,
        "data": os.path.abspath(args.data_index),
        "world_size": context.world_size,
        "items_per_source": args.items_per_source,
        "chunk_length": args.chunk_length,
        "native_full_frame_resize": False,
        "native_tile_size": config.native_tile_size,
        "native_tile_stride": config.native_tile_stride,
        "local_radii_pixels": list(config.local_radii_pixels),
        "local_feature_pooling": "multi-scale discrete token attention; no bilinear point sampling",
        "tracker_grid_side": config.tracker_grid_side,
        "point_track_count": (
            config.tracker_grid_side**2 * len(config.tracker_anchor_fractions)
        ),
        "tracker_measurement_role": "reliability-weighted auxiliary geometry; not dynamic GT",
        "binding_rule": "mutual multi-track core with robust IRLS transition; no single-track seed",
        "transition_horizons": list(config.transition_horizons),
        "transition_horizon_ms": [
            100 * horizon for horizon in config.transition_horizons
        ],
        "transition_fit_frames": [0, args.chunk_length // 2],
        "future_visual_target_frames": [
            args.chunk_length // 2 - 1 + horizon
            for horizon in config.transition_horizons
        ],
        "primary_target": "held native-image local DINO/SigLIP observation",
        "coordinate_target_role": "tracker-dependent auxiliary diagnostic only",
        "minimum_audit_valid_fraction": config.minimum_audit_valid_fraction,
        "source_checks": source_checks,
        "source_pass": source_pass,
        "passing_source_count": passing_sources,
        "required_passing_sources": 5,
        "decision": (
            "promote_native_reliable_transition_objective"
            if passing_sources >= 5
            else "reject_native_reliable_transition_objective"
        ),
        "source_metrics": source_metrics,
        "source_metric_valid_counts": source_valid_counts,
        "overall": overall.means(),
        "overall_metric_valid_counts": overall.valid_counts(),
        "source_counts": {
            name: accumulator.denominators.get("candidate_valid", 0.0)
            for name, accumulator in sources.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_wandb(args, report)
    print(json.dumps(report, sort_keys=True))
    finish_distributed_audit_v62()


if __name__ == "__main__":
    main()
