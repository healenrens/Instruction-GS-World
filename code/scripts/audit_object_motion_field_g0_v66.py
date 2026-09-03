#!/usr/bin/env python3
"""Four-GPU larger-sample G0 audit for compact object motion fields."""

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
from igsw.adaptive_gaussian_wm.held_motion_field_audit_v66 import (  # noqa: E402
    held_motion_field_audit_v66,
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
from igsw.adaptive_gaussian_wm.v66_config import (  # noqa: E402
    CONTRACT,
    ObjectMotionFieldAuditConfigV66,
)


BOOTSTRAP_METRICS = (
    "translation_gain_over_persistence",
    "affine_gain_over_persistence",
    "motion_field_gain_over_persistence",
    "motion_field_margin_over_translation",
    "motion_field_margin_over_affine",
    "motion_field_margin_over_rolled",
    "motion_field_margin_over_shuffled",
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
    parser.add_argument("--items_per_source", type=int, default=256)
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
    parser.add_argument("--wandb_group", default="object-motion-field-g0-v66")
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate_source(args, dataset, source_index, runtime, context):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    indices = shard_indices_v62(indices, context)
    rows = []
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        samples = [dataset[(index, args.chunk_length)] for index in selected]
        batch = collate_native_video_batch_v65(samples)
        device_batch = move_to_device(batch, context.device)
        bundle = runtime(device_batch)
        audit = held_motion_field_audit_v66(
            bundle, device_batch["sequence_index"], runtime.config
        )
        metric_values = {
            name: value.detach().float().cpu().tolist()
            for name, value in audit.metrics.items()
        }
        valid_values = {
            name: value.detach().bool().cpu().tolist()
            for name, value in audit.valid.items()
        }
        for position, dataset_index in enumerate(selected):
            rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "metrics": {
                        name: float(values[position])
                        for name, values in metric_values.items()
                    },
                    "valid": {
                        name: bool(values[position])
                        for name, values in valid_values.items()
                    },
                }
            )
    return rows


def metric_values(rows, name, mask):
    return torch.tensor(
        [
            row["metrics"][name]
            for row, selected in zip(rows, mask)
            if selected and row["valid"][name]
        ],
        dtype=torch.float32,
    )


def bootstrap_mean(values, samples, seed, minimum_count):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "evidence_sufficient": False,
        }
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(values), (samples, len(values)), generator=generator
    )
    estimates = values[indices].mean(dim=1)
    interval = torch.quantile(estimates, torch.tensor((0.025, 0.975)))
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
        "evidence_sufficient": len(values) >= minimum_count,
    }


def summarize_stratum(rows, mask, config, seed):
    metric_names = tuple(rows[0]["metrics"])
    means, counts, values_by_metric = {}, {}, {}
    for name in metric_names:
        values = metric_values(rows, name, mask)
        values_by_metric[name] = values
        counts[name] = len(values)
        means[name] = float(values.mean()) if len(values) else 0.0
    bootstrap = {
        name: bootstrap_mean(
            values_by_metric[name],
            config.bootstrap_samples,
            seed + offset,
            config.minimum_high_change_samples,
        )
        for offset, name in enumerate(BOOTSTRAP_METRICS)
    }
    return {"means": means, "valid_counts": counts, "bootstrap": bootstrap}


def source_summary(rows, config, seed):
    all_mask = [True] * len(rows)
    change_values = metric_values(rows, "high_change_score", all_mask)
    threshold = (
        float(torch.quantile(change_values, config.high_change_quantile))
        if len(change_values)
        else 0.0
    )
    high_change_mask = [
        row["valid"]["high_change_score"]
        and row["metrics"]["high_change_score"] >= threshold
        for row in rows
    ]
    return {
        "sample_count": len(rows),
        "high_change_threshold": threshold,
        "high_change_sample_count": sum(high_change_mask),
        "all_valid": summarize_stratum(rows, all_mask, config, seed),
        "high_change": summarize_stratum(
            rows, high_change_mask, config, seed + 100
        ),
    }, high_change_mask


def source_gate(summary, config):
    all_means = summary["all_valid"]["means"]
    high = summary["high_change"]["bootstrap"]

    def supported_positive(metric):
        interval = high[metric]
        return (
            interval["count"] >= config.minimum_high_change_samples
            and interval["ci95_low"] > 0.0
        )

    checks = {
        "audit_valid_fraction": (
            all_means["audit_valid"] >= config.minimum_audit_valid_fraction
        ),
        "enough_high_change_samples": (
            summary["high_change_sample_count"]
            >= config.minimum_high_change_samples
        ),
        "all_valid_not_worse_than_persistence": (
            all_means["motion_field_gain_over_persistence"] >= 0.0
        ),
        "high_change_beats_persistence": (
            supported_positive("motion_field_gain_over_persistence")
        ),
        "high_change_beats_affine": (
            supported_positive("motion_field_margin_over_affine")
        ),
        "high_change_beats_rolled_core": (
            supported_positive("motion_field_margin_over_rolled")
        ),
        "high_change_beats_shuffled_sample": (
            supported_positive("motion_field_margin_over_shuffled")
        ),
    }
    return checks


def flatten_wandb(report):
    payload = {
        "audit/passing_source_count": report["passing_source_count"],
        "audit/required_passing_sources": report["required_passing_sources"],
    }
    for stratum in ("all_valid", "high_change"):
        for metric in BOOTSTRAP_METRICS:
            payload[f"audit/overall/{stratum}/{metric}"] = report["overall"][
                stratum
            ]["means"][metric]
    for metric, interval in report["overall"]["high_change"]["bootstrap"].items():
        payload[f"audit/overall/high_change/{metric}_ci95_low"] = interval[
            "ci95_low"
        ]
        payload[f"audit/overall/high_change/{metric}_ci95_high"] = interval[
            "ci95_high"
        ]
    for source, summary in report["source_summaries"].items():
        payload[f"audit/{source}/coverage"] = summary["all_valid"]["means"][
            "audit_valid"
        ]
        payload[f"audit/{source}/high_change_threshold"] = summary[
            "high_change_threshold"
        ]
        payload[f"audit/{source}/high_change_count"] = summary[
            "high_change_sample_count"
        ]
        for stratum in ("all_valid", "high_change"):
            means = summary[stratum]["means"]
            for metric in BOOTSTRAP_METRICS:
                payload[f"audit/{source}/{stratum}/{metric}"] = means[metric]
        for metric, interval in summary["high_change"]["bootstrap"].items():
            payload[f"audit/{source}/high_change/{metric}_ci95_low"] = interval[
                "ci95_low"
            ]
            payload[f"audit/{source}/high_change/{metric}_ci95_high"] = interval[
                "ci95_high"
            ]
    return payload


def resolve_wandb_entity(wandb, requested_entity):
    return requested_entity or wandb.Api(timeout=120).default_entity


def write_wandb(args, report):
    if args.wandb_mode == "disabled":
        return
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    entity = resolve_wandb_entity(wandb, args.wandb_entity)
    run = wandb.init(
        project=args.wandb_project,
        entity=entity,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            "contract": report["contract"],
            "source_revision": args.source_revision,
            "items_per_source": args.items_per_source,
            "chunk_length": args.chunk_length,
            "local_motion_modes": report["local_motion_modes"],
            "bootstrap_samples": report["bootstrap_samples"],
        },
    )
    run.log(flatten_wandb(report))
    run.summary.update(report)
    run.finish()


def main():
    args = parse_args()
    context = initialize_distributed_audit_v62()
    config = ObjectMotionFieldAuditConfigV66()
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
    payloads = gather_rank_payloads_v62(local, context)
    if not context.is_main:
        finish_distributed_audit_v62()
        return
    source_rows = {name: [] for name in dataset.source_names}
    for payload in payloads:
        for name, rows in payload.items():
            source_rows[name].extend(rows)
    source_summaries, high_change_masks = {}, {}
    for index, name in enumerate(dataset.source_names):
        source_summaries[name], high_change_masks[name] = source_summary(
            source_rows[name], config, 6600 + 1000 * index
        )
    pooled_rows = []
    pooled_high_change_mask = []
    for name in dataset.source_names:
        pooled_rows.extend(source_rows[name])
        pooled_high_change_mask.extend(high_change_masks[name])
    overall = {
        "all_valid": summarize_stratum(
            pooled_rows, [True] * len(pooled_rows), config, 16600
        ),
        "high_change": summarize_stratum(
            pooled_rows, pooled_high_change_mask, config, 17600
        ),
        "high_change_sample_count": sum(pooled_high_change_mask),
    }
    source_checks = {
        name: source_gate(summary, config)
        for name, summary in source_summaries.items()
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
        "total_sample_count": sum(len(rows) for rows in source_rows.values()),
        "chunk_length": args.chunk_length,
        "transition_horizons": list(config.transition_horizons),
        "transition_horizon_ms": [
            100 * horizon for horizon in config.transition_horizons
        ],
        "model_classes": ["persistence", "translation", "affine", "motion_field"],
        "motion_field": "global affine plus shared object-local RBF residual modes",
        "local_motion_modes": config.local_motion_modes,
        "bootstrap_samples": config.bootstrap_samples,
        "high_change_definition": (
            "per-source upper half of held native-image persistence visual error"
        ),
        "primary_target": "held native-image local DINO/SigLIP observation",
        "coordinate_target_role": "tracker-dependent auxiliary diagnostic only",
        "historical_checkpoint_used": False,
        "student_or_dynamics_trained": False,
        "overall": overall,
        "source_summaries": source_summaries,
        "source_checks": source_checks,
        "source_pass": source_pass,
        "passing_source_count": passing_sources,
        "required_passing_sources": 5,
        "decision": (
            "promote_object_motion_field_g0"
            if passing_sources >= 5
            else "reject_object_motion_field_g0"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    finish_distributed_audit_v62()
    write_wandb(args, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
