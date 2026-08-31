"""Held-data structural audit of the frozen v62 object teacher target."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_audit_runtime_v62 import (  # noqa: E402
    ObjectTransitionAuditRuntimeV62,
)
from igsw.adaptive_gaussian_wm.teacher_target_structural_metrics_v62 import (  # noqa: E402
    teacher_target_structural_metrics_v62,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v62_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    ObjectTransitionConfigV62,
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--siglip_frame_batch", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--wandb_group", default="object-transition-v62b-teacher-audit")
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


class MetricAccumulator:
    def __init__(self):
        self.numerators = {}
        self.denominators = {}

    def add(self, metrics, valid):
        valid = valid.float()
        for name, value in metrics.items():
            weight = torch.ones_like(valid) if name == "object_valid" else valid
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

    def evidence(self):
        return {
            name: {
                "numerator": value,
                "denominator": self.denominators[name],
            }
            for name, value in self.numerators.items()
        }


@torch.no_grad()
def evaluate_source(args, dataset, source_index, runtime, device):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    accumulator = MetricAccumulator()
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        batch = default_collate([dataset[(index, 3)] for index in selected])
        bundle = runtime(move_to_device(batch, device))
        metrics = teacher_target_structural_metrics_v62(bundle)
        accumulator.add(metrics, bundle.observation.object_valid)
    return accumulator


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
            "items_per_source": args.items_per_source,
            "held_group_stride": args.held_group_stride,
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
    device = torch.device("cuda")
    config = ObjectTransitionConfigV62()
    config.validate()
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        "3",
        "100",
        0,
        17,
        group_partition="held",
        held_group_stride=args.held_group_stride,
    )
    runtime = ObjectTransitionAuditRuntimeV62(
        config,
        device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
    source_accumulators = {
        name: evaluate_source(args, dataset, index, runtime, device)
        for index, name in enumerate(dataset.source_names)
    }
    overall_accumulator = MetricAccumulator()
    for accumulator in source_accumulators.values():
        overall_accumulator.merge(accumulator)
    report = {
        "status": "completed",
        "contract": "teacher_target_structural_audit_v62b",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "source_revision": args.source_revision,
        "data": os.path.abspath(args.data_index),
        "held_group_stride": args.held_group_stride,
        "items_per_source": args.items_per_source,
        "teacher_only": True,
        "checkpoint_used": False,
        "corruption": "deterministic_half_track_roll",
        "source_metrics": {
            name: accumulator.means()
            for name, accumulator in source_accumulators.items()
        },
        "source_evidence": {
            name: accumulator.evidence()
            for name, accumulator in source_accumulators.items()
        },
        "overall": overall_accumulator.means(),
        "overall_evidence": overall_accumulator.evidence(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_wandb(args, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
