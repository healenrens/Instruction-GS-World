"""Held structural counterfactual evaluation for a trained v62 E0 checkpoint."""

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
from igsw.adaptive_gaussian_wm.distributed_audit_v62 import (  # noqa: E402
    finish_distributed_audit_v62,
    gather_rank_payloads_v62,
    initialize_distributed_audit_v62,
    shard_indices_v62,
)
from igsw.adaptive_gaussian_wm.object_codec_structural_eval_v62 import (  # noqa: E402
    evaluate_object_codec_structure_v62,
)
from igsw.adaptive_gaussian_wm.object_transition_teacher_runtime_v62 import (  # noqa: E402
    ObjectTransitionTeacherRuntimeV62,
)
from igsw.adaptive_gaussian_wm.teacher_object_autoencoder_v62 import (  # noqa: E402
    TeacherObjectAutoencoderV62,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v62_checkpointing import (  # noqa: E402
    load_e0_codec_checkpoint_v62,
)
from igsw.adaptive_gaussian_wm.v62_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    ObjectTransitionConfigV62,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument(
        "--wandb_group", default="object-transition-v62c-codec-structure"
    )
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


class ScalarAccumulator:
    def __init__(self):
        self.sums = {}
        self.count = 0

    def add(self, metrics, count):
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value) * count
        self.count += count

    def merge(self, other):
        for name, value in other.sums.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
        self.count += other.count

    def means(self):
        return {name: value / max(self.count, 1) for name, value in self.sums.items()}

    def state_dict(self):
        return {"sums": self.sums, "count": self.count}

    @classmethod
    def from_state_dict(cls, state):
        accumulator = cls()
        accumulator.sums = dict(state["sums"])
        accumulator.count = int(state["count"])
        return accumulator


@torch.no_grad()
def evaluate_source(args, dataset, source_index, model, teacher, context):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    indices = shard_indices_v62(indices, context)
    accumulator = ScalarAccumulator()
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        batch = default_collate([dataset[(index, 3)] for index in selected])
        observation = teacher(move_to_device(batch, context.device))
        metrics = evaluate_object_codec_structure_v62(model, observation)
        accumulator.add(metrics, len(selected))
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
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_step": report["checkpoint_step"],
        },
    )
    payload = {
        f"structural/overall/{name}": value for name, value in report["overall"].items()
    }
    for source, metrics in report["source_metrics"].items():
        payload.update(
            {f"structural/{source}/{name}": value for name, value in metrics.items()}
        )
    run.log(payload, step=report["checkpoint_step"])
    run.summary.update(report)
    run.finish()


def main():
    args = parse_args()
    context = initialize_distributed_audit_v62()
    config = ObjectTransitionConfigV62()
    checkpoint = load_e0_codec_checkpoint_v62(args.checkpoint, config)
    model = TeacherObjectAutoencoderV62(config).to(context.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    teacher = ObjectTransitionTeacherRuntimeV62(
        config,
        context.device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
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
    local_accumulators = {
        name: evaluate_source(args, dataset, index, model, teacher, context)
        for index, name in enumerate(dataset.source_names)
    }
    rank_payloads = gather_rank_payloads_v62(
        {
            name: accumulator.state_dict()
            for name, accumulator in local_accumulators.items()
        },
        context,
    )
    if not context.is_main:
        finish_distributed_audit_v62()
        return
    source_accumulators = {name: ScalarAccumulator() for name in dataset.source_names}
    for rank_payload in rank_payloads:
        for name, state in rank_payload.items():
            source_accumulators[name].merge(ScalarAccumulator.from_state_dict(state))
    overall_accumulator = ScalarAccumulator()
    for accumulator in source_accumulators.values():
        overall_accumulator.merge(accumulator)
    report = {
        "status": "completed",
        "contract": "object_codec_structural_counterfactual_v62c_unseen_query_v2",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "source_revision": args.source_revision,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data_index),
        "held_group_stride": args.held_group_stride,
        "items_per_source": args.items_per_source,
        "world_size": context.world_size,
        "conditions": [
            "normal",
            "full_context_held_query",
            "continuous_coordinate_holdout",
            "query_swap",
            "all_scene",
            "merge_all",
            "carrier_delete",
            "carrier_swap",
            "split_by_time",
        ],
        "continuous_query_protocol": {
            "encode_indices": "even point-track indices",
            "decode_indices": "odd point-track indices",
            "full_context_reference": "all point tracks encoded, odd indices decoded",
            "target": "external teacher support, DINO, SigLIP, visibility, lifecycle",
        },
        "metric_semantics": {
            "absolute_error": "support + DINO + SigLIP + 0.25 visibility + 0.25 lifecycle; lower is better",
            "cosine_error": "one minus cosine similarity to the external teacher feature; lower is better",
            "support_soft_iou": "soft intersection over union against external teacher support; higher is better",
            "gap_recovery": "one minus model error divided by compact baseline error; higher is better",
            "ratio_to_full_context": "half-track encoding error divided by all-track encoding error on identical odd queries; one is parity",
        },
        "source_metrics": {
            name: accumulator.means()
            for name, accumulator in source_accumulators.items()
        },
        "source_counts": {
            name: accumulator.count for name, accumulator in source_accumulators.items()
        },
        "overall": overall_accumulator.means(),
        "overall_count": overall_accumulator.count,
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
