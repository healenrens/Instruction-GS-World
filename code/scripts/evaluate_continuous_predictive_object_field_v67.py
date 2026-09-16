#!/usr/bin/env python3
"""Independent held-coordinate evaluation for v67, synchronized to W&B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_predictive_object_field_v67 import (  # noqa: E402
    ContinuousPredictiveObjectFieldV67,
)
from igsw.adaptive_gaussian_wm.continuous_predictive_teacher_v67 import (  # noqa: E402
    ContinuousPredictiveTeacherRuntimeV67,
    teacher_relation_evidence_v67,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.native_video_batch_v65 import (  # noqa: E402
    collate_native_video_batch_v65,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v67_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STATE_STAGE,
    STAGES,
    ContinuousPredictiveObjectFieldConfigV67,
)
from igsw.distributed import init_torchrun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--items_per_source", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--expected_world_size", type=int, default=8)
    parser.add_argument("--dino_frame_batch", type=int, default=96)
    parser.add_argument("--siglip_frame_batch", type=int, default=96)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument(
        "--wandb_group", default="continuous-predictive-object-field-v67-eval"
    )
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass
class Accumulator:
    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def add(self, values: dict[str, torch.Tensor], batch_size: int) -> None:
        for name, value in values.items():
            scalar = float(value.detach().float().mean())
            self.sums[name] = self.sums.get(name, 0.0) + scalar * batch_size
        self.count += batch_size

    def merge(self, other: "Accumulator") -> None:
        for name, value in other.sums.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
        self.count += other.count

    def means(self) -> dict[str, float]:
        return {name: value / max(self.count, 1) for name, value in self.sums.items()}

    def state_dict(self) -> dict:
        return {"sums": self.sums, "count": self.count}

    @classmethod
    def from_state_dict(cls, state):
        return cls(dict(state["sums"]), int(state["count"]))


def load_model(args, config, device):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "config": config.to_dict(),
        "historical_checkpoint_used": False,
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    require(not differences, f"v67 evaluation checkpoint differs: {differences}")
    model = ContinuousPredictiveObjectFieldV67(config, args.stage)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def relation_swap_metrics(output, target, config):
    relation, weight = teacher_relation_evidence_v67(
        target, config.source_frame, config
    )
    logits = output["source_relation"].logits.float()
    correct = F.binary_cross_entropy_with_logits(logits, relation, reduction="none")
    wrong = relation.roll(1, dims=1)
    shuffled = F.binary_cross_entropy_with_logits(logits, wrong, reduction="none")
    denominator = weight.sum().clamp_min(1e-6)
    correct = (correct * weight).sum() / denominator
    shuffled = (shuffled * weight).sum() / denominator
    return {
        "relation_correct_absolute_bce": correct,
        "relation_query_swap_bce": shuffled,
        "relation_query_swap_margin": shuffled - correct,
    }


@torch.no_grad()
def evaluate_source(args, dataset, source_index, teacher, model, context, config):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    indices = indices[context.rank :: context.world_size]
    accumulator = Accumulator()
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        samples = [dataset[(index, 8)] for index in selected]
        batch = move_to_device(
            collate_native_video_batch_v65(samples), torch.device(context.device)
        )
        target = teacher(batch)
        output = model(batch, target)
        values = dict(output["parts"])
        if args.stage == STATE_STAGE:
            values.update(relation_swap_metrics(output, target, config))
        values["decode_replacement_fraction"] = batch["decode_replaced"].float().mean()
        values["temporal_step_seconds"] = batch[
            "temporal_step_seconds"
        ].float().mean()
        accumulator.add(values, len(selected))
    return accumulator


def aggregate_rank_payloads(local, context):
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    merged = []
    for source_index in range(len(local)):
        accumulator = Accumulator()
        for rank_payload in gathered:
            accumulator.merge(Accumulator.from_state_dict(rank_payload[source_index]))
        merged.append(accumulator)
    return merged


def decision_for_source(stage: str, metrics: dict[str, float]) -> dict[str, bool]:
    if stage == STATE_STAGE:
        return {
            "external_relation_beats_query_swap": metrics[
                "relation_query_swap_margin"
            ]
            > 0.0,
            "shared_code_has_positive_rate_saving": metrics[
                "predictive_rate_saving"
            ]
            > 0.0,
            "majority_queries_have_positive_rate_saving": metrics[
                "predictive_rate_saving_positive_fraction"
            ]
            > 0.5,
            "heldout_semantic_is_bounded": 0.5
            * (metrics["heldout_dino_error"] + metrics["heldout_siglip_error"])
            < 1.0,
        }
    correct = metrics["effect_correct_field_error"]
    denominator = max(correct, 1e-8)
    return {
        "correct_beats_persistence_by_10pct": metrics[
            "effect_gain_over_persistence"
        ]
        / max(metrics["persistence_field_error"], 1e-8)
        >= 0.10,
        "correct_beats_zero_by_10pct": metrics["effect_gain_over_zero"]
        / max(metrics["zero_effect_field_error"], 1e-8)
        >= 0.10,
        "correct_beats_shuffled_by_10pct": metrics["effect_gain_over_shuffled"]
        / max(metrics["shuffled_effect_field_error"], 1e-8)
        >= 0.10,
        "direct_and_rollout_are_bounded": metrics["goal_path_field_error"]
        < denominator,
    }


def write_wandb(args, report, global_step):
    if args.wandb_mode == "disabled":
        return ""
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=("v67", args.stage, "held-coordinate", "independent-eval"),
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "checkpoint": os.path.abspath(args.checkpoint),
            "source_revision": args.source_revision,
            "items_per_source": args.items_per_source,
            "evaluation_partition": f"held:{args.held_group_stride}",
        },
    )
    payload = {}
    for name, value in report["overall"].items():
        payload[f"eval/overall/{name}"] = value
    for source, metrics in report["per_source"].items():
        for name, value in metrics.items():
            payload[f"eval/{source}/{name}"] = value
    run.log(payload, step=global_step)
    run.summary.update(
        {
            "decision/status": report["decision"],
            "decision/passed_sources": report["passed_sources"],
            "decision/required_sources": report["required_sources"],
        }
    )
    run_id = run.id
    run.finish()
    return run_id


def main() -> None:
    args = parse_args()
    context = init_torchrun()
    require(
        context.world_size == args.expected_world_size,
        "v67 held evaluator world size differs from explicit contract",
    )
    config = ContinuousPredictiveObjectFieldConfigV67()
    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        "8",
        "100,200,400",
        0,
        17,
        group_partition="held",
        held_group_stride=args.held_group_stride,
        preserve_native_rgb=True,
    )
    device = torch.device(context.device)
    model, checkpoint = load_model(args, config, device)
    teacher = ContinuousPredictiveTeacherRuntimeV67(
        config,
        device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
    local = [
        evaluate_source(
            args, dataset, source_index, teacher, model, context, config
        ).state_dict()
        for source_index in range(len(dataset.source_names))
    ]
    merged = aggregate_rank_payloads(local, context)
    if context.is_main:
        per_source = {
            name: accumulator.means()
            for name, accumulator in zip(dataset.source_names, merged)
        }
        overall_accumulator = Accumulator()
        for accumulator in merged:
            overall_accumulator.merge(accumulator)
        overall = overall_accumulator.means()
        checks = {
            source: decision_for_source(args.stage, metrics)
            for source, metrics in per_source.items()
        }
        passed = sum(all(values.values()) for values in checks.values())
        required = max(1, len(checks) - 1)
        report = {
            "status": "completed",
            "decision": "promote" if passed >= required else "reject",
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_step": int(checkpoint["global_step"]),
            "git_commit": checkpoint["git_commit"],
            "evaluation_source_revision": args.source_revision,
            "data": os.path.abspath(args.data_index),
            "items_per_source": args.items_per_source,
            "world_size": context.world_size,
            "passed_sources": passed,
            "required_sources": required,
            "fixed_object_count": False,
            "patch_grid_is_object_state": False,
            "tracker_is_dynamic_target": False,
            "overall": overall,
            "per_source": per_source,
            "checks": checks,
        }
        run_id = write_wandb(args, report, int(checkpoint["global_step"]))
        report["wandb_run_id"] = run_id
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
