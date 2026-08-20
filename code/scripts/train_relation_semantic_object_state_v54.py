"""Train v54 RGB-only Object State on the full multisource video index."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.experiment_tracking import (  # noqa: E402
    add_wandb_arguments,
    init_wandb_tracker,
    validate_wandb_arguments,
)
from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.group_balanced_sampler import build_training_sampler  # noqa: E402
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import FrozenPointTrackerRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.relation_semantic_object_state_v54 import (  # noqa: E402
    RelationSemanticObjectStateModel,
)
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v54_checkpointing import (  # noqa: E402
    restore_rng_state,
    validate_resume,
)
from igsw.adaptive_gaussian_wm.v54_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STAGE,
    RelationSemanticObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v54_training_loop import train_v54  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--chunk_lengths", default="3,4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=512)
    parser.add_argument("--tracker_batch_per_rank", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=192)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1_500)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=2_500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def validate_arguments(args, world_size: int) -> None:
    for name in (
        "data_index", "gate_report", "decode_report",
        "dino_checkpoint", "tracker_checkpoint",
    ):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v54 {name} is missing: {getattr(args, name)}")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError(f"v54 resume checkpoint is missing: {args.resume}")
    dimensions = (
        args.batch, args.grad_accum, args.tracker_batch_per_rank,
        args.workers + 1, args.prefetch_factor, args.dino_frame_batch, args.steps,
    )
    if min(dimensions) < 1:
        raise ValueError("v54 runtime dimensions must be positive")
    if args.tracker_batch_per_rank > args.batch:
        raise ValueError("v54 tracker subset exceeds local batch")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(
            f"v54 effective batch {effective} != target {args.target_global_batch}"
        )
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("v54 learning-rate floor is invalid")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("v54 warmup is invalid")
    if min(args.save_every, args.recovery_every, args.log_every) < 1:
        raise ValueError("v54 checkpoint and logging intervals must be positive")
    validate_wandb_arguments(args)


def read_gate(args) -> dict:
    with open(args.gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.git_commit,
        "historical_checkpoint_used": False,
        "dynamics_present": False,
        "latent_effect_present": False,
        "point_tracker_in_deployable_model": False,
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v54 startup gate differs: {differences}")
    return report


def main() -> None:
    args = parse_args()
    for name in (
        "data_index", "out", "gate_report", "decode_report",
        "dino_checkpoint", "tracker_checkpoint", "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    args.git_commit = args.source_revision
    context = init_torchrun()
    config = RelationSemanticObjectStateConfig()
    config.validate()
    validate_arguments(args, context.world_size)
    gate = read_gate(args)
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if args.resume else None
    )
    if checkpoint is not None:
        validate_resume(checkpoint, args, context.world_size, config)

    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        args.max_train_items,
        args.seed,
    )
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    model = RelationSemanticObjectStateModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if context.distributed else model
    )
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    point_tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    sampler = build_training_sampler(
        dataset, context.world_size, context.rank, args.seed,
        args.batch, args.grad_accum,
    )
    loader_options = {
        "dataset": dataset,
        "batch_size": args.batch,
        "sampler": sampler,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor / args.lr
    )
    start_step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint, context)
        start_step = int(checkpoint["global_step"])

    source_summary = {
        "source_names": list(dataset.source_names),
        "source_episode_counts": list(dataset.source_episode_counts),
        "source_task_counts": list(dataset.source_task_counts),
        "source_target_samples": list(dataset.source_target_samples),
        "runtime_missing_video_count": dataset.runtime_missing_video_count,
    }
    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        contract_path = os.path.join(args.out, "run_contract.json")
        if not args.resume and (
            os.path.lexists(os.path.join(args.out, "latest.pt"))
            or os.path.isfile(contract_path)
        ):
            raise ValueError(f"v54 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": STAGE,
            "git_commit": args.git_commit,
            "historical_checkpoint_used": False,
            "object_target": "persistent_relation_anchored_semantic_visual_entity",
            "point_tracker_role": "training_only_rotating_relation_teacher",
            "point_tracker_in_deployable_model": False,
            "hard_component_pseudo_labels": False,
            "dynamics_present": False,
            "latent_effect_present": False,
            "dino_fully_frozen": True,
            "dense_affinity_objective": False,
            "multisource_video_index": args.data_index,
            "config": config.to_dict(),
            "args": vars(args),
            "startup_gate": gate,
            **source_summary,
        }
        with open(contract_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if context.distributed:
        dist.barrier()
    wandb_tracker = init_wandb_tracker(
        args,
        context,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": STAGE,
            "git_commit": args.git_commit,
            "historical_checkpoint_used": False,
            "hard_component_pseudo_labels": False,
            "dense_affinity_objective": False,
            **source_summary,
            **config.to_dict(),
            **vars(args),
        },
    )
    if context.is_main:
        print(json.dumps({
            "event": "v54_start",
            "global_step": start_step,
            "examples": len(dataset),
            "world_size": context.world_size,
            "micro_batch": args.batch,
            "grad_accum": args.grad_accum,
            "effective_batch": args.target_global_batch,
            "tracker_batch_per_rank": args.tracker_batch_per_rank,
            **source_summary,
        }, sort_keys=True), flush=True)
    final_step = train_v54(
        model, wrapped, dino, point_tracker, loader, sampler, optimizer, scheduler,
        context, args, start_step, wandb_tracker,
    )
    if wandb_tracker is not None:
        wandb_tracker.finish()
    if context.is_main:
        print(json.dumps({
            "event": "v54_complete", "stage": STAGE, "global_step": final_step
        }), flush=True)
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
