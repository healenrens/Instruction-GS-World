"""Train the v52 RGB-only Object State after objective falsification passes."""

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
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_dataset import PointTrackObjectVideoDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_teacher import FrozenPointTrackerRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_world_model_v52 import (  # noqa: E402
    LearningObjectiveObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v52_checkpointing import (  # noqa: E402
    restore_rng_state,
    validate_resume,
)
from igsw.adaptive_gaussian_wm.v52_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    LearningObjectiveObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v52_training_loop import train_v52  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data_index", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--chunk_lengths", default="8,16,24,32")
    parser.add_argument("--temporal_strides", default="1,2,3,4")
    parser.add_argument("--temporal_step_ms", default="33,67,100,133")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=128)
    parser.add_argument("--tracker_sequence_batch", type=int, default=1)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=22_500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1_125)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=2_500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def validate_arguments(args, world_size: int, config) -> None:
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError(f"v52 resume checkpoint is missing: {args.resume}")
    if args.data_index and not os.path.isfile(args.data_index):
        raise ValueError(f"v52 multisource video index is missing: {args.data_index}")
    dimensions = (
        args.batch, args.grad_accum, args.workers + 1, args.prefetch_factor,
        args.dino_frame_batch, args.tracker_sequence_batch, args.steps,
    )
    if min(dimensions) < 1:
        raise ValueError("v52 runtime dimensions must be positive")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(f"v52 effective batch {effective} != target {args.target_global_batch}")
    if args.steps != config.promotion_step:
        raise ValueError("v52 Object State training must target step 22500")
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("v52 learning-rate floor is invalid")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("v52 warmup is invalid")
    if min(args.save_every, args.recovery_every, args.log_every) < 1:
        raise ValueError("v52 checkpoint/log intervals must be positive")
    if not os.path.isfile(args.dino_checkpoint):
        raise ValueError(f"v52 frozen DINO checkpoint is missing: {args.dino_checkpoint}")
    if not os.path.isfile(args.tracker_checkpoint):
        raise ValueError(f"v52 frozen tracker checkpoint is missing: {args.tracker_checkpoint}")
    validate_wandb_arguments(args)


def read_gate(path: str, args) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"v52 startup gate is missing: {path}")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.git_commit,
        "objective_falsification_status": "passed",
        "dynamics_present": False,
        "latent_effect_present": False,
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v52 startup gate differs: {differences}")
    return report


def main() -> None:
    args = parse_args()
    for name in (
        "data", "data_index", "out", "gate_report", "dino_checkpoint", "tracker_checkpoint", "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    args.git_commit = args.source_revision
    context = init_torchrun()
    config = LearningObjectiveObjectStateConfig()
    config.validate()
    validate_arguments(args, context.world_size, config)
    gate = read_gate(args.gate_report, args)
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if args.resume else None
    )
    if checkpoint is not None:
        validate_resume(checkpoint, args, context.world_size, config)
    dataset = (
        MultiSourcePointTrackObjectVideoDataset(
            args.data_index, "train", args.chunk_lengths, args.temporal_step_ms,
            args.max_train_items, args.seed,
        )
        if args.data_index
        else PointTrackObjectVideoDataset(
            args.data, "train", args.chunk_lengths, args.temporal_strides,
            args.max_train_items, args.seed,
        )
    )
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    source_summary = {
        "source_names": list(getattr(dataset, "source_names", ("robotwin",))),
        "source_episode_counts": list(getattr(dataset, "source_episode_counts", ())),
        "source_task_counts": list(getattr(dataset, "source_task_counts", ())),
        "source_target_samples": list(getattr(dataset, "source_target_samples", ())),
    }
    model = LearningObjectiveObjectWorldModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[context.local_rank], broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if context.distributed else model
    )
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, args.tracker_sequence_batch
    )
    sampler = build_training_sampler(
        dataset, context.world_size, context.rank, args.seed, args.batch, args.grad_accum
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor / args.lr
    )
    start_step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint, context)
        start_step = int(checkpoint["global_step"])
    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        contract_path = os.path.join(args.out, "run_contract.json")
        if checkpoint is None and (
            os.path.lexists(os.path.join(args.out, "latest.pt"))
            or os.path.isfile(contract_path)
        ):
            raise ValueError(f"v52 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": "object_state",
            "git_commit": args.git_commit,
            "historical_checkpoint_used": False,
            "object_target": "persistent_compositional_relation_constrained_visual_entity",
            "point_tracker_role": "training_only_correspondence_and_relation_evidence",
            "hard_component_pseudo_labels": False,
            "dynamics_present": False,
            "latent_effect_present": False,
            "dino_fully_frozen": True,
            "multisource_video_index": args.data_index,
            **source_summary,
            "config": config.to_dict(),
            "args": vars(args),
            "startup_gate": gate,
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
            "stage": "object_state",
            "git_commit": args.git_commit,
            "historical_checkpoint_used": False,
            "hard_component_pseudo_labels": False,
            "multisource_video_index": args.data_index,
            **source_summary,
            **config.to_dict(),
            **vars(args),
        },
    )
    if context.is_main:
        print(json.dumps({
            "event": "v52_start",
            "global_step": start_step,
            "examples": len(dataset),
            **source_summary,
            "world_size": context.world_size,
            "micro_batch": args.batch,
            "grad_accum": args.grad_accum,
            "effective_batch": args.target_global_batch,
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
        }, sort_keys=True), flush=True)
    final_step = train_v52(
        model, wrapped, dino, tracker, loader, sampler, optimizer, scheduler,
        context, args, start_step, wandb_tracker,
    )
    if wandb_tracker is not None:
        wandb_tracker.finish()
    if context.is_main:
        print(json.dumps({
            "event": "v52_complete", "stage": "object_state", "global_step": final_step
        }), flush=True)
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
