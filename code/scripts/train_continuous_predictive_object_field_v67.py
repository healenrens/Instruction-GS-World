"""Train v67 predictive state or posterior-conditioned object field Dynamics."""

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

from igsw.adaptive_gaussian_wm.continuous_predictive_object_field_v67 import (  # noqa: E402
    ContinuousPredictiveObjectFieldV67,
)
from igsw.adaptive_gaussian_wm.continuous_predictive_teacher_v67 import (  # noqa: E402
    ContinuousPredictiveTeacherRuntimeV67,
)
from igsw.adaptive_gaussian_wm.experiment_tracking import (  # noqa: E402
    add_wandb_arguments,
    init_wandb_tracker,
    validate_wandb_arguments,
)
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.native_video_batch_v65 import (  # noqa: E402
    collate_native_video_batch_v65,
)
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v67_checkpointing import (  # noqa: E402
    load_predictive_state_checkpoint_v67,
    restore_rng_state_v67,
    validate_resume_v67,
)
from igsw.adaptive_gaussian_wm.v67_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    DYNAMICS_STAGE,
    STAGES,
    ContinuousPredictiveObjectFieldConfigV67,
)
from igsw.adaptive_gaussian_wm.v67_training_loop import train_v67  # noqa: E402
from igsw.distributed import init_torchrun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--state_checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_compatible_source_revision", default="")
    parser.add_argument("--chunk_lengths", default="8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=192)
    parser.add_argument("--siglip_frame_batch", type=int, default=192)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor_ratio", type=float, default=0.10)
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


def validate_arguments(args, world_size):
    for name in ("data_index", "dino_checkpoint", "tracker_checkpoint"):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v67 {name} is missing: {getattr(args, name)}")
    if not os.path.isdir(args.siglip_checkpoint):
        raise ValueError("v67 SigLIP checkpoint must be a local directory")
    if args.stage == DYNAMICS_STAGE and not args.resume:
        if not os.path.isfile(args.state_checkpoint):
            raise ValueError("v67 Dynamics requires a predictive-state checkpoint")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError("v67 resume checkpoint is missing")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(
            f"v67 effective batch {effective} != target {args.target_global_batch}"
        )
    if args.chunk_lengths != "8" or args.temporal_step_ms != "100,200,400":
        raise ValueError(
            "v67 temporal contract is eight frames at 100/200/400 ms"
        )
    validate_wandb_arguments(args)


def main():
    args = parse_args()
    for name in (
        "data_index",
        "out",
        "dino_checkpoint",
        "siglip_checkpoint",
        "tracker_checkpoint",
        "state_checkpoint",
        "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    context = init_torchrun()
    validate_arguments(args, context.world_size)
    if not args.resume and os.path.lexists(os.path.join(args.out, "latest.pt")):
        raise ValueError(
            f"v67 output already has a checkpoint; use strict resume: {args.out}"
        )
    config = ContinuousPredictiveObjectFieldConfigV67()
    config.validate()
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    device = torch.device(context.device)
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if args.resume
        else None
    )
    if resume_checkpoint is not None:
        validate_resume_v67(resume_checkpoint, args, context, config)

    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        args.max_train_items,
        args.seed,
        group_partition="train",
        held_group_stride=args.held_group_stride,
        preserve_native_rgb=True,
    )
    model = ContinuousPredictiveObjectFieldV67(config, args.stage)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
    elif args.stage == DYNAMICS_STAGE:
        state_checkpoint = load_predictive_state_checkpoint_v67(
            args.state_checkpoint, config
        )
        model.load_state_dict(state_checkpoint["model"], strict=True)
        model.configure_stage(DYNAMICS_STAGE)
    model = model.to(device)
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if context.distributed
        else model
    )
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
    sampler = build_training_sampler(
        dataset,
        context.world_size,
        context.rank,
        args.seed,
        args.batch,
        args.grad_accum,
    )
    loader_options = {
        "dataset": dataset,
        "batch_size": args.batch,
        "sampler": sampler,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": args.workers > 0,
        "collate_fn": collate_native_video_batch_v65,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer.param_groups[0]["group_name"] = args.stage
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor_ratio
    )
    start_step = 0
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        restore_rng_state_v67(resume_checkpoint, context)
        start_step = int(resume_checkpoint["global_step"])

    source_summary = {
        "source_names": list(dataset.source_names),
        "source_episode_counts": list(dataset.source_episode_counts),
        "source_task_counts": list(dataset.source_task_counts),
        "source_target_samples": list(dataset.source_target_samples),
    }
    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "git_commit": args.source_revision,
            "historical_checkpoint_used": False,
            "state_checkpoint": args.state_checkpoint,
            "training_only_teachers": ["DINOv2-L", "SigLIP", "CoTracker"],
            "object_representation": "continuous_query_function",
            "tracker_is_dynamic_target": False,
            "fixed_object_count": False,
            "rgb_reconstruction": False,
            "config": config.to_dict(),
            "args": vars(args),
            **source_summary,
        }
        with open(
            os.path.join(args.out, "run_contract.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if context.distributed:
        dist.barrier()
    tracker = init_wandb_tracker(
        args,
        context,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            **config.to_dict(),
            **source_summary,
            **vars(args),
        },
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "event": "v67_start",
                    "stage": args.stage,
                    "global_step": start_step,
                    "world_size": context.world_size,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.target_global_batch,
                    "examples": len(dataset),
                    **source_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    final_step = train_v67(
        model,
        wrapped,
        teacher,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        start_step,
        tracker,
    )
    if tracker is not None:
        tracker.finish()
    if context.is_main:
        print(json.dumps({"event": "v67_complete", "global_step": final_step}))
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
