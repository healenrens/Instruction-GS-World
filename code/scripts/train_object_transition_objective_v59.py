#!/usr/bin/env python3
"""Train the isolated latent-effect object-transition objective."""

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
from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.latent_object_transition_v59 import (  # noqa: E402
    QueryObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v56_data_contract import (  # noqa: E402
    audit_decode_frontier,
)
from igsw.adaptive_gaussian_wm.v59_checkpointing import (  # noqa: E402
    load_v58_encoder,
    restore_rng_state,
    validate_resume,
)
from igsw.adaptive_gaussian_wm.v59_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STAGE,
    ObjectTransitionConfig,
)
from igsw.adaptive_gaussian_wm.v59_training_loop import train_v59  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--startup_gate", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--history_lengths", default="1,2,3,4")
    parser.add_argument("--teacher_future_frames", type=int, default=4)
    parser.add_argument("--chunk_lengths", default="5,6,7,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400,800")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=192)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--run_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--recovery_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def parse_lengths(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or tuple(sorted(set(values))) != values or min(values) < 1:
        raise ValueError("v59 lengths must be unique, increasing, and positive")
    return values


def validate_arguments(args, world_size, config):
    for name in (
        "data_index",
        "startup_gate",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
        "init_from",
    ):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v59 {name} is missing: {getattr(args, name)}")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError(f"v59 resume checkpoint is missing: {args.resume}")
    dimensions = (
        args.batch,
        args.grad_accum,
        args.workers + 1,
        args.prefetch_factor,
        args.dino_frame_batch,
        args.steps,
        args.teacher_future_frames,
    )
    if min(dimensions) < 1 or args.run_steps < 0:
        raise ValueError("v59 runtime dimensions are invalid")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(
            f"v59 effective batch {effective} != {args.target_global_batch}"
        )
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("v59 learning-rate floor is invalid")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("v59 warmup is invalid")
    if min(args.save_every, args.recovery_every, args.log_every) < 1:
        raise ValueError("v59 checkpoint and logging intervals must be positive")
    if args.teacher_future_frames != config.teacher_future_frames:
        raise ValueError("v59 teacher future differs from model config")
    validate_wandb_arguments(args)


def read_startup_gate(args):
    with open(args.startup_gate, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.git_commit,
        "data_index": args.data_index,
        "decode_report": args.decode_report,
        "dino_checkpoint": args.dino_checkpoint,
        "tracker_checkpoint": args.tracker_checkpoint,
        "init_from": args.init_from,
        "history_lengths": args.history_lengths,
        "teacher_future_frames": args.teacher_future_frames,
        "temporal_step_ms": args.temporal_step_ms,
        "encoder_frozen": True,
        "point_velocity_is_core_objective": False,
        "latent_effect_present": True,
        "effect_conditioned_dynamics_present": True,
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v59 startup gate differs: {differences}")
    return report


def main():
    args = parse_args()
    for name in (
        "data_index",
        "out",
        "startup_gate",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
        "init_from",
        "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    args.git_commit = args.source_revision
    args.parsed_history_lengths = parse_lengths(args.history_lengths)
    expected_chunks = tuple(
        history + args.teacher_future_frames for history in args.parsed_history_lengths
    )
    if parse_lengths(args.chunk_lengths) != expected_chunks:
        raise ValueError("v59 chunks must equal history plus teacher future")
    context = init_torchrun()
    config = ObjectTransitionConfig(teacher_future_frames=args.teacher_future_frames)
    config.validate()
    validate_arguments(args, context.world_size, config)
    startup_gate = read_startup_gate(args)
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if args.resume
        else None
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
    decode = audit_decode_frontier(
        args.decode_report, args.data_index, args.seed, dataset.source_names
    )
    if decode["status"] != "passed":
        raise ValueError(f"v59 decode frontier differs: {decode}")
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    model = QueryObjectTransitionModel(config).to(device)
    if checkpoint is None:
        init_report = load_v58_encoder(model, args.init_from)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
        init_report = checkpoint["v58_initialization"]
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
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
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
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    }
    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        contract_path = os.path.join(args.out, "run_contract.json")
        if not args.resume and (
            os.path.lexists(os.path.join(args.out, "latest.pt"))
            or os.path.isfile(contract_path)
        ):
            raise ValueError(f"v59 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": STAGE,
            "git_commit": args.git_commit,
            "state_encoder_frozen": True,
            "point_velocity_is_core_objective": False,
            "target": "track_relation_object_semantic_geometry_lifecycle",
            "config": config.to_dict(),
            "args": vars(args),
            "startup_gate": startup_gate,
            "v58_initialization": init_report,
            **source_summary,
        }
        if not args.resume:
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
            **source_summary,
            **config.to_dict(),
            **vars(args),
        },
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "event": "v59_start",
                    "global_step": start_step,
                    "world_size": context.world_size,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.target_global_batch,
                    "v58_initialization": init_report,
                    **source_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    final_step = train_v59(
        model,
        wrapped,
        dino,
        tracker,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        start_step,
        wandb_tracker,
        init_report,
    )
    if wandb_tracker is not None:
        wandb_tracker.finish()
    if context.is_main:
        print(json.dumps({"event": "v59_complete", "global_step": final_step}))


if __name__ == "__main__":
    main()
