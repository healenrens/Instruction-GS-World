"""Train observation-complete object state and latent effects with strict resume."""

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
    add_wandb_arguments, init_wandb_tracker, validate_wandb_arguments,
)
from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.group_balanced_sampler import build_training_sampler  # noqa: E402
from igsw.adaptive_gaussian_wm.observation_complete_world_model import (  # noqa: E402
    ObservationCompleteWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import TemporalObjectVideoDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v47_checkpointing import (  # noqa: E402
    restore_rng_state, validate_resume,
)
from igsw.adaptive_gaussian_wm.v47_config import (  # noqa: E402
    ARCHITECTURE, CHECKPOINT_VERSION, ObservationCompleteConfig,
)
from igsw.adaptive_gaussian_wm.v47_training_loop import train_v47  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--chunk_lengths", default="8,16,24,32")
    parser.add_argument("--temporal_strides", default="1,2,3,4")
    parser.add_argument("--observation_mask_probability", type=float, default=0.20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=128)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--state_phase_steps", type=int, default=15_000)
    parser.add_argument("--goal_phase_steps", type=int, default=30_000)
    parser.add_argument("--curriculum_ramp_steps", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2_500)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=2_500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def _validate_arguments(args, world_size: int) -> None:
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError(f"v47 resume checkpoint is missing: {args.resume}")
    if min(args.batch, args.grad_accum, args.workers + 1, args.prefetch_factor, args.dino_frame_batch) < 1:
        raise ValueError("v47 batch, accumulation, workers and DINO batch are invalid")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(f"v47 effective batch {effective} != target {args.target_global_batch}")
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("v47 learning-rate floor is invalid")
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("v47 step counts are invalid")
    if min(args.save_every, args.recovery_every, args.log_every) < 1:
        raise ValueError("v47 save/recovery/log intervals must be positive")
    if not args.source_revision:
        raise ValueError("v47 source revision is empty")
    if not os.path.isfile(args.dino_checkpoint):
        raise ValueError(f"v47 local DINO checkpoint is missing: {args.dino_checkpoint}")
    validate_wandb_arguments(args)


def _validate_gate(args) -> dict:
    with open(args.gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed", "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE, "git_commit": args.git_commit,
    }
    differences = {
        name: (report.get(name), value) for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v47 gate report differs from this run: {differences}")
    return report


def main() -> None:
    args = parse_args()
    args.data = os.path.abspath(args.data)
    args.out = os.path.abspath(args.out)
    args.gate_report = os.path.abspath(args.gate_report)
    args.dino_checkpoint = os.path.abspath(args.dino_checkpoint)
    args.git_commit = args.source_revision
    context = init_torchrun()
    _validate_arguments(args, context.world_size)
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    dataset = TemporalObjectVideoDataset(
        args.data, "train", args.chunk_lengths, args.temporal_strides,
        args.observation_mask_probability, args.max_train_items, args.seed, False,
    )
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    gate = _validate_gate(args)
    config = ObservationCompleteConfig(
        state_phase_steps=args.state_phase_steps,
        goal_phase_steps=args.goal_phase_steps,
        total_steps=args.steps,
        curriculum_ramp_steps=args.curriculum_ramp_steps,
    )
    config.validate()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        validate_resume(checkpoint, args, context.world_size, config)
    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        existing_contract = os.path.join(args.out, "run_contract.json")
        if checkpoint is None and (
            os.path.lexists(os.path.join(args.out, "latest.pt"))
            or os.path.isfile(existing_contract)
        ):
            raise ValueError(f"v47 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "git_commit": args.git_commit,
            "gate_report": args.gate_report,
            "historical_checkpoint_used": False,
            "teacher_sidecar_used": False,
            "language_used": False,
            "explicit_action_used": False,
            "config": config.to_dict(), "args": vars(args), "gate": gate,
        }
        with open(existing_contract, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if context.distributed:
        dist.barrier()
    model = ObservationCompleteWorldModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    wrapped = DistributedDataParallel(
        model, device_ids=[context.local_rank], broadcast_buffers=False,
        find_unused_parameters=False,
    ) if context.distributed else model
    encoder = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    sampler = build_training_sampler(
        dataset, context.world_size, context.rank, args.seed, args.batch, args.grad_accum
    )
    loader_options = {
        "dataset": dataset, "batch_size": args.batch, "sampler": sampler,
        "num_workers": args.workers, "pin_memory": True, "drop_last": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)
    optimizer = torch.optim.AdamW([
        {"params": model.state_encoder.parameters(), "lr": args.lr, "group_name": "state"},
        {
            "params": list(model.effect_posterior.parameters()) + list(model.dynamics.parameters()),
            "lr": args.lr, "group_name": "effect",
        },
        {"params": model.goal_effect_predictor.parameters(), "lr": args.lr, "group_name": "goal"},
    ], weight_decay=args.weight_decay)
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor / args.lr
    )
    start_step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint, context)
        start_step = int(checkpoint["global_step"])
    tracker = init_wandb_tracker(args, context, {
        "checkpoint_version": CHECKPOINT_VERSION, "architecture": ARCHITECTURE,
        "git_commit": args.git_commit, "historical_checkpoint_used": False,
        **config.to_dict(), **vars(args),
    })
    if context.is_main:
        print(json.dumps({
            "event": "v47_start", "global_step": start_step,
            "examples": len(dataset), "world_size": context.world_size,
            "micro_batch": args.batch, "grad_accum": args.grad_accum,
            "dino_frame_batch": args.dino_frame_batch, "workers_per_rank": args.workers,
            "dino_checkpoint": args.dino_checkpoint,
            "effective_batch": args.target_global_batch,
            "checkpoint_version": CHECKPOINT_VERSION, "architecture": ARCHITECTURE,
        }, sort_keys=True), flush=True)
    final_step = train_v47(
        model, wrapped, encoder, loader, sampler, optimizer, scheduler,
        context, args, start_step, tracker,
    )
    if tracker is not None:
        tracker.finish()
    if context.is_main:
        print(json.dumps({"event": "v47_complete", "global_step": final_step}), flush=True)
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
