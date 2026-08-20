"""Train v53 from full multisource videos in tokenizer or dynamics stage."""

from __future__ import annotations

import argparse
import json
import math
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
from igsw.adaptive_gaussian_wm.video_file_decoder import (  # noqa: E402
    VIDEO_DECODER_CONTRACT,
)
from igsw.adaptive_gaussian_wm.semantic_object_world_model_v53 import (  # noqa: E402
    SemanticObjectLatentWorldModel,
)
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v53_checkpointing import (  # noqa: E402
    restore_rng_state,
    validate_init_from,
    validate_resume,
)
from igsw.adaptive_gaussian_wm.v53_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    DECODER_CONTRACT,
    STAGES,
    SemanticObjectWorldModelConfig,
)
from igsw.adaptive_gaussian_wm.v53_training_loop import train_v53  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--init_from", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--chunk_lengths", default="3,4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=96)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--posterior_lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=2_500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def validate_arguments(args, world_size: int) -> None:
    for name in ("data_index", "gate_report", "decode_report", "dino_checkpoint"):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v53 {name} is missing: {getattr(args, name)}")
    if args.resume and args.init_from:
        raise ValueError("v53 cannot combine resume and init_from")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError(f"v53 resume checkpoint is missing: {args.resume}")
    if args.init_from and not os.path.isfile(args.init_from):
        raise ValueError(f"v53 initialization checkpoint is missing: {args.init_from}")
    if args.stage == "tokenizer" and args.init_from:
        raise ValueError(
            "v53 tokenizer stage must start without historical initialization"
        )
    if args.stage == "dynamics" and not (args.init_from or args.resume):
        raise ValueError("v53 dynamics needs tokenizer init_from or a dynamics resume")
    dimensions = (
        args.batch,
        args.grad_accum,
        args.workers + 1,
        args.prefetch_factor,
        args.dino_frame_batch,
        args.steps,
    )
    if min(dimensions) < 1:
        raise ValueError("v53 runtime dimensions must be positive")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(
            f"v53 effective batch {effective} != target {args.target_global_batch}"
        )
    if not 0.0 < args.lr_floor <= min(args.lr, args.posterior_lr):
        raise ValueError("v53 learning-rate floor is invalid")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("v53 warmup is invalid")
    if min(args.save_every, args.recovery_every, args.log_every) < 1:
        raise ValueError("v53 checkpoint and logging intervals must be positive")
    validate_wandb_arguments(args)
    with open(args.decode_report, encoding="utf-8") as handle:
        decode_report = json.load(handle)
    expected_decode = {
        "status": "passed",
        "contract": "multisource_v53_distributed_decode_frontier_v1",
        "decoder_contract": VIDEO_DECODER_CONTRACT,
        "data_index": args.data_index,
        "sampler_epoch": args.seed,
    }
    differences = {
        name: (decode_report.get(name), value)
        for name, value in expected_decode.items()
        if decode_report.get(name) != value
    }
    if differences:
        raise ValueError(f"v53 decode frontier differs: {differences}")


def read_gate(args) -> dict:
    with open(args.gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "decoder_contract": DECODER_CONTRACT,
        "git_commit": args.git_commit,
        "stage": args.stage,
        "point_tracker_used": False,
        "rgb_reconstruction_used": False,
        "numerical_stability_status": "passed",
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v53 startup gate differs: {differences}")
    sources = report.get("source_names", [])
    losses = report.get("numerical_stability_source_losses", {})
    if set(losses) != set(sources) | {"mixed"}:
        raise ValueError("v53 startup gate did not test every source and a mixed batch")
    if report.get("numerical_stability_updates", 0) < len(sources) + 1:
        raise ValueError("v53 startup gate ran too few numerical stability updates")
    if report.get("numerical_stability_maximum_micro_batch", 0) < args.batch:
        raise ValueError("v53 startup gate did not cover the training micro-batch")
    scalar_health = (
        *losses.values(),
        report.get("maximum_parameter_gradient", float("nan")),
        report.get("coordinate_basis_weight_maximum_gradient", float("nan")),
    )
    if not all(math.isfinite(value) for value in scalar_health):
        raise ValueError("v53 startup gate contains non-finite numerical health")
    return report


def _optimizer(model, args):
    if args.stage == "tokenizer":
        return torch.optim.AdamW(
            [
                {
                    "params": model.tokenizer.parameters(),
                    "lr": args.lr,
                    "group_name": "tokenizer",
                }
            ],
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        [
            {
                "params": model.dynamics.parameters(),
                "lr": args.lr,
                "group_name": "dynamics",
            },
            {
                "params": model.posterior.parameters(),
                "lr": args.posterior_lr,
                "group_name": "posterior",
            },
        ],
        weight_decay=args.weight_decay,
    )


def main() -> None:
    args = parse_args()
    for name in (
        "data_index",
        "out",
        "gate_report",
        "decode_report",
        "dino_checkpoint",
        "init_from",
        "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    args.git_commit = args.source_revision
    context = init_torchrun()
    validate_arguments(args, context.world_size)
    config = SemanticObjectWorldModelConfig()
    config.validate()
    gate = read_gate(args)
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False, mmap=True
        )
        validate_resume(checkpoint, args, context.world_size, config)
    elif args.init_from:
        checkpoint = torch.load(
            args.init_from, map_location="cpu", weights_only=False, mmap=True
        )
        validate_init_from(checkpoint, config)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        args.max_train_items,
        args.seed,
    )
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    model = SemanticObjectLatentWorldModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.configure_stage(args.stage)
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
    optimizer = _optimizer(model, args)
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor / args.lr
    )
    start_step = 0
    if args.resume:
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
            raise ValueError(f"v53 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "git_commit": args.git_commit,
            "historical_checkpoint_used": False,
            "frozen_perception_encoder": True,
            "point_tracker_used": False,
            "rgb_reconstruction_used": False,
            "instance_segmentation_used": False,
            "language_used": False,
            "explicit_action_used": False,
            "multisource_video_index": args.data_index,
            "init_from": args.init_from,
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
    tracker = init_wandb_tracker(
        args,
        context,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "git_commit": args.git_commit,
            "historical_checkpoint_used": False,
            "point_tracker_used": False,
            "rgb_reconstruction_used": False,
            **source_summary,
            **config.to_dict(),
            **vars(args),
        },
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "event": "v53_start",
                    "stage": args.stage,
                    "global_step": start_step,
                    "examples": len(dataset),
                    "world_size": context.world_size,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.target_global_batch,
                    **source_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    final_step = train_v53(
        model,
        wrapped,
        dino,
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
        print(
            json.dumps(
                {
                    "event": "v53_complete",
                    "stage": args.stage,
                    "global_step": final_step,
                }
            ),
            flush=True,
        )
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
