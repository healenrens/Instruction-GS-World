"""Train one of the four v61 single-Student Object State comparisons."""

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

from igsw.adaptive_gaussian_wm.carrier_teacher_v61 import (  # noqa: E402
    FrozenSiglipObjectTeacherV61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_world_model_v61 import (  # noqa: E402
    ContinuousCarrierObjectWorldModelV61,
)
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
from igsw.adaptive_gaussian_wm.v61_checkpointing import (  # noqa: E402
    restore_rng_state,
    validate_resume,
)
from igsw.adaptive_gaussian_wm.v61_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STAGE,
    VARIANTS,
    config_for_variant,
)
from igsw.adaptive_gaussian_wm.v61_training_loop import train_v61  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--chunk_lengths", default="4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--student_frame_batch", type=int, default=32)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--siglip_teacher_batch", type=int, default=64)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--backbone_lr", type=float, default=2e-6)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor_ratio", type=float, default=0.10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=2_500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def validate_arguments(args, world_size: int, config) -> None:
    required = (
        "data_index",
        "gate_report",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
    )
    for name in required:
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v61 {name} is missing: {getattr(args, name)}")
    if config.student_encoder == "siglip" or config.uses_object_semantics:
        if not os.path.isdir(args.siglip_checkpoint):
            raise ValueError("v61 SigLIP checkpoint must be a local directory")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError(f"v61 resume checkpoint is missing: {args.resume}")
    dimensions = (
        args.batch,
        args.grad_accum,
        args.workers + 1,
        args.prefetch_factor,
        args.student_frame_batch,
        args.dino_frame_batch,
        args.siglip_teacher_batch,
        args.steps,
    )
    if min(dimensions) < 1:
        raise ValueError("v61 runtime dimensions must be positive")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(
            f"v61 effective batch {effective} != target {args.target_global_batch}"
        )
    if not 0.0 < args.backbone_lr <= args.head_lr:
        raise ValueError("v61 backbone LR must be positive and no larger than head LR")
    if not 0.0 < args.lr_floor_ratio <= 1.0:
        raise ValueError("v61 LR floor ratio must stay within (0,1]")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("v61 warmup is invalid")
    validate_wandb_arguments(args)


def read_gate(args, config) -> dict:
    with open(args.gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STAGE,
        "variant": args.variant,
        "git_commit": args.git_commit,
        "student_encoder": config.student_encoder,
        "single_student_encoder": True,
        "dynamics_present": False,
        "point_tracker_in_deployable_model": False,
        "held_group_stride": args.held_group_stride,
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v61 startup report differs: {differences}")
    return report


def build_optimizer(model, args):
    backbone = [
        parameter
        for parameter in model.student.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone}
    heads = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone or not heads:
        raise ValueError(
            "v61 optimizer requires trainable backbone and head parameters"
        )
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr, "group_name": "backbone"},
            {"params": heads, "lr": args.head_lr, "group_name": "heads"},
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
        "tracker_checkpoint",
        "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    args.siglip_checkpoint = os.path.abspath(args.siglip_checkpoint)
    args.git_commit = args.source_revision
    context = init_torchrun()
    config = config_for_variant(args.variant)
    validate_arguments(args, context.world_size, config)
    gate = read_gate(args, config)
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
        group_partition="train",
        held_group_stride=args.held_group_stride,
    )
    decode_audit = audit_decode_frontier(
        args.decode_report, args.data_index, args.seed, dataset.source_names
    )
    if decode_audit["status"] != "passed":
        raise ValueError(f"v61 decode frontier differs: {decode_audit}")
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    model = ContinuousCarrierObjectWorldModelV61(
        config,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.student_frame_batch,
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
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
    dino_teacher = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    point_tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    semantic_teacher = (
        FrozenSiglipObjectTeacherV61(
            args.siglip_checkpoint, device, args.siglip_teacher_batch
        )
        if config.uses_object_semantics
        else None
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
    optimizer = build_optimizer(model, args)
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor_ratio
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
        if not args.resume and os.path.isfile(contract_path):
            raise ValueError(f"v61 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": STAGE,
            "variant": args.variant,
            "git_commit": args.git_commit,
            "single_student_encoder": True,
            "student_encoder": config.student_encoder,
            "training_only_teachers": ["DINOv2-L", "CoTracker", "SigLIP crops"]
            if config.uses_object_semantics
            else ["DINOv2-L", "CoTracker"],
            "dynamics_present": False,
            "historical_checkpoint_used": False,
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
            "variant": args.variant,
            "single_student_encoder": True,
            "student_encoder": config.student_encoder,
            **source_summary,
            **config.to_dict(),
            **vars(args),
        },
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "event": "v61_start",
                    "global_step": start_step,
                    "variant": args.variant,
                    "student_encoder": config.student_encoder,
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
    final_step = train_v61(
        model,
        wrapped,
        dino_teacher,
        point_tracker,
        semantic_teacher,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        start_step,
        wandb_tracker,
    )
    if wandb_tracker is not None:
        wandb_tracker.finish()
    if context.is_main:
        print(
            json.dumps({"event": "v61_complete", "global_step": final_step}), flush=True
        )
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
