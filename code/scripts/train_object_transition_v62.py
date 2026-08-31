"""Train v62 E0 object codec or E1 deterministic teacher-state oracle."""

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
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_teacher_runtime_v62 import (  # noqa: E402
    ObjectTransitionTeacherRuntimeV62,
)
from igsw.adaptive_gaussian_wm.teacher_object_autoencoder_v62 import (  # noqa: E402
    TeacherObjectAutoencoderV62,
)
from igsw.adaptive_gaussian_wm.teacher_transition_oracle_v62 import (  # noqa: E402
    TeacherTransitionOracleV62,
)
from igsw.adaptive_gaussian_wm.train_runtime import cosine_schedule  # noqa: E402
from igsw.adaptive_gaussian_wm.v56_data_contract import (  # noqa: E402
    audit_decode_frontier,
)
from igsw.adaptive_gaussian_wm.v62_checkpointing import (  # noqa: E402
    load_e0_codec_checkpoint_v62,
    restore_rng_state_v62,
    validate_resume_v62,
)
from igsw.adaptive_gaussian_wm.v62_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    E0_STAGE,
    E1_STAGE,
    STAGES,
    ObjectTransitionConfigV62,
)
from igsw.adaptive_gaussian_wm.v62_training_loop import train_v62  # noqa: E402
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--codec_checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_compatible_source_revision", default="")
    parser.add_argument("--chunk_lengths", default="3")
    parser.add_argument("--temporal_step_ms", default="100")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=128)
    parser.add_argument("--siglip_frame_batch", type=int, default=128)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=2e-4)
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


def validate_arguments(args, world_size):
    for name in (
        "data_index",
        "gate_report",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
    ):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError(f"v62 {name} is missing: {getattr(args, name)}")
    if not os.path.isdir(args.siglip_checkpoint):
        raise ValueError("v62 SigLIP checkpoint must be a local directory")
    if args.stage == E1_STAGE and not os.path.isfile(args.codec_checkpoint):
        raise ValueError("v62 E1 requires an E0 codec checkpoint")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError("v62 resume checkpoint is missing")
    effective = args.batch * args.grad_accum * world_size
    if effective != args.target_global_batch:
        raise ValueError(
            f"v62 effective batch {effective} != target {args.target_global_batch}"
        )
    if args.chunk_lengths != "3" or args.temporal_step_ms != "100":
        raise ValueError("v62 E0/E1 contract is exactly three frames at 100 ms")
    validate_wandb_arguments(args)


def read_gate(args):
    with open(args.gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "git_commit": args.source_revision,
        "historical_checkpoint_used": False,
        "patch_grid_is_object_target": False,
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    if differences:
        raise ValueError(f"v62 gate differs: {differences}")
    return report


def build_model(args, config):
    if args.stage == E0_STAGE:
        return TeacherObjectAutoencoderV62(config), None
    codec_checkpoint = load_e0_codec_checkpoint_v62(args.codec_checkpoint, config)
    model = TeacherTransitionOracleV62(config)
    model.load_codec_state(codec_checkpoint["model"])
    return model, codec_checkpoint


def main():
    args = parse_args()
    for name in (
        "data_index",
        "out",
        "gate_report",
        "decode_report",
        "dino_checkpoint",
        "siglip_checkpoint",
        "tracker_checkpoint",
        "codec_checkpoint",
        "resume",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    context = init_torchrun()
    validate_arguments(args, context.world_size)
    gate = read_gate(args)
    config = ObjectTransitionConfigV62()
    config.validate()
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    device = torch.device(context.device)
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if args.resume
        else None
    )
    if checkpoint is not None:
        validate_resume_v62(checkpoint, args, context, config)

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
        raise ValueError(f"v62 decode frontier differs: {decode_audit}")
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    model, codec_checkpoint = build_model(args, config)
    model = model.to(device)
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
    teacher = ObjectTransitionTeacherRuntimeV62(
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
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer.param_groups[0]["group_name"] = args.stage
    scheduler = cosine_schedule(
        optimizer, args.warmup_steps, args.steps, args.lr_floor_ratio
    )
    start_step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state_v62(checkpoint, context)
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
            raise ValueError(f"v62 output already contains a run: {args.out}")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "git_commit": args.source_revision,
            "historical_checkpoint_used": False,
            "e0_codec_checkpoint": args.codec_checkpoint,
            "training_only_teachers": ["DINOv2-L", "SigLIP", "CoTracker"],
            "patch_grid_is_object_target": False,
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
            **config.to_dict(),
            **source_summary,
            **vars(args),
        },
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "event": "v62_start",
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
    final_step = train_v62(
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
        print(json.dumps({"event": "v62_complete", "global_step": final_step}))
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
