"""Train a capacity-matched unstructured visual-sequence posterior core."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    collect_rng_states,
    restore_rng_state,
)
from igsw.adaptive_gaussian_wm.experiment_tracking import (  # noqa: E402
    add_wandb_arguments,
    init_wandb_tracker,
    validate_wandb_arguments,
)
from igsw.adaptive_gaussian_wm.flat_baseline_checkpointing import (  # noqa: E402
    flat_parameter_metrics,
    initialize_flat_dynamics,
    reference_contract,
    save_flat_checkpoint,
    source_digest,
    validate_flat_resume,
)
from igsw.adaptive_gaussian_wm.gradient_health import (  # noqa: E402
    clip_finite_grad_norm_,
)
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.matched_flat_objective import (  # noqa: E402
    FLAT_TRAIN_METRICS,
    flat_training_objective,
)
from igsw.adaptive_gaussian_wm.matched_flat_rgb import (  # noqa: E402
    batch_rgb_grids,
)
from igsw.adaptive_gaussian_wm.matched_flat_training_contract import (  # noqa: E402
    flat_training_mismatches,
)
from igsw.adaptive_gaussian_wm.matched_flat_world_model import (  # noqa: E402
    MatchedFlatLatentWorldModel,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    cosine_schedule,
    cuda_memory_metrics,
    move_to_device,
    reduce_metrics,
)
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_checkpoint", required=True)
    parser.add_argument("--required_reference_step", type=int, default=12000)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--sequence_anchors", default="3,5,8")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument(
        "--gradient_checkpointing",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--change_loss_weight", type=float, default=1.0)
    parser.add_argument("--history_loss_weight", type=float, default=1.0)
    parser.add_argument("--rgb_short_side", type=int, default=256)
    parser.add_argument("--rgb_pad_multiple", type=int, default=16)
    parser.add_argument("--rgb_loss_weight", type=float, default=0.5)
    parser.add_argument("--rgb_ssim_weight", type=float, default=0.2)
    parser.add_argument("--rgb_change_loss_weight", type=float, default=1.0)
    parser.add_argument("--rgb_change_threshold", type=float, default=0.04)
    parser.add_argument("--gap_reference", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=4000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    add_wandb_arguments(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "required_reference_step",
        "steps",
        "batch",
        "grad_accum",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("flat baseline positive arguments are invalid")
    if args.workers < 0 or args.max_train_items < 0:
        raise ValueError("workers and max_train_items cannot be negative")
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("lr_floor must be in (0, lr]")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup_steps must be in [0, steps)")
    loss_weights = (
        args.change_loss_weight,
        args.history_loss_weight,
        args.rgb_loss_weight,
        args.rgb_ssim_weight,
        args.rgb_change_loss_weight,
    )
    if min(loss_weights) < 0.0 or args.rgb_change_threshold <= 0.0:
        raise ValueError("flat baseline loss weights cannot be negative")
    if args.rgb_short_side < 16 or args.rgb_pad_multiple < 1:
        raise ValueError("flat baseline RGB resize configuration is invalid")
    if args.gap_reference <= 0.0 or args.log_every <= 0 or args.save_every < 0:
        raise ValueError("flat baseline runtime arguments are invalid")
    for name in ("reference_checkpoint", "data", "out"):
        if not os.path.isabs(getattr(args, name)):
            raise ValueError(f"{name} must be absolute")
    validate_wandb_arguments(args)


def main() -> None:
    args = parse_args()
    args.baseline_contract_version = 2
    args.modality_matching = "dino_rgb"
    validate_args(args)
    context = init_torchrun()
    training_mismatches = flat_training_mismatches(
        {
            "global_step": args.steps,
            "world_size": context.world_size,
            "args": vars(args),
        },
        args.steps,
    )
    if training_mismatches:
        raise ValueError(
            "flat baseline differs from its pre-registered training contract: "
            + json.dumps(training_mismatches, sort_keys=True)
        )
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = CausalVisualSequenceDataset(
        args.data,
        "train",
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        anchors=args.sequence_anchors,
        max_items=args.max_train_items,
        load_rgb=True,
        rgb_short_side=args.rgb_short_side,
        rgb_pad_multiple=args.rgb_pad_multiple,
    )
    args.data_sha256 = dataset.data_sha256
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    reference = torch.load(
        args.reference_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    architecture = {
        "feature_dim": dataset.feature_dim,
        **reference_contract(reference, args, dataset),
    }
    args.reference_checkpoint_sha256 = source_digest(
        args.reference_checkpoint,
        context,
    )
    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        if not args.resume and os.path.lexists(os.path.join(args.out, "latest.pt")):
            raise ValueError(f"output already contains a run: {args.out}")
    if context.distributed:
        dist.barrier()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        validate_flat_resume(checkpoint, args, context.world_size)
        if checkpoint.get("architecture") != architecture:
            raise ValueError("flat baseline architecture differs on resume")
    model = MatchedFlatLatentWorldModel(**architecture).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization = checkpoint.get("initialization", {})
        if (
            initialization.get("kind") != "object_dynamics_blocks_only"
            or initialization.get("source_checkpoint_sha256")
            != args.reference_checkpoint_sha256
        ):
            raise ValueError("flat baseline initialization provenance differs")
    else:
        initialization = initialize_flat_dynamics(
            model,
            reference,
            args.reference_checkpoint,
            args.reference_checkpoint_sha256,
        )
        if context.is_main:
            with open(
                os.path.join(args.out, "initialization_report.json"),
                "x",
                encoding="utf-8",
            ) as handle:
                json.dump(initialization, handle, indent=2, sort_keys=True)
                handle.write("\n")
    del reference
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if context.distributed
        else model
    )
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    sampler = build_training_sampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        seed=args.seed,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    if len(loader) < args.grad_accum:
        raise ValueError("flat baseline has fewer batches than one optimizer step")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule(
        optimizer,
        args.warmup_steps,
        args.steps,
        args.lr_floor / args.lr,
    )
    step = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint, context)
        step = int(checkpoint["global_step"])
    if not 0 <= step <= args.steps:
        raise ValueError("flat baseline checkpoint step is outside the run")
    parameter_metrics = flat_parameter_metrics(model)
    effective_batch = args.batch * context.world_size * args.grad_accum
    tracker = init_wandb_tracker(
        args,
        context,
        {
            "arguments": vars(args),
            "architecture": architecture,
            "parameters": parameter_metrics,
            "initialization": {
                name: value
                for name, value in initialization.items()
                if name != "loaded"
            },
            "dataset": {
                "examples": len(dataset),
                "contract": dataset.contract_label,
                "sha256": dataset.data_sha256,
            },
            "runtime": {
                "parallelism": "ddp_full_state_dict",
                "world_size": context.world_size,
                "effective_batch": effective_batch,
            },
            "scope": {
                "history_only": "shared_unstructured_zero_action_floor",
                "posterior_core": "matched_unstructured_continuous_action_oracle",
                "future_path": "posterior_bottleneck_only",
                "modality_matching": "dino_rgb",
                "canonical_action": "pooled_dino_effect_3_plus_rgb_logit_effect_3",
                "residual_action_dimensions": architecture["action_residual_dim"],
                "dense_readout": "dino_residual_plus_rgb_logit_residual",
                "excluded": "object_assignment_centers_covariance_gaussian_splatting",
            },
        },
    )
    if checkpoint is None:
        initial_rng_states = collect_rng_states(context)
        if context.is_main:
            save_flat_checkpoint(
                os.path.join(args.out, "flat_suite_0000000.pt"),
                model,
                optimizer,
                scheduler,
                args,
                0,
                initial_rng_states,
                architecture,
                initialization,
            )
        if context.distributed:
            dist.barrier()
    usable_batches = len(loader) // args.grad_accum * args.grad_accum
    updates_per_epoch = usable_batches // args.grad_accum
    epoch, update_offset = divmod(step, updates_per_epoch)
    skip_batches = update_offset * args.grad_accum
    start_step = step
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    metric_names = FLAT_TRAIN_METRICS
    while step < args.steps:
        sampler.set_epoch(args.seed + epoch)
        accumulated = {
            name: torch.zeros((), device=device) for name in metric_names
        }
        micro_count = 0
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index < skip_batches:
                continue
            if batch_index >= usable_batches:
                break
            batch = move_to_device(cpu_batch, device)
            micro_count += 1
            synchronize = micro_count == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            history_scale = signed_gap_scale(
                batch["history_times"],
                args.gap_reference,
            )
            future_scale = signed_gap_scale(
                batch["future_times"],
                args.gap_reference,
            )
            history_rgb_grid, future_rgb_grid = batch_rgb_grids(batch)
            with sync_context, amp_context():
                output = wrapped(
                    batch["history_features"],
                    batch["history_coordinates"],
                    history_scale,
                    history_rgb_grid,
                    batch["future_features"],
                    batch["future_coordinates"],
                    future_scale,
                    future_rgb_grid,
                )
                loss, metrics = flat_training_objective(output, batch, args)
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            for name, value in metrics.items():
                accumulated[name] += value.detach()
            if not synchronize:
                continue
            grad_norm = clip_finite_grad_norm_(model.named_parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            reduced = reduce_metrics(
                {
                    name: value / args.grad_accum
                    for name, value in accumulated.items()
                },
                context.world_size,
            )
            accumulated = {
                name: value * 0.0 for name, value in accumulated.items()
            }
            micro_count = 0
            step += 1
            if context.is_main and (
                step == 1 or step % args.log_every == 0 or step == args.steps
            ):
                elapsed = max(time.time() - started, 1e-6)
                completed = step - start_step
                record = {
                    "global_step": step,
                    "sampler_epoch": epoch,
                    "data_epoch": step / updates_per_epoch,
                    "samples_seen": step * effective_batch,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "steps_per_second": completed / elapsed,
                    "samples_per_second": effective_batch * completed / elapsed,
                    "wall_time_seconds": elapsed,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "world_size": context.world_size,
                    "effective_batch": effective_batch,
                    "updates_per_epoch": updates_per_epoch,
                    **cuda_memory_metrics(device),
                    **parameter_metrics,
                    **reduced,
                }
                record["posterior_gain_over_history"] = (
                    record["history_feature_mse"]
                    - record["posterior_feature_mse"]
                )
                record["posterior_rgb_gain_over_history"] = (
                    record["history_rgb_distance"]
                    - record["posterior_rgb_distance"]
                )
                with open(
                    os.path.join(args.out, "train.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if tracker is not None:
                    tracker.log(record)
            should_save = args.save_every > 0 and (
                step % args.save_every == 0 or step == args.steps
            )
            if should_save:
                rng_states = collect_rng_states(context)
                if context.is_main:
                    save_flat_checkpoint(
                        os.path.join(args.out, f"flat_suite_{step:07d}.pt"),
                        model,
                        optimizer,
                        scheduler,
                        args,
                        step,
                        rng_states,
                        architecture,
                        initialization,
                    )
                if context.distributed:
                    dist.barrier()
            if step >= args.steps:
                break
        epoch += 1
        skip_batches = 0
    if tracker is not None:
        tracker.finish()
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
