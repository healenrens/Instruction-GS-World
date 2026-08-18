"""DDP training loop for objective-first Object State learning."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .gradient_health import (
    clip_finite_grad_norm_,
    optimizer_group_grad_norms,
    parameter_prefix_grad_norms,
)
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .v52_checkpointing import collect_rng_states, save_checkpoint


def checkpoint_due(step: int, args):
    if step == args.steps or step % args.save_every == 0:
        return os.path.join(args.out, f"v52_object_state_{step:07d}.pt"), "milestone"
    if step == 1 or step % args.recovery_every == 0:
        return os.path.join(args.out, "v52_object_state_recovery.pt"), "recovery"
    return None


def summarize_window(by_length, counts, world_size):
    total = sum(counts.values())
    if total < 1:
        raise ValueError("v52 metric window is empty")
    result = {
        "metric_window_microbatches": float(total),
        "history_length_coverage": float(len(counts)),
    }
    global_sums = {}
    chunk_sum = stride_sum = 0.0
    for length, local in sorted(by_length.items()):
        count = counts[length]
        reduced = reduce_metrics(
            {name: value / count for name, value in local.items()}, world_size
        )
        stride = reduced.pop("_temporal_stride")
        result[f"history_h{length}_updates"] = float(count)
        for name, value in reduced.items():
            result[f"history_h{length}_{name}"] = value
            global_sums[name] = global_sums.get(name, 0.0) + value * count
        chunk_sum += length * count
        stride_sum += stride * count
    result.update({name: value / total for name, value in global_sums.items()})
    result.update(chunk_length=chunk_sum / total, temporal_stride=stride_sum / total)
    return result


def train_v52(
    model, wrapped, dino, point_tracker, loader, sampler, optimizer, scheduler,
    context, args, start_step, wandb_tracker,
):
    if start_step >= args.steps:
        return start_step
    device = torch.device(context.device)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    usable_batches = len(loader) // args.grad_accum * args.grad_accum
    updates_per_epoch = usable_batches // args.grad_accum
    if updates_per_epoch < 1:
        raise ValueError("v52 loader cannot supply one optimizer update")
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    resume_batch_offset = update_offset * args.grad_accum
    direct_seek = bool(getattr(sampler, "supports_start_index", False))
    log_path = os.path.join(args.out, "train.jsonl")
    started, step = time.time(), int(start_step)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    metric_by_length, metric_counts = {}, {}
    while step < args.steps:
        sampler.set_epoch(args.seed + epoch)
        direct_batch_offset = 0
        if direct_seek:
            sampler.set_start_index(resume_batch_offset * args.batch)
            direct_batch_offset = resume_batch_offset
        micro_count = 0
        for batch_index, cpu_batch in enumerate(loader):
            if not direct_seek and batch_index < resume_batch_offset:
                continue
            if batch_index + direct_batch_offset >= usable_batches:
                break
            batch = move_to_device(cpu_batch, device)
            features = dino(batch)
            evidence = point_tracker(batch, features.patches, features.grid_hw)
            micro_count += 1
            synchronize = micro_count == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize else nullcontext()
            )
            with sync_context, amp_context():
                output = wrapped(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                    evidence,
                    features.grid_hw,
                )
                loss = output["loss"] / args.grad_accum
            loss.backward()
            length = int(batch["chunk_length"][0])
            parts = metric_by_length.setdefault(length, {})
            values = {
                **output["parts"],
                "_temporal_stride": batch["temporal_stride"].float().mean(),
            }
            for name, value in values.items():
                parts[name] = parts.get(name, value.detach() * 0.0) + value.detach()
            metric_counts[length] = metric_counts.get(length, 0) + 1
            if not synchronize:
                continue
            next_step = step + 1
            collect = next_step % args.log_every == 0 or next_step == args.steps
            norms = optimizer_group_grad_norms(optimizer) if collect else {}
            if collect:
                norms.update(parameter_prefix_grad_norms(
                    trainable,
                    {
                        "state_encoder": ("state_encoder.",),
                        "student_tracklets": ("student_tracklets.",),
                        "compositional_decoder": ("decoder.",),
                        "motion_head": ("motion_readout.",),
                    },
                ))
            grad_norm = clip_finite_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step = next_step
            micro_count = 0
            metrics = None
            if collect:
                metrics = summarize_window(metric_by_length, metric_counts, context.world_size)
                metric_by_length, metric_counts = {}, {}
            reduced_norms = reduce_metrics(norms, context.world_size) if norms else {}
            if context.is_main and collect:
                effective_batch = args.batch * context.world_size * args.grad_accum
                elapsed = max(time.time() - started, 1e-6)
                completed = max(step - start_step, 1)
                record = {
                    "phase": "object_state",
                    "global_step": step,
                    "sampler_epoch": epoch,
                    "data_epoch": step / updates_per_epoch,
                    "samples_seen": step * effective_batch,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "grad_clip_coefficient": min(
                        1.0, args.max_grad_norm / max(float(grad_norm), 1e-12)
                    ),
                    "steps_per_second": completed / elapsed,
                    "samples_per_second": effective_batch * completed / elapsed,
                    "wall_time_seconds": elapsed,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "world_size": context.world_size,
                    "effective_batch": effective_batch,
                    "updates_per_epoch": updates_per_epoch,
                    **metrics,
                    **reduced_norms,
                }
                record.update(cuda_memory_metrics(device))
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if wandb_tracker is not None:
                    wandb_tracker.log(record)
            target = checkpoint_due(step, args)
            if target is not None:
                path, kind = target
                rng_states = collect_rng_states(context)
                manifest = None
                if context.is_main:
                    manifest = save_checkpoint(
                        path, model, optimizer, scheduler, args, step, rng_states, kind
                    )
                if context.distributed:
                    torch.distributed.barrier()
                if context.is_main:
                    event = {"event": "checkpoint_saved", **manifest}
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if wandb_tracker is not None:
                        wandb_tracker.record_checkpoint(manifest)
            if step >= args.steps:
                break
        epoch += 1
        resume_batch_offset = 0
    return step
