"""DDP training loop for the v48 object-state model."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .gradient_health import clip_finite_grad_norm_, optimizer_group_grad_norms
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .v48_checkpointing import collect_rng_states, save_checkpoint


def _checkpoint_due(step: int, args) -> tuple[str, str] | None:
    if step == args.steps or step % args.save_every == 0:
        return os.path.join(args.out, f"v48_{step:07d}.pt"), "milestone"
    if step == 1 or step % args.recovery_every == 0:
        return os.path.join(args.out, "v48_recovery.pt"), "recovery"
    return None


def _summarize_window(by_length, counts, world_size: int) -> dict[str, float]:
    total = sum(counts.values())
    if total < 1:
        raise ValueError("v48 metric window is empty")
    summary: dict[str, float] = {
        "metric_window_microbatches": float(total),
        "history_length_coverage": float(len(counts)),
    }
    global_sums: dict[str, float] = {}
    chunk_sum = stride_sum = observation_sum = 0.0
    for length, local_parts in sorted(by_length.items()):
        count = counts[length]
        reduced = reduce_metrics(
            {name: value / count for name, value in local_parts.items()}, world_size
        )
        stride = reduced.pop("_temporal_stride")
        observation = reduced.pop("_observation_fraction")
        summary[f"history_h{length}_updates"] = float(count)
        for name, value in reduced.items():
            summary[f"history_h{length}_{name}"] = value
            global_sums[name] = global_sums.get(name, 0.0) + value * count
        chunk_sum += length * count
        stride_sum += stride * count
        observation_sum += observation * count
    summary.update({name: value / total for name, value in global_sums.items()})
    summary.update(
        chunk_length=chunk_sum / total,
        temporal_stride=stride_sum / total,
        observation_fraction=observation_sum / total,
    )
    return summary


def train_v48(
    model,
    wrapped,
    encoder,
    loader,
    sampler,
    optimizer,
    scheduler,
    context,
    args,
    start_step: int,
    tracker,
) -> int:
    if start_step >= args.steps:
        return start_step
    device = torch.device(context.device)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    usable_batches = len(loader) // args.grad_accum * args.grad_accum
    updates_per_epoch = usable_batches // args.grad_accum
    if updates_per_epoch < 1:
        raise ValueError("v48 loader cannot supply one optimizer update")
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    resume_batch_offset = update_offset * args.grad_accum
    direct_seek = bool(getattr(sampler, "supports_start_index", False))
    log_path = os.path.join(args.out, "train.jsonl")
    started = time.time()
    step = int(start_step)
    optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        sampler.set_epoch(args.seed + epoch)
        direct_batch_offset = 0
        if direct_seek:
            sampler.set_start_index(resume_batch_offset * args.batch)
            direct_batch_offset = resume_batch_offset
        by_length: dict[int, dict[str, torch.Tensor]] = {}
        counts: dict[int, int] = {}
        micro_count = 0
        for batch_index, cpu_batch in enumerate(loader):
            if not direct_seek and batch_index < resume_batch_offset:
                continue
            effective_index = batch_index + direct_batch_offset
            if effective_index >= usable_batches:
                break
            batch = move_to_device(cpu_batch, device)
            features = encoder(batch)
            micro_count += 1
            synchronize = micro_count == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            with sync_context, amp_context():
                output = wrapped(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                    batch["observation_mask"],
                )
                loss = output["loss"] / args.grad_accum
            loss.backward()
            length = int(batch["chunk_length"][0])
            parts = by_length.setdefault(length, {})
            values = {
                **output["parts"],
                "_temporal_stride": batch["temporal_stride"].float().mean(),
                "_observation_fraction": batch["observation_mask"].float().mean(),
            }
            for name, value in values.items():
                parts[name] = parts.get(name, value.detach() * 0.0) + value.detach()
            counts[length] = counts.get(length, 0) + 1
            if not synchronize:
                continue
            collect = step == start_step or (step + 1) % args.log_every == 0
            group_norms = optimizer_group_grad_norms(optimizer) if collect else {}
            grad_norm = clip_finite_grad_norm_(model.named_parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            reduced_norms = reduce_metrics(group_norms, context.world_size) if group_norms else {}
            metrics = _summarize_window(by_length, counts, context.world_size)
            by_length, counts, micro_count = {}, {}, 0
            if context.is_main and (
                step == 1 or step % args.log_every == 0 or step == args.steps
            ):
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
                if device.type == "cuda":
                    record.update(cuda_memory_metrics(device))
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if tracker is not None:
                    tracker.log(record)
            target = _checkpoint_due(step, args)
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
                    if tracker is not None:
                        tracker.record_checkpoint(manifest)
            if step >= args.steps:
                break
        epoch += 1
        resume_batch_offset = 0
    return step
