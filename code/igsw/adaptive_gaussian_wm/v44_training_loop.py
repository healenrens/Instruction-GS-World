"""DDP training loop for a single resumable v44 curriculum run."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .gradient_health import clip_finite_grad_norm_, optimizer_group_grad_norms
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .v44_checkpointing import collect_rng_states, save_checkpoint


def _checkpoint_due(step: int, args) -> tuple[str, str] | None:
    milestone = args.save_every > 0 and step % args.save_every == 0
    recovery = step == 1 or (
        args.recovery_every > 0 and step % args.recovery_every == 0
    )
    final = step == args.steps
    if final or milestone:
        return os.path.join(args.out, f"v44_{step:07d}.pt"), "milestone"
    if recovery:
        return os.path.join(args.out, "v44_recovery.pt"), "recovery"
    return None


def train_v44(
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
        raise ValueError("v44 loader cannot supply one optimizer update")
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
        accumulated: dict[str, torch.Tensor] = {}
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
                result = wrapped(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                    batch["observation_mask"],
                    step,
                )
                loss = result["loss"] / args.grad_accum
            loss.backward()
            for name, value in result["parts"].items():
                accumulated[name] = (
                    accumulated.get(name, value.detach() * 0.0) + value.detach()
                )
            if not synchronize:
                continue
            collect = step == start_step or (step + 1) % args.log_every == 0
            group_norms = optimizer_group_grad_norms(optimizer) if collect else {}
            grad_norm = clip_finite_grad_norm_(
                model.named_parameters(), args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = reduce_metrics(
                {name: value / args.grad_accum for name, value in accumulated.items()},
                context.world_size,
            )
            if group_norms:
                metrics.update(reduce_metrics(group_norms, context.world_size))
            accumulated = {}
            micro_count = 0
            if context.is_main and (
                step == 1 or step % args.log_every == 0 or step == args.steps
            ):
                effective_batch = args.batch * context.world_size * args.grad_accum
                elapsed = max(time.time() - started, 1e-6)
                completed = max(step - start_step, 1)
                record = {
                    "phase": result["curriculum"].phase,
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
                    "chunk_length": float(batch["chunk_length"].float().mean()),
                    "temporal_stride": float(batch["temporal_stride"].float().mean()),
                    "observation_fraction": float(
                        batch["observation_mask"].float().mean()
                    ),
                    **metrics,
                }
                record.update(
                    {
                        f"lr_{group.get('group_name', index)}": group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    }
                )
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
