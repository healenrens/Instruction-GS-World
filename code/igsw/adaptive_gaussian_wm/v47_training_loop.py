"""DDP training loop for the grounded v47 state-to-effect curriculum."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .gradient_health import clip_finite_grad_norm_, optimizer_group_grad_norms
from .length_metric_window import LengthMetricWindow
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .v47_checkpointing import collect_rng_states, save_checkpoint
from .v47_curriculum import curriculum_at


def _checkpoint_due(step: int, args) -> tuple[str, str] | None:
    milestone = args.save_every > 0 and step % args.save_every == 0
    recovery = step == 1 or (args.recovery_every > 0 and step % args.recovery_every == 0)
    if step == args.steps or milestone:
        return os.path.join(args.out, f"v47_{step:07d}.pt"), "milestone"
    if recovery:
        return os.path.join(args.out, "v47_recovery.pt"), "recovery"
    return None


def apply_curriculum_lrs(optimizer, scheduler, step: int, config) -> None:
    curriculum = curriculum_at(step, config)
    scheduled = scheduler.get_last_lr()
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("group_name", str(index))
        active = (
            curriculum.state_weight > 0.0 if name == "state" else
            curriculum.effect_weight > 0.0 if name == "effect" else
            curriculum.goal_weight > 0.0 if name == "goal" else False
        )
        group["lr"] = scheduled[index] if active else 0.0


def train_v47(
    model, wrapped, encoder, loader, sampler, optimizer, scheduler,
    context, args, start_step: int, tracker,
) -> int:
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
        raise ValueError("v47 loader cannot supply one optimizer update")
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    resume_batch_offset = update_offset * args.grad_accum
    direct_seek = bool(getattr(sampler, "supports_start_index", False))
    log_path = os.path.join(args.out, "train.jsonl")
    started = time.time()
    step = int(start_step)
    metric_window = LengthMetricWindow()
    apply_curriculum_lrs(optimizer, scheduler, step, model.config)
    optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        sampler.set_epoch(args.seed + epoch)
        direct_batch_offset = 0
        if direct_seek:
            sampler.set_start_index(resume_batch_offset * args.batch)
            direct_batch_offset = resume_batch_offset
        by_length: dict[int, dict[str, torch.Tensor]] = {}
        length_counts: dict[int, int] = {}
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
            sync_context = wrapped.no_sync() if context.distributed and not synchronize else nullcontext()
            with sync_context, amp_context():
                result = wrapped(
                    features.patches, features.coordinates, features.valid,
                    batch["frame_times"], batch["observation_mask"], step,
                )
                loss = result["loss"] / args.grad_accum
            loss.backward()
            length = int(batch["chunk_length"][0])
            length_parts = by_length.setdefault(length, {})
            for name, value in result["parts"].items():
                length_parts[name] = (
                    length_parts.get(name, value.detach() * 0.0) + value.detach()
                )
            for name, value in (
                ("_temporal_stride", batch["temporal_stride"].float().mean()),
                ("_observation_fraction", batch["observation_mask"].float().mean()),
            ):
                length_parts[name] = (
                    length_parts.get(name, value.detach() * 0.0) + value.detach()
                )
            length_counts[length] = length_counts.get(length, 0) + 1
            if not synchronize:
                continue
            collect = step == start_step or (step + 1) % args.log_every == 0
            group_norms = optimizer_group_grad_norms(optimizer) if collect else {}
            grad_norm = clip_finite_grad_norm_(model.named_parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            apply_curriculum_lrs(optimizer, scheduler, step, model.config)
            optimizer.zero_grad(set_to_none=True)
            reduced_group_norms = (
                reduce_metrics(group_norms, context.world_size) if group_norms else {}
            )
            micro_count = 0
            for length, length_parts in sorted(by_length.items()):
                count = length_counts[length]
                length_metrics = reduce_metrics(
                    {name: value / count for name, value in length_parts.items()},
                    context.world_size,
                )
                temporal_stride = length_metrics.pop("_temporal_stride")
                observation_fraction = length_metrics.pop("_observation_fraction")
                if context.is_main:
                    metric_window.add(
                        length_metrics,
                        float(length),
                        temporal_stride,
                        observation_fraction,
                        count,
                    )
            by_length = {}
            length_counts = {}
            if context.is_main and (step == 1 or step % args.log_every == 0 or step == args.steps):
                window_metrics = metric_window.summarize_and_reset()
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
                    **window_metrics,
                    **reduced_group_norms,
                }
                record.update({
                    f"lr_{group.get('group_name', index)}": group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                })
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
