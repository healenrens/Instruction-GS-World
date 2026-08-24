"""DDP loop for the isolated object-transition objective."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .gradient_health import clip_finite_grad_norm_, optimizer_group_grad_norms
from .object_transition_objective_v59 import object_transition_objective_v59
from .query_object_teacher_v57 import (
    build_query_object_teacher_v57,
    query_teacher_contract_metrics,
)
from .query_transition_target_v59 import build_query_transition_target_v59
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .trajectory_relation_teacher_v56 import build_trajectory_relation_teacher_v56
from .v59_checkpointing import collect_rng_states, save_checkpoint


def checkpoint_due(step: int, args, launch_stop_step: int):
    if step in {args.steps, launch_stop_step} or step % args.save_every == 0:
        return os.path.join(args.out, f"v59_transition_{step:07d}.pt"), "milestone"
    if step == 1 or step % args.recovery_every == 0:
        return os.path.join(args.out, "v59_transition_recovery.pt"), "recovery"
    return None


def _source_fractions(reference, counts, source_names):
    total = sum(counts.values())
    if not total:
        return {}
    return {
        f"source_{name}_sample_fraction": reference.new_tensor(
            counts.get(index, 0) / total
        )
        for index, name in enumerate(source_names)
    }


def _summarize(parts, count, source_counts, source_names, world_size):
    averaged = {name: value / count for name, value in parts.items()}
    result = reduce_metrics(averaged, world_size)
    reference = next(iter(averaged.values()))
    result.update(
        reduce_metrics(
            _source_fractions(reference, source_counts, source_names), world_size
        )
    )
    return result


def train_v59(
    model,
    wrapped,
    dino,
    point_tracker,
    loader,
    sampler,
    optimizer,
    scheduler,
    context,
    args,
    start_step,
    wandb_tracker,
    init_report,
):
    if start_step >= args.steps:
        return start_step
    launch_stop_step = args.steps
    if args.run_steps:
        launch_stop_step = min(args.steps, start_step + args.run_steps)
    device = torch.device(context.device)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    usable_batches = len(loader) // args.grad_accum * args.grad_accum
    updates_per_epoch = usable_batches // args.grad_accum
    if updates_per_epoch < 1:
        raise ValueError("v59 loader cannot supply one optimizer update")
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    resume_batch_offset = update_offset * args.grad_accum
    direct_seek = bool(getattr(sampler, "supports_start_index", False))
    log_path = os.path.join(args.out, "train.jsonl")
    started, step = time.time(), int(start_step)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    metric_parts, metric_count, source_counts = {}, 0, {}
    source_names = tuple(loader.dataset.source_names)
    while step < launch_stop_step:
        sampler.set_epoch(args.seed + epoch)
        direct_batch_offset = 0
        if direct_seek:
            sampler.set_start_index(resume_batch_offset * args.batch)
            direct_batch_offset = resume_batch_offset
        micro_count = 0
        for batch_index, cpu_batch in enumerate(loader):
            if not direct_seek and batch_index < resume_batch_offset:
                continue
            absolute_batch = batch_index + direct_batch_offset
            if absolute_batch >= usable_batches:
                break
            batch = move_to_device(cpu_batch, device)
            chunk_lengths = batch["chunk_length"].unique()
            if len(chunk_lengths) != 1:
                raise RuntimeError("v59 local microbatch mixes temporal contracts")
            observed_frames = int(chunk_lengths.item()) - args.teacher_future_frames
            features = dino(batch)
            evidence = point_tracker(batch, features.patches, features.grid_hw)
            relation = build_trajectory_relation_teacher_v56(
                evidence, model.config, batch["frame_times"]
            )
            binding = build_query_object_teacher_v57(
                evidence,
                relation,
                model.config,
                observed_frames=observed_frames,
            )
            target = build_query_transition_target_v59(
                evidence,
                relation,
                binding,
                batch["frame_times"],
                observed_frames,
                model.config,
            )
            micro_count += 1
            synchronize = micro_count == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            with sync_context, amp_context():
                output = wrapped(
                    features.patches[:, :observed_frames],
                    features.coordinates[:, :observed_frames],
                    features.valid[:, :observed_frames],
                    batch["frame_times"][:, :observed_frames],
                    binding.query_coordinate,
                    target,
                )
                total, parts = object_transition_objective_v59(
                    output, target, model.config
                )
                loss = total / args.grad_accum
            loss.backward()
            metrics = {
                **parts,
                **query_teacher_contract_metrics(binding),
                "history_frames": loss.new_tensor(observed_frames),
                "teacher_future_frames": loss.new_tensor(args.teacher_future_frames),
                "temporal_step_seconds": batch["temporal_step_seconds"].float().mean(),
                "decode_replacement_fraction": batch["decode_replaced"].float().mean(),
            }
            for source_index in batch["source_index"].tolist():
                index = int(source_index)
                source_counts[index] = source_counts.get(index, 0) + 1
            for name, value in metrics.items():
                tensor = value if torch.is_tensor(value) else loss.new_tensor(value)
                metric_parts[name] = metric_parts.get(name, tensor.detach() * 0.0)
                metric_parts[name] = metric_parts[name] + tensor.detach()
            metric_count += 1
            if not synchronize:
                continue
            next_step = step + 1
            collect = next_step % args.log_every == 0 or next_step == launch_stop_step
            norms = optimizer_group_grad_norms(optimizer) if collect else {}
            grad_norm = clip_finite_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step, micro_count = next_step, 0
            window = None
            if collect:
                window = _summarize(
                    metric_parts,
                    metric_count,
                    source_counts,
                    source_names,
                    context.world_size,
                )
                metric_parts, metric_count, source_counts = {}, 0, {}
            reduced_norms = reduce_metrics(norms, context.world_size) if norms else {}
            if context.is_main and collect:
                effective_batch = args.batch * context.world_size * args.grad_accum
                elapsed = max(time.time() - started, 1e-6)
                completed = max(step - start_step, 1)
                record = {
                    "phase": "object_transition_objective",
                    "global_step": step,
                    "sampler_epoch": epoch,
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
                    **window,
                    **reduced_norms,
                    **cuda_memory_metrics(device),
                }
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if wandb_tracker is not None:
                    wandb_tracker.log(record)
            destination = checkpoint_due(step, args, launch_stop_step)
            if destination is not None:
                path, kind = destination
                rng_states = collect_rng_states(context)
                manifest = None
                if context.is_main:
                    manifest = save_checkpoint(
                        path,
                        model,
                        optimizer,
                        scheduler,
                        args,
                        step,
                        rng_states,
                        kind,
                        init_report,
                    )
                if context.distributed:
                    torch.distributed.barrier()
                if context.is_main:
                    event = {"event": "checkpoint_saved", **manifest}
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if wandb_tracker is not None:
                        wandb_tracker.record_checkpoint(manifest)
            if step >= launch_stop_step:
                break
        epoch += 1
        resume_batch_offset = 0
    return step
