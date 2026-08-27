"""DDP loop for v61 Object State encoder comparisons."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .carrier_teacher_v61 import build_object_components_v61
from .gradient_health import (
    clip_finite_grad_norm_,
    optimizer_group_grad_norms,
    parameter_prefix_grad_norms,
)
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .trajectory_relation_teacher_v56 import build_trajectory_relation_teacher_v56
from .v61_checkpointing import collect_rng_states, save_checkpoint


def _checkpoint_due(step: int, args):
    if step == args.steps or step % args.save_every == 0:
        return os.path.join(args.out, f"v61_{args.variant}_{step:07d}.pt"), "milestone"
    if step == 1 or step % args.recovery_every == 0:
        return os.path.join(args.out, f"v61_{args.variant}_recovery.pt"), "recovery"
    return None


def _source_fractions(reference, counts, names):
    total = sum(counts.values())
    return {
        f"source_{name}_sample_fraction": reference.new_tensor(counts.get(i, 0) / total)
        for i, name in enumerate(names)
    }


def _summarize(parts, count, source_counts, task_groups, source_names, world_size):
    averaged = {name: value / count for name, value in parts.items()}
    result = reduce_metrics(averaged, world_size)
    reference = next(iter(averaged.values()))
    result.update(
        reduce_metrics(
            _source_fractions(reference, source_counts, source_names), world_size
        )
    )
    result.update(
        reduce_metrics(
            {
                "source_coverage_rank_mean": reference.new_tensor(len(source_counts)),
                "task_group_coverage_rank_mean": reference.new_tensor(len(task_groups)),
                "metric_window_microbatches": reference.new_tensor(count),
            },
            world_size,
        )
    )
    return result


def train_v61(
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
):
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
        raise ValueError("v61 loader cannot supply one optimizer update")
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    resume_batch_offset = update_offset * args.grad_accum
    direct_seek = bool(getattr(sampler, "supports_start_index", False))
    log_path = os.path.join(args.out, "train.jsonl")
    started, step = time.time(), int(start_step)
    trainable = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    metric_parts, metric_count = {}, 0
    source_counts, task_groups = {}, set()
    source_names = tuple(loader.dataset.source_names)
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
            teacher_features = dino_teacher(batch)
            evidence = point_tracker(
                batch, teacher_features.patches, teacher_features.grid_hw
            )
            relation = build_trajectory_relation_teacher_v56(
                evidence, model.config, batch["frame_times"]
            )
            components = build_object_components_v61(
                batch,
                evidence,
                relation,
                model.config.object_roots,
                semantic_teacher,
            )
            micro_count += 1
            synchronize = micro_count == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            with sync_context, amp_context():
                output = wrapped(batch, evidence, relation, components)
                loss = output["loss"] / args.grad_accum
            loss.backward()
            values = {
                **output["parts"],
                "chunk_length": batch["chunk_length"].float().mean(),
                "temporal_stride": batch["temporal_stride"].float().mean(),
                "temporal_step_seconds": batch["temporal_step_seconds"].float().mean(),
                "decode_replacement_fraction": batch["decode_replaced"].float().mean(),
            }
            for value in batch["source_index"].tolist():
                index = int(value)
                source_counts[index] = source_counts.get(index, 0) + 1
            task_groups.update(
                int(value) for value in batch["task_group_index"].tolist()
            )
            for name, value in values.items():
                tensor = value if torch.is_tensor(value) else loss.new_tensor(value)
                metric_parts[name] = (
                    metric_parts.get(name, tensor.detach() * 0.0) + tensor.detach()
                )
            metric_count += 1
            if not synchronize:
                continue
            next_step = step + 1
            collect = next_step % args.log_every == 0 or next_step == args.steps
            norms = optimizer_group_grad_norms(optimizer) if collect else {}
            if collect:
                norms.update(
                    parameter_prefix_grad_norms(
                        trainable,
                        {
                            "student_backbone": ("student.backbone.",),
                            "student_projector": ("student.projector.",),
                            "carrier_state": ("state_encoder.carriers.",),
                            "object_roots": ("state_encoder.roots.",),
                            "alignment_heads": (
                                "identity_to_dino.",
                                "root_to_siglip.",
                                "motion_readout.",
                            ),
                        },
                    )
                )
            grad_norm = clip_finite_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step, micro_count = next_step, 0
            metrics = None
            if collect:
                metrics = _summarize(
                    metric_parts,
                    metric_count,
                    source_counts,
                    task_groups,
                    source_names,
                    context.world_size,
                )
                metric_parts, metric_count = {}, 0
                source_counts, task_groups = {}, set()
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
                    "lr": scheduler.get_last_lr()[-1],
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_heads": optimizer.param_groups[-1]["lr"],
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
                    **cuda_memory_metrics(device),
                }
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if wandb_tracker is not None:
                    wandb_tracker.log(record)
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
                    if wandb_tracker is not None:
                        wandb_tracker.record_checkpoint(manifest)
            if step >= args.steps:
                break
        epoch += 1
        resume_batch_offset = 0
    return step
