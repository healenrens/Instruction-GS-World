"""DDP loop for v53 tokenizer and latent object-dynamics stages."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch

from .gradient_health import clip_finite_grad_norm_, optimizer_group_grad_norms
from .train_runtime import cuda_memory_metrics, move_to_device, reduce_metrics
from .v53_checkpointing import collect_rng_states, save_checkpoint


def select_stage_frames(batch: dict, stage: str) -> dict:
    frames = batch["video_rgb"].shape[1]
    if frames < 2:
        raise ValueError("v53 batch has fewer than two frames")
    if stage == "tokenizer" and frames > 2:
        indices = torch.tensor(
            (0, frames // 2, frames - 1), device=batch["video_rgb"].device
        )
    else:
        indices = torch.tensor((0, frames - 1), device=batch["video_rgb"].device)
    selected = dict(batch)
    for name in ("video_rgb", "video_pixel_valid", "observation_mask", "frame_times"):
        selected[name] = batch[name].index_select(1, indices)
    return selected


def _checkpoint_due(step: int, args):
    stem = f"v53_{args.stage}"
    if step == args.steps or step % args.save_every == 0:
        return os.path.join(args.out, f"{stem}_{step:07d}.pt"), "milestone"
    if step == 1 or step % args.recovery_every == 0:
        return os.path.join(args.out, f"{stem}_recovery.pt"), "recovery"
    return None


def _summarize(metrics, count, source_counts, task_groups, source_names, world_size):
    if count < 1:
        raise ValueError("v53 metric window is empty")
    reduced = reduce_metrics(
        {name: value / count for name, value in metrics.items()}, world_size
    )
    source_total = sum(source_counts.values())
    if source_total:
        template = next(iter(metrics.values())).detach().float() * 0.0
        fractions = {
            f"source_{name}_sample_fraction": template
            + source_counts.get(index, 0) / source_total
            for index, name in enumerate(source_names)
        }
        reduced.update(reduce_metrics(fractions, world_size))
        reduced.update(
            reduce_metrics(
                {
                    "source_coverage_rank_mean": template + float(len(source_counts)),
                    "task_group_coverage_rank_mean": template + float(len(task_groups)),
                },
                world_size,
            )
        )
    reduced["metric_window_microbatches"] = float(count)
    return reduced


def train_v53(
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
        raise ValueError("v53 loader cannot supply one optimizer update")
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    resume_batch_offset = update_offset * args.grad_accum
    direct_seek = bool(getattr(sampler, "supports_start_index", False))
    log_path = os.path.join(args.out, "train.jsonl")
    trainable = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    metric_sums, metric_count = {}, 0
    source_counts, task_groups = {}, set()
    source_names = tuple(loader.dataset.source_names)
    started, step = time.time(), int(start_step)
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
            batch = select_stage_frames(move_to_device(cpu_batch, device), args.stage)
            features = dino(batch)
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
                )
                loss = output["loss"] / args.grad_accum
            loss.backward()
            values = {
                **output["parts"],
                "temporal_stride": batch["temporal_stride"].float().mean(),
                "temporal_step_seconds": batch["temporal_step_seconds"].float().mean(),
                "chunk_length": batch["chunk_length"].float().mean(),
            }
            for name, value in values.items():
                metric_sums[name] = (
                    metric_sums.get(name, value.detach() * 0.0) + value.detach()
                )
            metric_count += 1
            for value in batch["source_index"].tolist():
                source_counts[int(value)] = source_counts.get(int(value), 0) + 1
            task_groups.update(
                int(value) for value in batch["task_group_index"].tolist()
            )
            if not synchronize:
                continue
            next_step = step + 1
            collect = next_step % args.log_every == 0 or next_step == args.steps
            norms = optimizer_group_grad_norms(optimizer) if collect else {}
            grad_norm = clip_finite_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step, micro_count = next_step, 0
            reduced_norms = reduce_metrics(norms, context.world_size) if norms else {}
            summarized = None
            if collect:
                summarized = _summarize(
                    metric_sums,
                    metric_count,
                    source_counts,
                    task_groups,
                    source_names,
                    context.world_size,
                )
            if context.is_main and collect:
                effective_batch = args.batch * context.world_size * args.grad_accum
                elapsed = max(time.time() - started, 1e-6)
                completed = max(step - start_step, 1)
                record = {
                    "phase": args.stage,
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
                    **summarized,
                    **reduced_norms,
                    **cuda_memory_metrics(device),
                }
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if wandb_tracker is not None:
                    wandb_tracker.log(record)
                metric_sums, metric_count = {}, 0
                source_counts, task_groups = {}, set()
            elif collect:
                metric_sums, metric_count = {}, 0
                source_counts, task_groups = {}, set()
            target_path = _checkpoint_due(step, args)
            if target_path is not None:
                path, kind = target_path
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
