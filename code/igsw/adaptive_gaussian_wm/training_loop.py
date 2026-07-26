"""Shared phased training loop for the adaptive Gaussian world model."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .checkpointing import collect_rng_states, save_checkpoint
from .diagnostic_statistics import finalize_diagnostic_metrics
from .gradient_health import clip_finite_grad_norm_
from .train_runtime import (
    cuda_memory_metrics,
    move_to_device,
    reduce_metrics,
)
from .training_modes import update_target_for_training


def train_phase(
    phase: str,
    phase_steps: int,
    start_step: int,
    global_step: int,
    model,
    wrapped,
    loader: DataLoader,
    sampler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context,
    args,
    weights,
    experiment_tracker=None,
) -> tuple[int, int]:
    if start_step >= phase_steps:
        return start_step, global_step
    device = torch.device(context.device)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    log_path = os.path.join(args.out, "train.jsonl")
    step = start_step
    usable_batches = len(loader) // args.grad_accum * args.grad_accum
    updates_per_epoch = usable_batches // args.grad_accum
    epoch, update_offset = divmod(start_step, updates_per_epoch)
    skip_batches = update_offset * args.grad_accum
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < phase_steps:
        sampler.set_epoch(
            args.seed + epoch + (100000 if phase != "representation" else 0)
        )
        accumulated: dict[str, torch.Tensor] = {}
        micro_count = 0
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index < skip_batches:
                continue
            if batch_index >= usable_batches:
                break
            batch = move_to_device(cpu_batch, device)
            micro_count += 1
            synchronize = micro_count == args.grad_accum
            collect_diagnostics = (
                step + 1 == 1
                or (step + 1) % args.log_every == 0
                or step + 1 == phase_steps
            )
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            with sync_context, amp_context():
                if phase == "representation":
                    model_phase = (
                        "object_memory_representation_loss"
                        if args.architecture == "object_memory_v1"
                        else "representation"
                    )
                    result = wrapped(
                        batch,
                        phase=model_phase,
                        loss_weights=(
                            weights
                            if model_phase == "object_memory_representation_loss"
                            else None
                        ),
                        collect_diagnostics=collect_diagnostics,
                    )
                    loss, parts = result["loss"], result["parts"]
                else:
                    result = wrapped(
                        batch,
                        phase=(
                            "posterior_dynamics_loss"
                            if args.posterior_dynamics_gate
                            else "joint_loss"
                        ),
                        loss_weights=weights,
                        collect_diagnostics=collect_diagnostics,
                    )
                    loss, parts = result["loss"], result["parts"]
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            for name, value in parts.items():
                accumulated[name] = accumulated.get(
                    name, value.detach() * 0.0
                ) + value.detach()
            if not synchronize:
                continue
            grad_norm = clip_finite_grad_norm_(
                model.named_parameters(),
                5.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_target_for_training(model, args.posterior_dynamics_gate)
            step += 1
            global_step += 1
            metrics = finalize_diagnostic_metrics(
                reduce_metrics(
                    {
                        name: value / args.grad_accum
                        for name, value in accumulated.items()
                    },
                    context.world_size,
                )
            )
            accumulated = {}
            micro_count = 0
            if context.is_main and (
                step == 1 or step % args.log_every == 0 or step == phase_steps
            ):
                effective_batch = (
                    args.batch * context.world_size * args.grad_accum
                )
                elapsed = max(time.time() - started, 1e-6)
                completed_steps = step - start_step
                steps_per_second = completed_steps / elapsed
                record = {
                    "phase": phase,
                    "phase_step": step,
                    "global_step": global_step,
                    "sampler_epoch": epoch,
                    "data_epoch": step / updates_per_epoch,
                    "phase_samples_seen": step * effective_batch,
                    "samples_seen": global_step * effective_batch,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "steps_per_second": steps_per_second,
                    "samples_per_second": (
                        effective_batch * steps_per_second
                    ),
                    "wall_time_seconds": elapsed,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "world_size": context.world_size,
                    "effective_batch": effective_batch,
                    "updates_per_epoch": updates_per_epoch,
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
                if experiment_tracker is not None:
                    experiment_tracker.log(record)
            should_save = args.save_every > 0 and (
                global_step % args.save_every == 0 or step == phase_steps
            )
            if should_save:
                rng_states = collect_rng_states(context)
                if context.is_main:
                    save_checkpoint(
                        os.path.join(args.out, f"{phase}_{step:07d}.pt"),
                        model,
                        optimizer,
                        scheduler,
                        args,
                        phase,
                        step,
                        global_step,
                        rng_states,
                    )
                if context.distributed:
                    torch.distributed.barrier()
            if step >= phase_steps:
                break
        epoch += 1
        skip_batches = 0
    return step, global_step
