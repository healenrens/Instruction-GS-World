"""Shared phased training loop for the adaptive Gaussian world model."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .checkpointing import checkpoint_target, collect_rng_states, save_checkpoint
from .diagnostic_statistics import finalize_diagnostic_metrics
from .gradient_health import clip_finite_grad_norm_, optimizer_group_grad_norms
from .train_runtime import (
    cuda_memory_metrics,
    move_to_device,
    reduce_metrics,
)
from .training_modes import update_target_for_training
from .training_health import (
    CORRESPONDENCE_MASS_METRICS,
    enforce_object_memory_training_health,
)


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
    feature_runtime=None,
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
    resume_batch_offset = update_offset * args.grad_accum
    direct_sampler_seek = bool(
        getattr(sampler, "supports_start_index", False)
    )
    if context.is_main and resume_batch_offset:
        print(
            json.dumps(
                {
                    "event": "resume_data_position",
                    "epoch": epoch,
                    "optimizer_update_offset": update_offset,
                    "local_batch_offset": resume_batch_offset,
                    "local_sample_offset": resume_batch_offset * args.batch,
                    "direct_sampler_seek": direct_sampler_seek,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < phase_steps:
        action_free_phase = phase in ("representation", "readout")
        sampler.set_epoch(args.seed + epoch + (0 if action_free_phase else 100000))
        direct_batch_offset = 0
        if direct_sampler_seek:
            sampler.set_start_index(resume_batch_offset * args.batch)
            direct_batch_offset = resume_batch_offset
        accumulated: dict[str, torch.Tensor] = {}
        micro_count = 0
        for batch_index, cpu_batch in enumerate(loader):
            if not direct_sampler_seek and batch_index < resume_batch_offset:
                continue
            effective_batch_index = batch_index + direct_batch_offset
            if effective_batch_index >= usable_batches:
                break
            batch = move_to_device(cpu_batch, device)
            if feature_runtime is not None:
                batch = feature_runtime(batch)
            if model.config.object_region_memory:
                model.set_curriculum_step(global_step)
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
                if action_free_phase:
                    model_phase = (
                        "object_memory_representation_loss"
                        if args.architecture
                        in ("object_memory_v1", "object_memory_v2", "object_memory_v3")
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
                    if args.training_stage == "prior":
                        model_phase = "history_prior_loss"
                    elif args.training_stage == "posterior":
                        model_phase = "posterior_dynamics_loss"
                    else:
                        model_phase = (
                            "posterior_dynamics_loss"
                            if args.posterior_dynamics_gate
                            else "joint_loss"
                        )
                    result = wrapped(
                        batch,
                        phase=model_phase,
                        loss_weights=weights,
                        collect_diagnostics=collect_diagnostics,
                    )
                    loss, parts = result["loss"], result["parts"]
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            for name, value in parts.items():
                if name in CORRESPONDENCE_MASS_METRICS:
                    previous = accumulated.get(name)
                    accumulated[name] = (
                        value.detach()
                        if previous is None
                        else torch.maximum(previous, value.detach())
                    )
                else:
                    accumulated[name] = (
                        accumulated.get(name, value.detach() * 0.0) + value.detach()
                    )
            if not synchronize:
                continue
            group_grad_norms = (
                optimizer_group_grad_norms(optimizer) if collect_diagnostics else {}
            )
            grad_norm = clip_finite_grad_norm_(
                model.named_parameters(),
                5.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_target_for_training(
                model,
                args.posterior_dynamics_gate,
                freeze_target=(
                    args.training_stage == "prior"
                    or (
                        args.training_stage == "readout"
                        and args.readout_scope == "isolated"
                    )
                ),
            )
            step += 1
            global_step += 1
            if model.config.object_region_memory:
                model.set_curriculum_step(global_step)
            metrics = finalize_diagnostic_metrics(
                reduce_metrics(
                    {
                        name: (
                            value
                            if name in CORRESPONDENCE_MASS_METRICS
                            else value / args.grad_accum
                        )
                        for name, value in accumulated.items()
                    },
                    context.world_size,
                    CORRESPONDENCE_MASS_METRICS,
                )
            )
            if group_grad_norms:
                metrics.update(reduce_metrics(group_grad_norms, context.world_size))
            enforce_object_memory_training_health(model.config, metrics)
            accumulated = {}
            micro_count = 0
            if context.is_main and (
                step == 1 or step % args.log_every == 0 or step == phase_steps
            ):
                effective_batch = args.batch * context.world_size * args.grad_accum
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
                    "grad_clip_scale": min(1.0, 5.0 / max(float(grad_norm), 1e-12)),
                    "grad_clipped": float(float(grad_norm) > 5.0),
                    "steps_per_second": steps_per_second,
                    "samples_per_second": (effective_batch * steps_per_second),
                    "wall_time_seconds": elapsed,
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "world_size": context.world_size,
                    "effective_batch": effective_batch,
                    "updates_per_epoch": updates_per_epoch,
                    **metrics,
                }
                if model.config.object_region_memory:
                    record["curriculum_phase"] = result["curriculum"].phase
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
            target = checkpoint_target(
                args.out,
                phase,
                step,
                global_step,
                phase_steps,
                args.save_every,
                args.recovery_every,
            )
            if target is not None:
                checkpoint_path, checkpoint_kind = target
                rng_states = collect_rng_states(context)
                manifest = None
                if context.is_main:
                    manifest = save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        args,
                        phase,
                        step,
                        global_step,
                        rng_states,
                        checkpoint_kind,
                    )
                if context.distributed:
                    torch.distributed.barrier()
                if context.is_main:
                    event = {"event": "checkpoint_saved", **manifest}
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if experiment_tracker is not None:
                        experiment_tracker.record_checkpoint(manifest)
            if step >= phase_steps:
                break
        epoch += 1
        resume_batch_offset = 0
    return step, global_step


def train_stages(
    representation_step,
    joint_step,
    global_step,
    model,
    wrapped,
    loader,
    sampler,
    optimizer,
    scheduler,
    context,
    args,
    weights,
    experiment_tracker,
    feature_runtime,
):
    representation_step, global_step = train_phase(
        "readout" if args.training_stage == "readout" else "representation",
        args.representation_steps,
        representation_step,
        global_step,
        model,
        wrapped,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        weights,
        experiment_tracker,
        feature_runtime,
    )
    joint_step, global_step = train_phase(
        (
            "posterior"
            if args.training_stage == "posterior"
            else "prior" if args.training_stage == "prior" else "joint"
        ),
        args.joint_steps,
        joint_step,
        global_step,
        model,
        wrapped,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        weights,
        experiment_tracker,
        feature_runtime,
    )
    return representation_step, joint_step, global_step
