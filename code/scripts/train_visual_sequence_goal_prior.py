"""Train a language-free history, physical-time, and image-goal action Prior."""
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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
    collect_rng_states,
    restore_rng_state,
)
from igsw.adaptive_gaussian_wm.goal_conditioning import (  # noqa: E402
    ObjectGoalConditioner,
)
from igsw.adaptive_gaussian_wm.goal_prior_checkpointing import (  # noqa: E402
    file_sha256,
    load_goal_prior_delta,
    save_goal_prior_checkpoint,
    validate_goal_prior_resume,
)
from igsw.adaptive_gaussian_wm.goal_prior_objective import (  # noqa: E402
    GoalPriorObjective,
)
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    cosine_schedule,
    cuda_memory_metrics,
    move_to_device,
    reduce_metrics,
    validate_data_model_contract,
)
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--sequence_anchors", default="3,5,8")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--effect_weight", type=float, default=0.5)
    parser.add_argument("--goal_anchor_weight", type=float, default=0.5)
    parser.add_argument("--goal_rank_weight", type=float, default=0.5)
    parser.add_argument("--goal_relative_margin", type=float, default=0.05)
    parser.add_argument("--action_activity_floor", type=float, default=0.25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "steps",
        "batch",
        "grad_accum",
        "history_frames",
        "future_frames",
        "save_every",
        "log_every",
    )
    invalid = [name for name in positive if getattr(args, name) <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("lr_floor must be in (0, lr]")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    non_negative = (
        "effect_weight",
        "goal_anchor_weight",
        "goal_rank_weight",
        "goal_relative_margin",
    )
    invalid = [name for name in non_negative if getattr(args, name) < 0.0]
    if invalid:
        raise ValueError(f"arguments must be non-negative: {invalid}")
    if not 0.0 < args.action_activity_floor <= 1.0:
        raise ValueError("action_activity_floor must be in (0, 1]")


def _source_digest(path: str, context) -> str:
    digest = file_sha256(path) if context.is_main else None
    if context.distributed:
        values = [digest]
        dist.broadcast_object_list(values, src=0)
        digest = values[0]
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("failed to establish base checkpoint SHA256")
    return digest


def _active_parameters(
    model,
    conditioner: ObjectGoalConditioner,
) -> tuple[list[nn.Parameter], list[str]]:
    candidates = [
        *model.latent_actions.prior.parameters(),
        *model.latent_actions.prior_condition_parameters(),
    ]
    active = {id(parameter): parameter for parameter in candidates}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in active)
    conditioner.requires_grad_(True)
    names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    return [*active.values(), *conditioner.parameters()], names


def _build_dataset(args, config) -> CausalVisualSequenceDataset:
    dataset = CausalVisualSequenceDataset(
        args.data,
        "train",
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        anchors=args.sequence_anchors,
        max_items=args.max_train_items,
        load_rgb=config.rgb_supervision,
        explicit_goal=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(
        config,
        dataset,
        language_enabled=False,
        rgb_enabled=config.rgb_supervision,
    )
    return dataset


def _load_base(args, device: torch.device):
    source = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if source.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("goal Prior requires a current-version base checkpoint")
    source_args = source.get("args", {})
    if (
        source.get("parallelism") != "ddp_full_state_dict"
        or source.get("phase") != "joint"
        or not source_args.get("posterior_core_training", False)
        or int(source_args.get("representation_steps", -1)) != 0
        or int(source.get("phase_step", -1))
        != int(source_args.get("joint_steps", -2))
    ):
        raise ValueError("goal Prior requires a completed posterior-Core checkpoint")
    config = AdaptiveGaussianWMConfig(**source["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(source["model"], strict=True)
    model.eval()
    return model, config, source


def _make_loader(args, dataset, context):
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
    usable_batches = len(loader) // args.grad_accum * args.grad_accum
    if usable_batches == 0:
        raise ValueError("not enough batches for one goal Prior optimizer step")
    return loader, sampler, usable_batches


def main() -> None:
    args = parse_args()
    _validate_args(args)
    context = init_torchrun()
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)

    source_sha256 = _source_digest(args.checkpoint, context)
    model, config, source = _load_base(args, device)
    dataset = _build_dataset(args, config)
    args.sequence_data_sha256 = getattr(dataset, "data_sha256", "")
    if source["args"].get("sequence_data_sha256") != args.sequence_data_sha256:
        raise ValueError("goal Prior data manifest differs from Core checkpoint")
    assert_same_paths(dataset.paths, context, "image-goal Prior sequences")
    conditioner = ObjectGoalConditioner(config).to(device)
    parameters, active_names = _active_parameters(model, conditioner)
    objective = GoalPriorObjective(
        model,
        conditioner,
        args.effect_weight,
        args.goal_anchor_weight,
        args.goal_rank_weight,
        args.goal_relative_margin,
        args.action_activity_floor,
    ).to(device)

    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        if not args.resume and os.path.lexists(os.path.join(args.out, "latest.pt")):
            raise ValueError(f"output already contains a run: {args.out}")
    if context.distributed:
        dist.barrier()
    wrapped = (
        DistributedDataParallel(
            objective,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if context.distributed
        else objective
    )
    loader, sampler, usable_batches = _make_loader(args, dataset, context)
    balanced_updates = sampler.num_samples // (args.batch * args.grad_accum)
    if args.steps < balanced_updates:
        raise ValueError(
            "goal Prior steps do not cover one balanced data epoch: "
            f"{args.steps} < {balanced_updates}"
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule(
        optimizer,
        round(args.steps * args.warmup_fraction),
        args.steps,
        args.lr_floor / args.lr,
    )
    step = 0
    if args.resume:
        resume = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        validate_goal_prior_resume(
            resume,
            args,
            source_sha256,
            active_names,
            context.world_size,
        )
        load_goal_prior_delta(resume, model, conditioner, active_names)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        step = int(resume["global_step"])
        restore_rng_state(resume, context)

    if context.is_main:
        print(
            f"[goal-prior] world={context.world_size} "
            f"effective_batch={args.batch * context.world_size * args.grad_accum} "
            f"sequences={len(dataset)} action_dim={config.action_dim} "
            f"goal_time=explicit language=off "
            f"sampler={type(sampler).__name__} "
            f"balanced_updates_per_epoch={balanced_updates} "
            f"source_sha256={source_sha256}",
            flush=True,
        )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    epoch = step * args.grad_accum // usable_batches
    skip_batches = step * args.grad_accum % usable_batches
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    while step < args.steps:
        sampler.set_epoch(args.seed + epoch)
        accumulated: dict[str, torch.Tensor] = {}
        micro_step = 0
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index >= usable_batches:
                break
            if batch_index < skip_batches:
                continue
            batch = move_to_device(cpu_batch, device)
            synchronize = micro_step + 1 == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            with sync_context, amp_context():
                parts = wrapped(batch)
                loss = parts["loss"] / args.grad_accum
            loss.backward()
            for name, value in parts.items():
                accumulated[name] = accumulated.get(
                    name,
                    value.detach().new_zeros(()),
                ) + value.detach() / args.grad_accum
            micro_step += 1
            if not synchronize:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = reduce_metrics(
                accumulated
                | {
                    "gradient_norm": torch.as_tensor(
                        gradient_norm,
                        device=device,
                    ),
                    "lr": torch.tensor(
                        optimizer.param_groups[0]["lr"],
                        device=device,
                    ),
                },
                context.world_size,
            )
            if context.is_main and (
                step == 1 or step % args.log_every == 0 or step == args.steps
            ):
                record = {
                    "step": step,
                    "elapsed_seconds": time.time() - started,
                    "steps_per_second": step / max(time.time() - started, 1e-6),
                    "micro_batch": args.batch,
                    "grad_accum": args.grad_accum,
                    "world_size": context.world_size,
                    "effective_batch": (
                        args.batch * context.world_size * args.grad_accum
                    ),
                    **cuda_memory_metrics(device),
                    **metrics,
                }
                with open(
                    os.path.join(args.out, "train.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            should_save = (
                step % args.save_every == 0 or step == args.steps
            )
            if should_save:
                rng_states = collect_rng_states(context)
                if context.is_main:
                    save_goal_prior_checkpoint(
                        os.path.join(args.out, f"goal_prior_{step:07d}.pt"),
                        model,
                        conditioner,
                        active_names,
                        optimizer,
                        scheduler,
                        args,
                        step,
                        source_sha256,
                        rng_states,
                    )
                if context.distributed:
                    dist.barrier()
            if step >= args.steps:
                break
            accumulated = {}
            micro_step = 0
        epoch += 1
        skip_batches = 0
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
