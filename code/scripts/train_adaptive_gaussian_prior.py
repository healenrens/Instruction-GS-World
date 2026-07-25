"""Train the history-and-instruction latent-action Prior on real causal pairs."""
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
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.instruction_groups import (  # noqa: E402
    build_condition_task_bank,
    build_paraphrase_index_bank,
)
from igsw.adaptive_gaussian_wm.real_prior_objective import (  # noqa: E402
    RealPriorObjective,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    cosine_schedule,
    move_to_device,
    reduce_metrics,
    validate_data_model_contract,
)
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


PRIOR_CHECKPOINT_KIND = "real_history_instruction_prior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_floor", type=float, default=1e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--effect_weight", type=float, default=0.5)
    parser.add_argument("--instruction_anchor_weight", type=float, default=0.0)
    parser.add_argument("--instruction_rank_weight", type=float, default=0.0)
    parser.add_argument("--instruction_relative_margin", type=float, default=0.05)
    parser.add_argument("--action_activity_floor", type=float, default=1.0)
    parser.add_argument("--train_language_projector", action="store_true")
    parser.add_argument("--token_conditioner_only", action="store_true")
    parser.add_argument("--freeze_token_conditioner", action="store_true")
    parser.add_argument("--dynamics_relative_objective", action="store_true")
    parser.add_argument("--paraphrase_positive_weight", type=float, default=0.0)
    parser.add_argument("--task_semantic_contrast_weight", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _active_prior_parameters(
    model,
    train_language_projector: bool,
    token_conditioner_only: bool,
    freeze_token_conditioner: bool,
) -> list[nn.Parameter]:
    if token_conditioner_only and freeze_token_conditioner:
        raise ValueError("token conditioner cannot be trained and frozen")
    if token_conditioner_only:
        conditioner = model.latent_actions.prior_token_conditioner
        if conditioner is None:
            raise ValueError(
                "token-conditioner-only training requires token Prior"
            )
        if train_language_projector:
            raise ValueError(
                "token-conditioner-only cannot train language projector"
            )
        candidates = list(conditioner.parameters())
    else:
        candidates = [
            *model.latent_actions.prior.parameters(),
            *model.latent_actions.prior_condition_parameters(),
        ]
        if model.language_effect_alignment is not None:
            candidates.extend(model.language_effect_alignment.parameters())
        if train_language_projector and model.language_condition is not None:
            candidates.extend(model.language_condition.parameters())
        if freeze_token_conditioner:
            conditioner = model.latent_actions.prior_token_conditioner
            if conditioner is None:
                raise ValueError("model has no token conditioner to freeze")
            frozen = {id(parameter) for parameter in conditioner.parameters()}
            candidates = [
                parameter
                for parameter in candidates
                if id(parameter) not in frozen
            ]
    unique = {id(parameter): parameter for parameter in candidates}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in unique)
    return list(unique.values())


def _save_checkpoint(
    path: str,
    model: AdaptiveGaussianObjectWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    step: int,
) -> None:
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": PRIOR_CHECKPOINT_KIND,
        "parallelism": "ddp_full_state_dict",
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "model": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "args": vars(args),
        "active_model_parameters": sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        "global_step": step,
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(state, temporary)
    os.replace(temporary, path)
    latest = os.path.join(os.path.dirname(path), "latest.pt")
    temporary_link = f"{latest}.tmp.{os.getpid()}"
    if os.path.lexists(temporary_link):
        os.unlink(temporary_link)
    os.symlink(os.path.basename(path), temporary_link)
    os.replace(temporary_link, latest)


def _validate_resume(checkpoint: dict, args: argparse.Namespace) -> None:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("prior resume requires the current checkpoint version")
    if checkpoint.get("checkpoint_kind") != PRIOR_CHECKPOINT_KIND:
        raise ValueError("resume checkpoint is not a real-data Prior checkpoint")
    saved = checkpoint["args"]
    immutable = (
        "checkpoint",
        "data",
        "dino",
        "condition_cache",
        "condition_feature_sha256",
        "condition_token_sha256",
        "steps",
        "batch",
        "grad_accum",
        "max_train_items",
        "lr",
        "lr_floor",
        "warmup_fraction",
        "weight_decay",
        "effect_weight",
        "instruction_anchor_weight",
        "instruction_rank_weight",
        "instruction_relative_margin",
        "action_activity_floor",
        "train_language_projector",
        "token_conditioner_only",
        "freeze_token_conditioner",
        "dynamics_relative_objective",
        "paraphrase_positive_weight",
        "paraphrase_bank_coverage",
        "task_semantic_contrast_weight",
        "seed",
        "amp",
    )
    mismatches = {
        name: (saved.get(name), getattr(args, name))
        for name in immutable
        if saved.get(name) != getattr(args, name)
    }
    if mismatches:
        raise ValueError(f"prior resume arguments differ: {mismatches}")


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch <= 0 or args.grad_accum <= 0:
        raise ValueError("steps, batch, and grad_accum must be positive")
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("lr_floor must be in (0, lr]")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if args.effect_weight < 0.0:
        raise ValueError("effect_weight must be non-negative")
    if args.instruction_anchor_weight < 0.0:
        raise ValueError("instruction_anchor_weight must be non-negative")
    if args.instruction_rank_weight < 0.0:
        raise ValueError("instruction_rank_weight must be non-negative")
    if args.instruction_relative_margin < 0.0:
        raise ValueError("instruction_relative_margin must be non-negative")
    if not 0.0 < args.action_activity_floor <= 1.0:
        raise ValueError("action_activity_floor must be in (0, 1]")
    if args.paraphrase_positive_weight < 0.0:
        raise ValueError("paraphrase positive weight must be non-negative")
    if args.paraphrase_positive_weight and not args.dynamics_relative_objective:
        raise ValueError("paraphrase positives require relative Dynamics loss")
    if args.paraphrase_positive_weight and not args.effect_weight:
        raise ValueError("paraphrase positives require effect loss")
    if args.task_semantic_contrast_weight < 0.0:
        raise ValueError("task semantic contrast weight must be non-negative")
    context = init_torchrun()
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)

    source = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if source.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Prior training requires the current checkpoint version")
    config = AdaptiveGaussianWMConfig(**source["config"])
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        "train",
        max_items=args.max_train_items,
        condition_cache=args.condition_cache,
        load_rgb=config.rgb_semantic_action,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    if dataset.condition_store is None:
        raise ValueError("Prior training requires cached instruction features")
    args.condition_feature_sha256 = dataset.condition_store.feature_sha256
    args.condition_token_sha256 = (
        dataset.condition_store.token_feature_sha256
    )
    condition_task_bank = build_condition_task_bank(
        dataset.all_paths,
        dataset.condition_store,
    )
    paraphrase_index = build_paraphrase_index_bank(condition_task_bank)
    observed = condition_task_bank >= 0
    args.paraphrase_bank_coverage = float(
        (paraphrase_index[observed] >= 0).float().mean()
    )
    assert_same_paths(dataset.paths, context, "Prior training pairs")
    validate_data_model_contract(config, dataset, True, True)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(source["model"], strict=True)
    model.eval()
    parameters = _active_prior_parameters(
        model,
        args.train_language_projector,
        args.token_conditioner_only,
        args.freeze_token_conditioner,
    )
    objective = RealPriorObjective(
        model,
        args.effect_weight,
        args.instruction_anchor_weight,
        args.instruction_rank_weight,
        args.instruction_relative_margin,
        args.action_activity_floor,
        args.dynamics_relative_objective,
        args.paraphrase_positive_weight,
        dataset.condition_store.features,
        dataset.condition_store.token_features,
        dataset.condition_store.token_valid,
        paraphrase_index,
        condition_task_bank,
        args.task_semantic_contrast_weight,
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
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
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
        raise ValueError("not enough batches for one Prior optimizer step")
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
        _validate_resume(resume, args)
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        step = int(resume["global_step"])

    if context.is_main:
        print(
            f"[prior] world={context.world_size} "
            f"effective_batch={args.batch * context.world_size * args.grad_accum} "
            f"pairs={len(dataset)} action_anchor="
            f"{'object_slot' if config.object_aligned_actions else 'global'} "
            f"action_dim={config.action_dim} "
            f"action_residual_dim={config.action_residual_dim} "
            f"action_residual_gate={config.action_residual_gate} "
            f"action_residual_dropout={config.action_residual_dropout} "
            f"semantic_action_basis={'learned' if config.learned_semantic_action_basis else 'fixed'} "
            f"rgb_action_targets={config.rgb_semantic_action} "
            f"language_effect_weight={config.language_effect_weight} "
            f"effect_weight={args.effect_weight} "
            f"instruction_anchor_weight={args.instruction_anchor_weight} "
            f"instruction_rank_weight={args.instruction_rank_weight} "
            f"instruction_relative_margin={args.instruction_relative_margin} "
            f"action_activity_floor={args.action_activity_floor} "
            f"train_language_projector={args.train_language_projector} "
            f"token_conditioner_only={args.token_conditioner_only} "
            f"freeze_token_conditioner={args.freeze_token_conditioner}",
            f"dynamics_relative_objective={args.dynamics_relative_objective}",
            f"paraphrase_positive_weight={args.paraphrase_positive_weight}",
            f"paraphrase_bank_coverage={args.paraphrase_bank_coverage}",
            f"task_semantic_contrast_weight="
            f"{args.task_semantic_contrast_weight}",
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
            if batch_index >= usable_batches or batch_index < skip_batches:
                continue
            batch = move_to_device(cpu_batch, device)
            synchronize = micro_step + 1 == args.grad_accum
            sync_context = (
                nullcontext()
                if synchronize or not context.distributed
                else wrapped.no_sync()
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
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
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
            if context.is_main and (step == 1 or step % args.log_every == 0):
                elapsed = time.time() - started
                record = {
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": step / max(elapsed, 1e-6),
                    "peak_memory_gb": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    ),
                    "memory_headroom_fraction": (
                        1.0
                        - torch.cuda.max_memory_allocated(device)
                        / torch.cuda.get_device_properties(device).total_memory
                    ),
                    **metrics,
                }
                with open(
                    os.path.join(args.out, "train.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            if context.is_main and (
                step % args.save_every == 0 or step == args.steps
            ):
                _save_checkpoint(
                    os.path.join(args.out, f"prior_{step:07d}.pt"),
                    model,
                    optimizer,
                    scheduler,
                    args,
                    step,
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
