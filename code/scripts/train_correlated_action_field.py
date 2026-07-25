"""Two-stage DDP training for the strict-causal correlated action world model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402
from igsw.latent_particle_wm.action_field import (  # noqa: E402
    ActionFieldConfig,
    CorrelatedActionField,
)
from igsw.latent_particle_wm.action_objectives import (  # noqa: E402
    ActionLossWeights,
    posterior_joint_loss,
)
from igsw.latent_particle_wm.pair_data import CausalPairDataset  # noqa: E402


_RESUME_MUTABLE_ARGS = {"log_every", "out", "resume", "save_every", "workers"}
_PATH_ARGS = {"data", "dino"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--posterior_steps", type=int, default=1000)
    parser.add_argument("--prior_steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--control_rows", type=int, default=16)
    parser.add_argument("--control_cols", type=int, default=16)
    parser.add_argument("--active_count", type=int, default=192)
    parser.add_argument("--dino_dim", type=int, default=32)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--render_height", type=int, default=98)
    parser.add_argument("--render_width", type=int, default=130)
    parser.add_argument("--posterior_lr", type=float, default=3e-4)
    parser.add_argument("--prior_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--w_move", type=float, default=20.0)
    parser.add_argument("--w_deterministic", type=float, default=0.25)
    parser.add_argument("--w_appearance", type=float, default=0.2)
    parser.add_argument("--w_visibility", type=float, default=0.1)
    parser.add_argument("--w_rgb", type=float, default=0.5)
    parser.add_argument("--w_dino", type=float, default=0.1)
    parser.add_argument("--w_effect", type=float, default=0.1)
    parser.add_argument("--w_action_alignment", type=float, default=1.0)
    parser.add_argument("--w_usage", type=float, default=0.2)
    parser.add_argument("--usage_margin", type=float, default=0.01)
    parser.add_argument("--w_local", type=float, default=0.1)
    parser.add_argument("--w_alpha", type=float, default=0.01)
    parser.add_argument("--w_slot", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        reduced[key] = float(tensor)
    return reduced


def collect_rng_states(context) -> list[dict]:
    local_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device=context.device),
        "python": random.getstate(),
    }
    if not context.distributed:
        return [local_state]
    states: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("failed to gather RNG state from every rank")
    return [state for state in states if state is not None]


def restore_rng_state(resume_state: dict, context) -> None:
    if "rng_states" in resume_state:
        states = resume_state["rng_states"]
        if len(states) != context.world_size:
            raise ValueError(
                "resume checkpoint world size differs from current world size: "
                f"{len(states)} != {context.world_size}"
            )
        state = states[context.rank]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], device=context.device)
        random.setstate(state["python"])
        return

    if context.distributed:
        raise ValueError(
            "legacy checkpoint has no per-rank RNG states; exact distributed resume "
            "is unsupported"
        )
    torch.set_rng_state(resume_state["torch_rng_state"])
    torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])
    random.setstate(resume_state["python_rng_state"])


def validate_resume_args(saved: dict, current: argparse.Namespace) -> None:
    mismatches = {}
    for name, current_value in vars(current).items():
        if name in _RESUME_MUTABLE_ARGS:
            continue
        if name not in saved:
            mismatches[name] = {"checkpoint": "<missing>", "current": current_value}
            continue
        saved_value = saved[name]
        if name in _PATH_ARGS:
            saved_value = os.path.abspath(saved_value)
            current_value = os.path.abspath(current_value)
        if saved_value != current_value:
            mismatches[name] = {
                "checkpoint": saved_value,
                "current": current_value,
            }
    if mismatches:
        raise ValueError(
            "resume-critical arguments differ from checkpoint: "
            + json.dumps(mismatches, sort_keys=True)
        )


def save_checkpoint(
    path: str,
    model: CorrelatedActionField,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    phase: str,
    posterior_step: int,
    prior_step: int,
    rng_states: list[dict],
) -> None:
    state = {
        "checkpoint_version": 2,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "args": vars(args),
        "phase": phase,
        "posterior_step": posterior_step,
        "prior_step": prior_step,
        "world_size": len(rng_states),
        "rng_states": rng_states,
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


def train_phase(
    phase: str,
    model: CorrelatedActionField,
    wrapped,
    loader: DataLoader,
    loader_generator: torch.Generator,
    sampler: DistributedSampler,
    context,
    args: argparse.Namespace,
    start_step: int,
    total_steps: int,
    posterior_step: int,
    prior_step: int,
    resume_state: dict | None,
) -> tuple[int, int]:
    if start_step >= total_steps:
        return posterior_step, prior_step
    model.set_training_phase(phase)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = args.posterior_lr if phase == "posterior" else args.prior_lr
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule(optimizer, total_steps, args.warmup_steps)
    if resume_state and resume_state["phase"] == phase and start_step < total_steps:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    device = torch.device(context.device)
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp == "bf16"
        else nullcontext()
    )
    step = start_step
    optimizer_steps_per_epoch = len(loader) // args.grad_accum
    initial_epoch = start_step // optimizer_steps_per_epoch
    initial_skip = start_step % optimizer_steps_per_epoch
    epoch = initial_epoch
    weights = ActionLossWeights(
        move=args.w_move,
        deterministic=args.w_deterministic,
        appearance=args.w_appearance,
        visibility=args.w_visibility,
        rgb=args.w_rgb,
        dino=args.w_dino,
        effect=args.w_effect,
        action_alignment=args.w_action_alignment,
        usage=args.w_usage,
        usage_margin=args.usage_margin,
        local=args.w_local,
        alpha=args.w_alpha,
        slot=args.w_slot,
    )
    log_path = os.path.join(args.out, "train.jsonl")
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        phase_offset = 100000 if phase == "prior" else 0
        sampler.set_epoch(args.seed + epoch + phase_offset)
        loader_generator.manual_seed(
            args.seed + context.rank + epoch + phase_offset
        )
        accumulated: dict[str, torch.Tensor] = {}
        for batch_index, cpu_batch in enumerate(loader):
            optimizer_step_in_epoch = batch_index // args.grad_accum
            micro_step = batch_index % args.grad_accum + 1
            if optimizer_step_in_epoch >= optimizer_steps_per_epoch:
                break
            if epoch == initial_epoch and optimizer_step_in_epoch < initial_skip:
                continue
            batch = move_to_device(cpu_batch, device)
            synchronize = micro_step == args.grad_accum
            sync_context = (
                wrapped.no_sync()
                if context.distributed and not synchronize
                else nullcontext()
            )
            with sync_context, amp_context():
                output = wrapped(batch, phase)
                if phase == "posterior":
                    loss, parts = posterior_joint_loss(
                        model,
                        batch,
                        output,
                        weights,
                        args.render_height,
                        args.render_width,
                    )
                else:
                    loss = output["prior_loss"]
                    parts = {"loss": loss.detach(), "flow": loss.detach()}
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            for key, value in parts.items():
                accumulated[key] = accumulated.get(key, value.detach() * 0.0) + value.detach()
            if not synchronize:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            averaged = {
                key: value / args.grad_accum
                for key, value in accumulated.items()
            }
            accumulated = {}
            if phase == "posterior":
                posterior_step = step
            else:
                prior_step = step

            if step == 1 or step % args.log_every == 0 or step == total_steps:
                metrics = reduce_metrics(averaged, context.world_size)
                record = {
                    "phase": phase,
                    "step": step,
                    "total_steps": total_steps,
                    "lr": optimizer.param_groups[0]["lr"],
                    **metrics,
                }
                if context.is_main:
                    print(json.dumps(record, sort_keys=True), flush=True)
                    with open(log_path, "a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            if step % args.save_every == 0 or step == total_steps:
                rng_states = collect_rng_states(context)
                if context.is_main:
                    checkpoint_path = os.path.join(args.out, f"{phase}_{step:06d}.pt")
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        args,
                        phase,
                        posterior_step,
                        prior_step,
                        rng_states,
                    )
                if context.distributed:
                    dist.barrier()
            if step >= total_steps:
                break
        epoch += 1
    if context.distributed:
        dist.barrier()
    return posterior_step, prior_step


def main() -> None:
    args = parse_args()
    if args.grad_accum < 1 or args.batch < 1:
        raise ValueError("batch and grad_accum must be positive")
    context = init_torchrun()
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.out, exist_ok=True)

    dataset = CausalPairDataset(
        args.data,
        "train",
        args.control_rows,
        args.control_cols,
        args.active_count,
        args.dino,
    )
    missing_sidecars = [
        path
        for path in dataset.paths
        if not os.path.exists(os.path.join(args.dino, os.path.basename(path)))
    ]
    if missing_sidecars:
        raise FileNotFoundError(f"missing DINO sidecars: {missing_sidecars[:5]}")
    assert_same_paths(dataset.paths, context, "training pair paths")
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        drop_last=True,
    )
    loader_generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        generator=loader_generator,
    )
    if len(loader) < args.grad_accum:
        raise ValueError("not enough batches for one optimizer step")

    config = ActionFieldConfig(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        action_dim=args.action_dim,
        dino_dim=args.dino_dim,
        control_rows=args.control_rows,
        control_cols=args.control_cols,
        flow_steps=args.flow_steps,
    )
    model = CorrelatedActionField(config).to(device)
    resume_state = None
    posterior_step = prior_step = 0
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_args(resume_state["args"], args)
        resume_config = ActionFieldConfig(**resume_state["config"]).to_dict()
        if resume_config != config.to_dict():
            raise ValueError("resume checkpoint model config differs from command-line config")
        model.load_state_dict(resume_state["model"], strict=True)
        posterior_step = int(resume_state["posterior_step"])
        prior_step = int(resume_state["prior_step"])
        if posterior_step > args.posterior_steps or prior_step > args.prior_steps:
            raise ValueError("resume steps exceed requested training steps")
        if resume_state["phase"] == "prior" and posterior_step < args.posterior_steps:
            raise ValueError("cannot extend posterior training from a prior-stage checkpoint")
        restore_rng_state(resume_state, context)
    model.set_training_phase("all")
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if context.distributed
        else model
    )

    posterior_step, prior_step = train_phase(
        "posterior",
        model,
        wrapped,
        loader,
        loader_generator,
        sampler,
        context,
        args,
        posterior_step,
        args.posterior_steps,
        posterior_step,
        prior_step,
        resume_state,
    )
    posterior_step, prior_step = train_phase(
        "prior",
        model,
        wrapped,
        loader,
        loader_generator,
        sampler,
        context,
        args,
        prior_step,
        args.prior_steps,
        posterior_step,
        prior_step,
        resume_state,
    )
    if context.is_main:
        print(
            f"[train-action-field] DONE posterior={posterior_step} prior={prior_step} "
            f"out={os.path.abspath(args.out)}",
            flush=True,
        )
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
