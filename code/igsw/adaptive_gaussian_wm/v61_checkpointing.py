"""Strict full-state checkpoint contract for v61 comparison runs."""

from __future__ import annotations

import json
import os
import random

import torch
import torch.distributed as dist

from .v61_config import ARCHITECTURE, CHECKPOINT_VERSION, STAGE


def collect_rng_states(context) -> list[dict]:
    local = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(torch.device(context.device)),
        "python": random.getstate(),
    }
    if not context.distributed:
        return [local]
    states: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(states, local)
    if any(state is None for state in states):
        raise RuntimeError("failed to gather v61 RNG states")
    return [state for state in states if state is not None]


def restore_rng_state(checkpoint: dict, context) -> None:
    states = checkpoint["rng_states"]
    if len(states) != context.world_size:
        raise ValueError("v61 resume GPU count differs")
    state = states[context.rank]
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device=torch.device(context.device))
    random.setstate(state["python"])


def validate_resume(checkpoint: dict, args, world_size: int, config) -> None:
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STAGE,
        "parallelism": "ddp_full_state_dict",
        "git_commit": args.git_commit,
        "world_size": world_size,
        "config": config.to_dict(),
    }
    differences = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if differences:
        raise ValueError(f"v61 resume header differs: {differences}")
    saved = checkpoint.get("args", {})
    immutable = (
        "variant",
        "data_index",
        "chunk_lengths",
        "temporal_step_ms",
        "held_group_stride",
        "batch",
        "grad_accum",
        "target_global_batch",
        "steps",
        "backbone_lr",
        "head_lr",
        "lr_floor_ratio",
        "weight_decay",
        "warmup_steps",
        "seed",
        "amp",
        "dino_checkpoint",
        "siglip_checkpoint",
        "tracker_checkpoint",
    )
    argument_differences = {
        name: (saved.get(name), getattr(args, name))
        for name in immutable
        if saved.get(name) != getattr(args, name)
    }
    if argument_differences:
        raise ValueError(f"v61 resume arguments differ: {argument_differences}")


def save_checkpoint(
    path, model, optimizer, scheduler, args, global_step, rng_states, checkpoint_kind
):
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STAGE,
        "parallelism": "ddp_full_state_dict",
        "git_commit": args.git_commit,
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "args": vars(args),
        "global_step": int(global_step),
        "world_size": len(rng_states),
        "rng_states": rng_states,
        "checkpoint_kind": checkpoint_kind,
        "historical_checkpoint_used": False,
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
    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STAGE,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": os.path.abspath(path),
        "global_step": int(global_step),
        "world_size": len(rng_states),
        "git_commit": args.git_commit,
        "size_bytes": os.path.getsize(path),
    }
    with open(
        os.path.join(os.path.dirname(path), "checkpoint_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
