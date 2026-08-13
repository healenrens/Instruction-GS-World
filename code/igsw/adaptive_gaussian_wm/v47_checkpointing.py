"""Strict v47 checkpoint and exact-resume contract."""

from __future__ import annotations

import json
import os
import random

import torch
import torch.distributed as dist

from .v47_config import ARCHITECTURE, CHECKPOINT_VERSION


def collect_rng_states(context) -> list[dict]:
    device = torch.device(context.device)
    local = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
    }
    if not context.distributed:
        return [local]
    states: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(states, local)
    if any(state is None for state in states):
        raise RuntimeError("failed to gather v47 RNG states")
    return [state for state in states if state is not None]


def restore_rng_state(checkpoint: dict, context) -> None:
    states = checkpoint["rng_states"]
    if len(states) != context.world_size:
        raise ValueError("v47 resume RNG count differs from world size")
    state = states[context.rank]
    torch.set_rng_state(state["torch"])
    if torch.device(context.device).type == "cuda":
        if state["cuda"] is None:
            raise ValueError("v47 resume has no CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"], device=torch.device(context.device))
    random.setstate(state["python"])


def validate_resume(checkpoint: dict, args, world_size: int, config) -> None:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v47 resume requires a version-47 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v47 resume architecture differs")
    if checkpoint.get("parallelism") != "ddp_full_state_dict":
        raise ValueError("v47 resume requires a DDP full state dict")
    if checkpoint.get("world_size") != world_size:
        raise ValueError("v47 resume world size differs")
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("v47 resume model/curriculum config differs")
    saved = checkpoint.get("args", {})
    immutable = (
        "data", "chunk_lengths", "temporal_strides",
        "observation_mask_probability", "batch", "grad_accum",
        "target_global_batch", "workers", "prefetch_factor",
        "dino_frame_batch", "dino_checkpoint", "source_revision",
        "steps", "lr", "lr_floor", "weight_decay",
        "warmup_steps", "seed", "amp",
    )
    differences = {
        name: (saved.get(name, "<missing>"), getattr(args, name))
        for name in immutable
        if saved.get(name, "<missing>") != getattr(args, name)
    }
    if differences:
        raise ValueError(f"v47 resume-critical arguments differ: {differences}")


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    args,
    global_step: int,
    rng_states: list[dict],
    checkpoint_kind: str,
) -> dict:
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "parallelism": "ddp_full_state_dict",
        "git_commit": args.git_commit,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "args": vars(args),
        "global_step": int(global_step),
        "world_size": len(rng_states),
        "rng_states": rng_states,
        "checkpoint_kind": checkpoint_kind,
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
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
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": os.path.abspath(path),
        "global_step": int(global_step),
        "world_size": len(rng_states),
        "git_commit": args.git_commit,
        "size_bytes": os.path.getsize(path),
    }
    manifest_path = os.path.join(os.path.dirname(path), "checkpoint_manifest.json")
    temporary_manifest = f"{manifest_path}.tmp.{os.getpid()}"
    with open(temporary_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_manifest, manifest_path)
    return manifest
