"""Checkpoint contracts for frozen-state v61 carrier Dynamics."""

from __future__ import annotations

import json
import os

import torch

from .v61_config import (
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    DYNAMICS_ARCHITECTURE,
    DYNAMICS_STAGE,
    STAGE,
)


def validate_state_checkpoint_v61(
    checkpoint: dict, config, source_revision: str, held_group_stride: int
):
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STAGE,
        "git_commit": source_revision,
        "config": config.to_dict(),
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if differences:
        raise ValueError(f"v61 Dynamics state checkpoint differs: {differences}")
    if checkpoint.get("args", {}).get("held_group_stride") != held_group_stride:
        raise ValueError("v61 state and Dynamics held-group partitions differ")


def validate_dynamics_resume_v61(checkpoint, args, world_size, config):
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": DYNAMICS_ARCHITECTURE,
        "stage": DYNAMICS_STAGE,
        "parallelism": "ddp_full_state_dict",
        "git_commit": args.git_commit,
        "world_size": world_size,
        "config": config.to_dict(),
        "state_checkpoint": args.state_checkpoint,
        "effect_capacity": args.effect_capacity,
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if differences:
        raise ValueError(f"v61 Dynamics resume header differs: {differences}")
    immutable = (
        "variant",
        "effect_capacity",
        "data_index",
        "chunk_lengths",
        "temporal_step_ms",
        "held_group_stride",
        "batch",
        "grad_accum",
        "target_global_batch",
        "steps",
        "lr",
        "weight_decay",
        "warmup_steps",
        "seed",
        "amp",
        "dino_checkpoint",
        "siglip2_checkpoint",
        "tracker_checkpoint",
    )
    saved = checkpoint.get("args", {})
    argument_differences = {
        name: (saved.get(name), getattr(args, name))
        for name in immutable
        if saved.get(name) != getattr(args, name)
    }
    if argument_differences:
        raise ValueError(
            f"v61 Dynamics resume arguments differ: {argument_differences}"
        )


def save_dynamics_checkpoint_v61(
    path, model, optimizer, scheduler, args, step, rng_states, checkpoint_kind
):
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": DYNAMICS_ARCHITECTURE,
        "stage": DYNAMICS_STAGE,
        "parallelism": "ddp_full_state_dict",
        "git_commit": args.git_commit,
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "args": vars(args),
        "global_step": int(step),
        "world_size": len(rng_states),
        "rng_states": rng_states,
        "checkpoint_kind": checkpoint_kind,
        "state_checkpoint": args.state_checkpoint,
        "effect_capacity": args.effect_capacity,
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
        "architecture": DYNAMICS_ARCHITECTURE,
        "stage": DYNAMICS_STAGE,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": os.path.abspath(path),
        "global_step": int(step),
        "world_size": len(rng_states),
        "git_commit": args.git_commit,
        "size_bytes": os.path.getsize(path),
        "state_checkpoint": args.state_checkpoint,
        "effect_capacity": args.effect_capacity,
    }
    with open(
        os.path.join(os.path.dirname(path), "checkpoint_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
