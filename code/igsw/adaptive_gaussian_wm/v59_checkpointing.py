"""Initialization and strict checkpoint contract for v59 transition training."""

from __future__ import annotations

import json
import os
import random

import torch

from .v58_config import (
    ARCHITECTURE as V58_ARCHITECTURE,
    CHECKPOINT_VERSION as V58_CHECKPOINT_VERSION,
)
from .v59_config import ARCHITECTURE, CHECKPOINT_VERSION, STAGE


def load_v58_encoder(model, path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("checkpoint_version") != V58_CHECKPOINT_VERSION:
        raise ValueError("v59 initialization requires a v58 checkpoint")
    if checkpoint.get("architecture") != V58_ARCHITECTURE:
        raise ValueError("v59 initialization checkpoint is not the v58 state model")
    encoder = {
        name.removeprefix("encoder."): value
        for name, value in checkpoint["model"].items()
        if name.startswith("encoder.")
    }
    result = model.encoder.load_state_dict(encoder, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("v59 did not strictly load the v58 encoder")
    return {
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "source_architecture": checkpoint["architecture"],
        "source_global_step": int(checkpoint["global_step"]),
        "source_git_commit": checkpoint.get("git_commit"),
        "loaded_encoder_tensors": len(encoder),
    }


def collect_rng_states(context):
    local = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(torch.device(context.device)),
        "python": random.getstate(),
    }
    if not context.distributed:
        return [local]
    states = [None for _ in range(context.world_size)]
    torch.distributed.all_gather_object(states, local)
    return states


def restore_rng_state(checkpoint: dict, context) -> None:
    states = checkpoint.get("rng_states")
    if not isinstance(states, list) or len(states) != context.world_size:
        raise ValueError("v59 resume RNG count differs from visible GPU count")
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
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if differences:
        raise ValueError(f"v59 resume header differs: {differences}")
    saved = checkpoint.get("args", {})
    immutable = (
        "data_index",
        "decode_report",
        "init_from",
        "chunk_lengths",
        "history_lengths",
        "teacher_future_frames",
        "temporal_step_ms",
        "batch",
        "grad_accum",
        "target_global_batch",
        "steps",
        "lr",
        "lr_floor",
        "weight_decay",
        "warmup_steps",
        "seed",
        "amp",
        "dino_checkpoint",
        "tracker_checkpoint",
    )
    changed = {
        name: (saved.get(name, "<missing>"), getattr(args, name))
        for name in immutable
        if saved.get(name, "<missing>") != getattr(args, name)
    }
    if changed:
        raise ValueError(f"v59 resume-critical arguments differ: {changed}")


def save_checkpoint(
    path, model, optimizer, scheduler, args, step, rng_states, kind, init_report
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
        "global_step": int(step),
        "world_size": len(rng_states),
        "rng_states": rng_states,
        "checkpoint_kind": kind,
        "v58_initialization": init_report,
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
        "stage": STAGE,
        "checkpoint_kind": kind,
        "checkpoint_path": os.path.abspath(path),
        "global_step": int(step),
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
