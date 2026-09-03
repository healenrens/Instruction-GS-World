"""Full-state checkpoint contract for continuous predictive object field v67."""

from __future__ import annotations

import json
import os
import random

import torch
import torch.distributed as dist

from .v67_config import ARCHITECTURE, CHECKPOINT_VERSION, STATE_STAGE


def collect_rng_states_v67(context):
    local = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(torch.device(context.device)),
        "python": random.getstate(),
    }
    if not context.distributed:
        return [local]
    states = [None] * context.world_size
    dist.all_gather_object(states, local)
    return states


def restore_rng_state_v67(checkpoint, context) -> None:
    state = checkpoint["rng_states"][context.rank]
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device=torch.device(context.device))
    random.setstate(state["python"])


def load_predictive_state_checkpoint_v67(path: str, config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STATE_STAGE,
        "config": config.to_dict(),
        "historical_checkpoint_used": False,
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if differences:
        raise ValueError(f"v67 predictive-state checkpoint differs: {differences}")
    return checkpoint


def validate_resume_v67(checkpoint, args, context, config) -> None:
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "parallelism": "ddp_full_state_dict",
        "world_size": context.world_size,
        "config": config.to_dict(),
        "historical_checkpoint_used": False,
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if differences:
        raise ValueError(f"v67 resume header differs: {differences}")
    revisions = {args.source_revision}
    if args.resume_compatible_source_revision:
        revisions.add(args.resume_compatible_source_revision)
    if checkpoint.get("git_commit") not in revisions:
        raise ValueError("v67 resume source revision differs")
    immutable = (
        "stage",
        "data_index",
        "out",
        "chunk_lengths",
        "temporal_step_ms",
        "held_group_stride",
        "batch",
        "grad_accum",
        "target_global_batch",
        "max_train_items",
        "steps",
        "lr",
        "lr_floor_ratio",
        "weight_decay",
        "warmup_steps",
        "seed",
        "amp",
        "dino_checkpoint",
        "siglip_checkpoint",
        "tracker_checkpoint",
        "state_checkpoint",
    )
    saved = checkpoint["args"]
    argument_differences = {
        name: (saved.get(name), getattr(args, name))
        for name in immutable
        if saved.get(name) != getattr(args, name)
    }
    if argument_differences:
        raise ValueError(f"v67 resume arguments differ: {argument_differences}")


def save_checkpoint_v67(
    path, model, optimizer, scheduler, args, step, rng_states, checkpoint_kind
):
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "parallelism": "ddp_full_state_dict",
        "git_commit": args.source_revision,
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
        "stage": args.stage,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": os.path.abspath(path),
        "global_step": int(step),
        "world_size": len(rng_states),
        "git_commit": args.source_revision,
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
