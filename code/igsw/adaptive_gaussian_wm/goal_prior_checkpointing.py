"""Strict delta checkpoints for image-goal Prior training."""
from __future__ import annotations

import hashlib
import json
import os

import torch

from .goal_conditioning import GOAL_CONDITION_VERSION


GOAL_PRIOR_CHECKPOINT_VERSION = 3
GOAL_PRIOR_CHECKPOINT_KIND = "visual_sequence_image_goal_prior"


def file_sha256(path: str, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(state: dict, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(state, temporary)
    os.replace(temporary, path)
    latest = os.path.join(os.path.dirname(path), "latest.pt")
    temporary_link = f"{latest}.tmp.{os.getpid()}"
    if os.path.lexists(temporary_link):
        os.unlink(temporary_link)
    os.symlink(os.path.basename(path), temporary_link)
    os.replace(temporary_link, latest)


def save_goal_prior_checkpoint(
    path: str,
    model,
    conditioner,
    active_model_parameters: list[str],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args,
    step: int,
    source_sha256: str,
    rng_states: list[dict],
) -> None:
    model_state = model.state_dict()
    missing = sorted(set(active_model_parameters).difference(model_state))
    if missing:
        raise ValueError(f"active model parameters missing from state dict: {missing}")
    state = {
        "checkpoint_version": GOAL_PRIOR_CHECKPOINT_VERSION,
        "checkpoint_kind": GOAL_PRIOR_CHECKPOINT_KIND,
        "goal_condition_version": GOAL_CONDITION_VERSION,
        "parallelism": "ddp_delta_state_dict",
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_model_config": model.config.to_dict(),
        "model_delta": {
            name: model_state[name].detach().cpu()
            for name in active_model_parameters
        },
        "goal_conditioner": {
            name: value.detach().cpu()
            for name, value in conditioner.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "active_model_parameters": active_model_parameters,
        "global_step": step,
        "world_size": len(rng_states),
        "rng_states": rng_states,
    }
    _atomic_save(state, path)


def _normalized_argument(args, name: str):
    value = getattr(args, name)
    if name in ("checkpoint", "data") and value:
        return os.path.abspath(value)
    return value


def validate_goal_prior_resume(
    checkpoint: dict,
    args,
    source_sha256: str,
    active_model_parameters: list[str],
    world_size: int,
) -> None:
    required = {
        "checkpoint_version",
        "checkpoint_kind",
        "goal_condition_version",
        "parallelism",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "model_delta",
        "goal_conditioner",
        "optimizer",
        "scheduler",
        "args",
        "active_model_parameters",
        "global_step",
        "world_size",
        "rng_states",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"goal Prior resume is missing sections: {missing}")
    if checkpoint["checkpoint_version"] != GOAL_PRIOR_CHECKPOINT_VERSION:
        raise ValueError("goal Prior resume checkpoint version differs")
    if checkpoint["checkpoint_kind"] != GOAL_PRIOR_CHECKPOINT_KIND:
        raise ValueError("resume checkpoint is not an image-goal Prior delta")
    if checkpoint["goal_condition_version"] != GOAL_CONDITION_VERSION:
        raise ValueError("goal conditioner contract version differs")
    if checkpoint["parallelism"] != "ddp_delta_state_dict":
        raise ValueError("goal Prior resume requires DDP delta state")
    if (
        checkpoint["world_size"] != world_size
        or len(checkpoint["rng_states"]) != world_size
    ):
        raise ValueError("goal Prior resume world size differs")
    source_path = os.path.abspath(args.checkpoint)
    if checkpoint["source_checkpoint"] != source_path:
        raise ValueError("goal Prior resume points to a different base checkpoint")
    if checkpoint["source_checkpoint_sha256"] != source_sha256:
        raise ValueError("goal Prior base checkpoint SHA256 differs")
    if checkpoint["active_model_parameters"] != active_model_parameters:
        raise ValueError("goal Prior trainable parameter set differs")

    immutable = (
        "checkpoint",
        "data",
        "steps",
        "batch",
        "grad_accum",
        "max_train_items",
        "history_frames",
        "future_frames",
        "sequence_anchors",
        "sequence_data_sha256",
        "lr",
        "lr_floor",
        "warmup_fraction",
        "weight_decay",
        "effect_weight",
        "goal_anchor_weight",
        "goal_rank_weight",
        "goal_relative_margin",
        "action_activity_floor",
        "seed",
        "amp",
    )
    saved = checkpoint["args"]
    mismatches = {}
    for name in immutable:
        current = _normalized_argument(args, name)
        previous = saved.get(name)
        if name in ("checkpoint", "data") and previous:
            previous = os.path.abspath(previous)
        if previous != current:
            mismatches[name] = {
                "checkpoint": previous,
                "current": current,
            }
    if mismatches:
        raise ValueError(
            "goal Prior resume arguments differ: "
            + json.dumps(mismatches, sort_keys=True)
        )


def load_goal_prior_delta(
    checkpoint: dict,
    model,
    conditioner,
    active_model_parameters: list[str],
) -> None:
    delta = checkpoint["model_delta"]
    if sorted(delta) != active_model_parameters:
        raise ValueError("goal Prior model delta keys differ from active parameters")
    model_state = model.state_dict()
    with torch.no_grad():
        for name in active_model_parameters:
            source = delta[name]
            target = model_state.get(name)
            if target is None or target.shape != source.shape:
                raise ValueError(f"goal Prior delta tensor mismatch: {name}")
            target.copy_(source)
    conditioner.load_state_dict(checkpoint["goal_conditioner"], strict=True)
