"""Shared runtime contracts for object-centered held-set carrier audits."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json

import torch

from .checkpointing import CHECKPOINT_VERSION
from .config import AdaptiveGaussianWMConfig
from .dynamics_runtime import run_object_dynamics
from .model import AdaptiveGaussianObjectWorldModel
from .scale import signed_gap_scale
from .v28_training import ARCHITECTURE


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def autocast_context():
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )


def load_object_memory_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    version = int(checkpoint.get("checkpoint_version", -1))
    require(
        28 <= version <= CHECKPOINT_VERSION,
        f"checkpoint version {version} is outside [28,{CHECKPOINT_VERSION}]",
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    require(config.architecture == ARCHITECTURE, "checkpoint is not object_memory_v1")
    require(config.persistent_object_memory, "checkpoint has no object memory")
    require(config.factorized_dynamics, "checkpoint has no factorized Dynamics")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "phase": checkpoint.get("phase"),
        "phase_step": checkpoint.get("phase_step"),
        "git_commit": checkpoint.get("git_commit"),
        "training_stage": checkpoint.get("args", {}).get("training_stage"),
    }
    del checkpoint
    return metadata, model.eval()


def action_free_prediction(model, batch: dict, history: dict):
    history_scale = signed_gap_scale(batch["history_times"], model.config.gap_reference)
    future_scale = signed_gap_scale(batch["future_times"], model.config.gap_reference)
    actions = torch.zeros(
        batch["history_features"].shape[0],
        future_scale.shape[1],
        model.config.action_tokens,
        model.config.action_dim,
        device=batch["history_features"].device,
        dtype=history["slots"].dtype,
    )
    history_mask = torch.zeros(
        *history["slots"].shape[:3],
        dtype=torch.bool,
        device=history["slots"].device,
    )
    return run_object_dynamics(
        model,
        history["slots"],
        history["activity"],
        history_scale,
        future_scale,
        actions,
        history_mask,
        history["center"],
        None,
        history_relative_scale=history.get("relative_scale"),
        history_relative_disparity=history.get("relative_disparity"),
        history_relations=history.get("relations"),
        history_existence=history.get("existence"),
    )


def append_row(rows: dict[str, list[torch.Tensor]], name: str, value) -> None:
    tensor = torch.as_tensor(value).detach().float().reshape(-1).cpu()
    rows.setdefault(name, []).append(tensor)


def future_content_swap_differences(
    model,
    batch: dict,
    history: dict,
    dynamics,
) -> dict[str, float]:
    swapped = dict(batch)
    swapped["future_features"] = batch["future_features"].roll(shifts=1, dims=-2)
    input_difference = (
        (batch["future_features"].float() - swapped["future_features"].float())
        .abs()
        .max()
    )
    require(
        float(input_difference) > 1e-6,
        "future-content intervention did not change the future feature tensor",
    )
    with autocast_context():
        swapped_history = model.encode_history(swapped)
        swapped_dynamics = action_free_prediction(model, swapped, swapped_history)
    history_values = (
        (history["slots"].float() - swapped_history["slots"].float()).abs().max(),
        (history["center"].float() - swapped_history["center"].float()).abs().max(),
        (history["relative_scale"].float() - swapped_history["relative_scale"].float())
        .abs()
        .max(),
    )
    dynamics_values = (
        (dynamics.future_slots.float() - swapped_dynamics.future_slots.float())
        .abs()
        .max(),
        (dynamics.future_centers.float() - swapped_dynamics.future_centers.float())
        .abs()
        .max(),
        (
            dynamics.future_relative_scale.float()
            - swapped_dynamics.future_relative_scale.float()
        )
        .abs()
        .max(),
    )
    return {
        "future_content_swap_input_max_difference": float(input_difference),
        "history_future_content_swap_max_difference": float(
            torch.stack(history_values).max()
        ),
        "dynamics_future_content_swap_max_difference": float(
            torch.stack(dynamics_values).max()
        ),
    }


def load_v31_baseline_report(
    path: str,
    checkpoint_sha256: str,
    data_sha256: str,
    split: str,
    held_items: int,
) -> dict:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "completed",
        "contract": "object_centered_adaptive_carrier_capacity_v1",
        "checkpoint_sha256": checkpoint_sha256,
        "data_manifest_sha256": data_sha256,
        "held_split": split,
        "held_items": held_items,
    }
    mismatch = {
        name: {"report": report.get(name), "expected": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    require(not mismatch, f"v31 baseline report differs: {mismatch}")
    return report
