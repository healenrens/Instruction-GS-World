"""CPU checks for the pre-registered matched-flat optimization contract."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.matched_flat_training_contract import (  # noqa: E402
    CONTRACT_NAME,
    flat_training_mismatches,
    object_training_mismatches,
    training_contract_summary,
)


STEP = 12000


def object_checkpoint() -> dict:
    return {
        "global_step": STEP,
        "world_size": 4,
        "args": {
            "representation_steps": 0,
            "joint_steps": STEP,
            "batch": 2,
            "grad_accum": 32,
            "workers": 2,
            "lr": 5e-5,
            "lr_floor": 5e-6,
            "warmup_steps": 500,
            "warmup_fraction": 0.0,
            "weight_decay": 1e-4,
            "seed": 17,
            "amp": "bf16",
            "max_train_items": 0,
            "rgb_loss_weight": 0.5,
            "rgb_ssim_weight": 0.2,
            "rgb_change_loss_weight": 1.0,
            "rgb_change_threshold": 0.04,
        },
    }


def flat_checkpoint() -> dict:
    return {
        "global_step": STEP,
        "world_size": 4,
        "initialization": {"kind": "object_dynamics_blocks_only"},
        "args": {
            "steps": STEP,
            "batch": 2,
            "grad_accum": 32,
            "workers": 2,
            "gradient_checkpointing": "on",
            "lr": 2e-4,
            "lr_floor": 2e-5,
            "warmup_steps": 600,
            "weight_decay": 1e-4,
            "change_loss_weight": 1.0,
            "history_loss_weight": 1.0,
            "rgb_loss_weight": 0.5,
            "rgb_ssim_weight": 0.2,
            "rgb_change_loss_weight": 1.0,
            "rgb_change_threshold": 0.04,
            "seed": 17,
            "amp": "bf16",
            "max_train_items": 0,
            "modality_matching": "dino_rgb",
        },
    }


def main() -> None:
    object_state = object_checkpoint()
    flat_state = flat_checkpoint()
    if object_training_mismatches(object_state, STEP):
        raise AssertionError("valid object training contract failed")
    if flat_training_mismatches(flat_state, STEP):
        raise AssertionError("valid flat training contract failed")
    summary = training_contract_summary(object_state, flat_state, STEP)
    if summary["optimizer_contract"] != CONTRACT_NAME:
        raise AssertionError("training summary omitted the contract identity")
    corruptions = {
        "object_lr": ("object", "lr", 1e-4),
        "flat_lr": ("flat", "lr", 5e-5),
        "flat_batch": ("flat", "batch", 1),
        "flat_seed": ("flat", "seed", 18),
        "flat_world_size": ("flat", "world_size", 8),
    }
    detected = []
    for name, (side, field, value) in corruptions.items():
        state = deepcopy(object_state if side == "object" else flat_state)
        if field == "world_size":
            state[field] = value
        else:
            state["args"][field] = value
        mismatches = (
            object_training_mismatches(state, STEP)
            if side == "object"
            else flat_training_mismatches(state, STEP)
        )
        if field not in mismatches:
            raise AssertionError(f"contract corruption was not detected: {name}")
        detected.append(name)
    print(json.dumps({
        "status": "ok",
        "contract": CONTRACT_NAME,
        "detected_corruptions": detected,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
