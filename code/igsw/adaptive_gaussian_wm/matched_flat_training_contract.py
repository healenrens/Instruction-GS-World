"""Pre-registered optimization contract for the matched-flat sanity baseline."""
from __future__ import annotations


CONTRACT_NAME = "posterior_core_matched_flat_optimization_v1"

OBJECT_STATIC = {
    "world_size": 4,
    "representation_steps": 0,
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
}

FLAT_STATIC = {
    "world_size": 4,
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
}


def expected_object_training(step: int) -> dict:
    return {"global_step": step, "joint_steps": step, **OBJECT_STATIC}


def expected_flat_training(step: int) -> dict:
    return {"global_step": step, "steps": step, **FLAT_STATIC}


def _observed(checkpoint: dict, expected: dict) -> dict:
    args = checkpoint.get("args", {})
    return {
        name: (
            checkpoint.get(name)
            if name in {"global_step", "world_size"}
            else args.get(name)
        )
        for name in expected
    }


def _mismatches(observed: dict, expected: dict) -> dict:
    return {
        name: {"observed": observed.get(name), "required": value}
        for name, value in expected.items()
        if observed.get(name) != value
    }


def object_training_values(checkpoint: dict, step: int) -> dict:
    return _observed(checkpoint, expected_object_training(step))


def flat_training_values(checkpoint: dict, step: int) -> dict:
    return _observed(checkpoint, expected_flat_training(step))


def object_training_mismatches(checkpoint: dict, step: int) -> dict:
    expected = expected_object_training(step)
    return _mismatches(_observed(checkpoint, expected), expected)


def flat_training_mismatches(checkpoint: dict, step: int) -> dict:
    expected = expected_flat_training(step)
    return _mismatches(_observed(checkpoint, expected), expected)


def training_contract_summary(
    object_checkpoint: dict,
    flat_checkpoint: dict,
    step: int,
) -> dict:
    object_values = object_training_values(object_checkpoint, step)
    flat_values = flat_training_values(flat_checkpoint, step)
    initialization = flat_checkpoint.get("initialization", {})
    return {
        "optimizer_contract": CONTRACT_NAME,
        "object_checkpoint_phase_steps": object_values["global_step"],
        "flat_additional_steps": flat_values["global_step"],
        "object_effective_global_batch": (
            object_values["world_size"]
            * object_values["batch"]
            * object_values["grad_accum"]
        ),
        "flat_effective_global_batch": (
            flat_values["world_size"]
            * flat_values["batch"]
            * flat_values["grad_accum"]
        ),
        "shared_dynamics_initialization": initialization.get("kind"),
        "optimization_budget_bias": (
            "flat_receives_additional_updates_after_object_source"
        ),
        "modality_matching": flat_checkpoint.get("args", {}).get(
            "modality_matching"
        ),
        "flat_rgb_supervision": flat_values["rgb_loss_weight"] > 0.0,
        "flat_posterior_observes_future_rgb": True,
        "history_encoder_input": "dino_only",
        "object_training": object_values,
        "flat_training": flat_values,
    }
