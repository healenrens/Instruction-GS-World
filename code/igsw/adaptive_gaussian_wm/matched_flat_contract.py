"""Checkpoint identity contract for object-versus-flat evaluation."""
from __future__ import annotations

import json

from .checkpointing import CHECKPOINT_VERSION as OBJECT_CHECKPOINT_VERSION
from .config import AdaptiveGaussianWMConfig
from .flat_baseline_checkpointing import (
    FLAT_CHECKPOINT_KIND,
    FLAT_CHECKPOINT_VERSION,
)
from .matched_flat_training_contract import (
    flat_training_mismatches,
    object_training_mismatches,
)


def available_samples(dataset) -> int:
    full_length = getattr(dataset, "_full_length", None)
    return len(dataset) if full_length is None else int(full_length)


def validate_object_flat_checkpoints(
    object_checkpoint: dict,
    flat_checkpoint: dict,
    object_sha256: str,
    dataset,
    required_step: int,
) -> tuple[AdaptiveGaussianWMConfig, dict]:
    if object_checkpoint.get("checkpoint_version") != OBJECT_CHECKPOINT_VERSION:
        raise ValueError("object-flat evaluation requires a current object checkpoint")
    if object_checkpoint.get("phase") != "joint":
        raise ValueError("object-flat evaluation requires a joint object checkpoint")
    if int(object_checkpoint.get("global_step", -1)) != required_step:
        raise ValueError("object checkpoint step differs from the evaluation contract")
    if flat_checkpoint.get("checkpoint_version") != FLAT_CHECKPOINT_VERSION:
        raise ValueError("flat checkpoint version differs")
    if flat_checkpoint.get("checkpoint_kind") != FLAT_CHECKPOINT_KIND:
        raise ValueError("flat checkpoint kind differs")
    if int(flat_checkpoint.get("global_step", -1)) != required_step:
        raise ValueError("flat checkpoint step differs from the evaluation contract")

    flat_args = flat_checkpoint.get("args", {})
    object_args = object_checkpoint.get("args", {})
    object_effective_batch = int(object_args.get("batch", 0)) * int(
        object_args.get("grad_accum", 0)
    ) * int(object_checkpoint.get("world_size", 0))
    flat_effective_batch = int(flat_args.get("batch", 0)) * int(
        flat_args.get("grad_accum", 0)
    ) * int(flat_checkpoint.get("world_size", 0))
    expected = {
        "object_data_sha256": object_args.get("sequence_data_sha256"),
        "flat_data_sha256": flat_args.get("data_sha256"),
        "flat_reference_sha256": flat_args.get("reference_checkpoint_sha256"),
    }
    required = {
        "object_data_sha256": dataset.data_sha256,
        "flat_data_sha256": dataset.data_sha256,
        "flat_reference_sha256": object_sha256,
    }
    mismatches = {
        name: {"checkpoint": expected[name], "required": value}
        for name, value in required.items()
        if expected[name] != value
    }
    for name, value in (
        ("history_frames", dataset.history_frames),
        ("future_frames", dataset.future_frames),
        ("sequence_anchors", ",".join(map(str, dataset.anchors))),
    ):
        if object_args.get(name) != value or flat_args.get(name) != value:
            mismatches[name] = {
                "object": object_args.get(name),
                "flat": flat_args.get(name),
                "required": value,
            }
    if object_effective_batch != 256 or flat_effective_batch != 256:
        mismatches["effective_global_batch"] = {
            "object": object_effective_batch,
            "flat": flat_effective_batch,
            "required": 256,
        }
    object_training = object_training_mismatches(object_checkpoint, required_step)
    flat_training = flat_training_mismatches(flat_checkpoint, required_step)
    if object_training:
        mismatches["object_training"] = object_training
    if flat_training:
        mismatches["flat_training"] = flat_training
    if mismatches:
        raise ValueError(
            "object-flat identity contract differs: "
            + json.dumps(mismatches, sort_keys=True)
        )

    initialization = flat_checkpoint.get("initialization", {})
    if (
        initialization.get("kind") != "object_dynamics_blocks_only"
        or initialization.get("source_checkpoint_sha256") != object_sha256
        or initialization.get("source_global_step") != required_step
        or initialization.get("missing")
        or initialization.get("shape_mismatch")
    ):
        raise ValueError("flat shared-Dynamics initialization provenance differs")
    config = AdaptiveGaussianWMConfig(**object_checkpoint["config"])
    if config.condition_dim != 0 or not config.rgb_supervision:
        raise ValueError("object-flat evaluation requires no language and RGB anchoring")
    if (
        not config.canonical_semantic_action
        or not config.rgb_semantic_action
        or not config.bounded_residual_action
        or config.action_residual_dim != 8
        or config.action_dim != 14
    ):
        raise ValueError("object-flat evaluation requires the object 6+8 RGB action")
    flat_rgb_expected = {
        "baseline_contract_version": 2,
        "modality_matching": "dino_rgb",
        "rgb_short_side": config.rgb_short_side,
        "rgb_pad_multiple": config.rgb_pad_multiple,
        "rgb_loss_weight": config.rgb_loss_weight,
        "rgb_ssim_weight": config.rgb_ssim_weight,
        "rgb_change_loss_weight": config.rgb_change_loss_weight,
        "rgb_change_threshold": config.rgb_change_threshold,
    }
    flat_rgb_mismatches = {
        name: {"flat": flat_args.get(name), "required": value}
        for name, value in flat_rgb_expected.items()
        if flat_args.get(name) != value
    }
    if flat_rgb_mismatches:
        raise ValueError(
            "flat RGB training contract differs: "
            + json.dumps(flat_rgb_mismatches, sort_keys=True)
        )
    architecture = flat_checkpoint.get("architecture", {})
    architecture_expected = {
        "feature_dim": dataset.feature_dim,
        "rgb_channels": 3,
        "model_dim": config.model_dim,
        "state_tokens": config.object_slots,
        "action_tokens": config.action_tokens,
        "action_dim": config.action_dim,
        "action_residual_dim": config.action_residual_dim,
        "dynamics_layers": config.dynamics_layers,
        "heads": config.heads,
    }
    if any(
        int(architecture.get(name, -1)) != value
        for name, value in architecture_expected.items()
    ):
        raise ValueError("flat architecture does not match object bottleneck or data")
    return config, architecture
