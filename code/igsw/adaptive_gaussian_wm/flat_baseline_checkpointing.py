"""Identity and checkpoint contracts for the matched unstructured baseline."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from .checkpointing import CHECKPOINT_VERSION as OBJECT_CHECKPOINT_VERSION
from .goal_prior_checkpointing import file_sha256
from .matched_flat_training_contract import object_training_mismatches
from .matched_flat_world_model import MatchedFlatLatentWorldModel


FLAT_CHECKPOINT_VERSION = 4
FLAT_CHECKPOINT_KIND = "visual_sequence_matched_flat_dino_rgb_baseline"


def source_digest(path: str, context) -> str:
    digest = file_sha256(path) if context.is_main else None
    if context.distributed:
        values = [digest]
        dist.broadcast_object_list(values, src=0)
        digest = values[0]
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("failed to establish reference checkpoint SHA256")
    return digest


def reference_contract(
    checkpoint: dict,
    args,
    dataset,
) -> dict:
    if checkpoint.get("checkpoint_version") != OBJECT_CHECKPOINT_VERSION:
        raise ValueError("flat baseline requires a current object checkpoint")
    if checkpoint.get("phase") != "joint":
        raise ValueError("flat baseline reference must be a joint checkpoint")
    if int(checkpoint.get("global_step", -1)) != args.required_reference_step:
        raise ValueError("flat baseline reference step differs")
    saved = checkpoint.get("args", {})
    expected = {
        "data_format": "sequence",
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "sequence_anchors": args.sequence_anchors,
        "sequence_data_sha256": dataset.data_sha256,
    }
    mismatches = {
        name: {"checkpoint": saved.get(name), "baseline": value}
        for name, value in expected.items()
        if saved.get(name) != value
    }
    config = checkpoint.get("config", {})
    if int(config.get("condition_dim", -1)) != 0:
        mismatches["condition_dim"] = {
            "checkpoint": config.get("condition_dim"),
            "baseline": 0,
        }
    if float(config.get("gap_reference", -1.0)) != args.gap_reference:
        mismatches["gap_reference"] = {
            "checkpoint": config.get("gap_reference"),
            "baseline": args.gap_reference,
        }
    rgb_expected = {
        "rgb_supervision": True,
        "rgb_short_side": args.rgb_short_side,
        "rgb_pad_multiple": args.rgb_pad_multiple,
        "rgb_loss_weight": args.rgb_loss_weight,
        "rgb_ssim_weight": args.rgb_ssim_weight,
        "rgb_change_loss_weight": args.rgb_change_loss_weight,
        "rgb_change_threshold": args.rgb_change_threshold,
        "canonical_semantic_action": True,
        "rgb_semantic_action": True,
        "bounded_residual_action": True,
    }
    for name, value in rgb_expected.items():
        if config.get(name) != value:
            mismatches[name] = {
                "checkpoint": config.get(name),
                "baseline": value,
            }
    action_dim = int(config.get("action_dim", 0))
    canonical_dim = 6 if config.get("canonical_semantic_action") else 0
    residual_dim = action_dim - canonical_dim
    if residual_dim != 8 or action_dim != 6 + residual_dim:
        mismatches["action_layout"] = {
            "checkpoint": {
                "action_dim": action_dim,
                "action_residual_dim": residual_dim,
            },
            "baseline": {
                "action_dim": 14,
                "action_residual_dim": 8,
            },
        }
    training_mismatches = object_training_mismatches(
        checkpoint,
        args.required_reference_step,
    )
    if training_mismatches:
        mismatches["object_training"] = training_mismatches
    if mismatches:
        raise ValueError(
            "flat baseline reference contract differs: "
            + json.dumps(mismatches, sort_keys=True)
        )
    architecture = {
        "rgb_channels": 3,
        "model_dim": int(config.get("model_dim", 0)),
        "state_tokens": int(config.get("object_slots", 0)),
        "action_tokens": int(config.get("action_tokens", 0)),
        "action_dim": action_dim,
        "action_residual_dim": residual_dim,
        "dynamics_layers": int(config.get("dynamics_layers", 0)),
        "heads": int(config.get("heads", 0)),
        "dropout": float(config.get("dropout", -1.0)),
        "checkpoint_blocks": args.gradient_checkpointing == "on",
    }
    if min(
        architecture[name]
        for name in (
            "model_dim",
            "state_tokens",
            "action_tokens",
            "action_dim",
            "action_residual_dim",
            "dynamics_layers",
            "heads",
        )
    ) <= 0:
        raise ValueError("reference unstructured baseline architecture is invalid")
    return architecture


def save_flat_checkpoint(
    path: str,
    model: MatchedFlatLatentWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args,
    step: int,
    rng_states: list[dict],
    architecture: dict,
    initialization: dict,
) -> None:
    state = {
        "checkpoint_version": FLAT_CHECKPOINT_VERSION,
        "checkpoint_kind": FLAT_CHECKPOINT_KIND,
        "parallelism": "ddp_full_state_dict",
        "model": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "architecture": architecture,
        "initialization": initialization,
        "global_step": step,
        "world_size": len(rng_states),
        "rng_states": rng_states,
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


def initialize_flat_dynamics(
    model: MatchedFlatLatentWorldModel,
    object_checkpoint: dict,
    source_path: str,
    source_sha256: str,
) -> dict:
    target_state = model.state_dict()
    source_state = object_checkpoint.get("model", {})
    target_keys = sorted(
        name
        for name in target_state
        if name.startswith("dynamics.") and name.split(".")[1].isdigit()
    )
    loaded = []
    missing = []
    shape_mismatch = {}
    shared_parameters = 0
    for target_name in target_keys:
        suffix = target_name.removeprefix("dynamics.")
        source_name = f"dynamics.blocks.{suffix}"
        source_value = source_state.get(source_name)
        if source_value is None:
            missing.append({"target": target_name, "source": source_name})
            continue
        if source_value.shape != target_state[target_name].shape:
            shape_mismatch[target_name] = {
                "source": list(source_value.shape),
                "target": list(target_state[target_name].shape),
            }
            continue
        target_state[target_name] = source_value.to(
            dtype=target_state[target_name].dtype
        )
        loaded.append({"target": target_name, "source": source_name})
        shared_parameters += target_state[target_name].numel()
    if not target_keys or missing or shape_mismatch or len(loaded) != len(target_keys):
        raise ValueError(
            "matched flat Dynamics initialization is incomplete: "
            + json.dumps(
                {
                    "target_keys": len(target_keys),
                    "loaded": len(loaded),
                    "missing": missing,
                    "shape_mismatch": shape_mismatch,
                },
                sort_keys=True,
            )
        )
    model.load_state_dict(target_state, strict=True)
    return {
        "kind": "object_dynamics_blocks_only",
        "source_checkpoint": os.path.abspath(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_version": object_checkpoint.get("checkpoint_version"),
        "source_global_step": object_checkpoint.get("global_step"),
        "loaded_tensors": len(loaded),
        "shared_parameters": shared_parameters,
        "flat_parameter_fraction": shared_parameters
        / sum(parameter.numel() for parameter in model.parameters()),
        "loaded": loaded,
        "missing": [],
        "shape_mismatch": {},
        "excluded": [
            "object_allocator",
            "object_aggregator",
            "object_aligned_posterior",
            "gaussian_readout",
            "object_rgb_semantic_action_anchor",
            "object_dynamics_input_and_output_adapters",
        ],
    }


def validate_flat_resume(checkpoint: dict, args, world_size: int) -> None:
    if checkpoint.get("checkpoint_version") != FLAT_CHECKPOINT_VERSION:
        raise ValueError("flat baseline checkpoint version differs")
    if checkpoint.get("checkpoint_kind") != FLAT_CHECKPOINT_KIND:
        raise ValueError("flat baseline checkpoint kind differs")
    if checkpoint.get("parallelism") != "ddp_full_state_dict":
        raise ValueError("flat baseline resume requires a DDP full state dict")
    if checkpoint.get("world_size") != world_size:
        raise ValueError("flat baseline resume world size differs")
    if len(checkpoint.get("rng_states", [])) != world_size:
        raise ValueError("flat baseline resume RNG state count differs")
    immutable = (
        "reference_checkpoint",
        "reference_checkpoint_sha256",
        "required_reference_step",
        "baseline_contract_version",
        "modality_matching",
        "data",
        "data_sha256",
        "history_frames",
        "future_frames",
        "sequence_anchors",
        "steps",
        "batch",
        "grad_accum",
        "max_train_items",
        "gradient_checkpointing",
        "lr",
        "lr_floor",
        "warmup_steps",
        "weight_decay",
        "change_loss_weight",
        "history_loss_weight",
        "rgb_short_side",
        "rgb_pad_multiple",
        "rgb_loss_weight",
        "rgb_ssim_weight",
        "rgb_change_loss_weight",
        "rgb_change_threshold",
        "gap_reference",
        "seed",
        "amp",
    )
    saved = checkpoint.get("args", {})
    mismatches = {}
    for name in immutable:
        previous = saved.get(name, "<missing>")
        current = getattr(args, name)
        if name in ("data", "reference_checkpoint") and previous != "<missing>":
            previous = os.path.abspath(previous)
            current = os.path.abspath(current)
        if previous != current:
            mismatches[name] = {"checkpoint": previous, "current": current}
    if mismatches:
        raise ValueError(
            "flat baseline resume arguments differ: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _parameters(*modules) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def flat_parameter_metrics(model: MatchedFlatLatentWorldModel) -> dict[str, int]:
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "history_parameters": model.history_queries.numel() + _parameters(
            model.history_input,
            model.history_attention,
            model.history_output,
        ),
        "posterior_parameters": model.posterior_queries.numel() + _parameters(
            model.posterior_input,
            model.posterior_gap,
            model.posterior_attention,
            model.posterior_output,
        ),
        "dynamics_parameters": _parameters(
            model.action_input,
            model.residual_action_input,
            model.action_attention,
            model.gap_input,
            model.dynamics,
        ),
        "readout_parameters": _parameters(
            model.readout_input,
            model.readout_attention,
            model.feature_readout_output,
            model.rgb_readout_output,
        ),
    }
