"""Strict resume and explicit warm-start handling for adaptive WM checkpoints."""

from __future__ import annotations

import json
import os
import random

import torch
import torch.distributed as dist


CHECKPOINT_VERSION = 38


def collect_rng_states(context) -> list[dict]:
    device = torch.device(context.device)
    local_state = {
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state(device=device) if device.type == "cuda" else None
        ),
        "python": random.getstate(),
    }
    if not context.distributed:
        return [local_state]
    states: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("failed to gather RNG state from every rank")
    return [state for state in states if state is not None]


def restore_rng_state(checkpoint: dict, context) -> None:
    states = checkpoint["rng_states"]
    if len(states) != context.world_size:
        raise ValueError("resume RNG state count differs from world size")
    state = states[context.rank]
    torch.set_rng_state(state["torch"])
    device = torch.device(context.device)
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("resume checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"], device=device)
    random.setstate(state["python"])


def save_checkpoint(
    path: str,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args,
    phase: str,
    phase_step: int,
    global_step: int,
    rng_states: list[dict],
    checkpoint_kind: str,
) -> dict:
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "parallelism": "ddp_full_state_dict",
        "git_commit": getattr(args, "git_commit", ""),
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": model.config.to_dict(),
        "args": vars(args),
        "phase": phase,
        "phase_step": phase_step,
        "global_step": global_step,
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
    size_bytes = os.path.getsize(path)
    if size_bytes <= 0:
        raise RuntimeError(f"checkpoint is empty after save: {path}")
    latest = os.path.join(os.path.dirname(path), "latest.pt")
    temporary_link = f"{latest}.tmp.{os.getpid()}"
    if os.path.lexists(temporary_link):
        os.unlink(temporary_link)
    os.symlink(os.path.basename(path), temporary_link)
    os.replace(temporary_link, latest)
    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": os.path.abspath(path),
        "checkpoint_file": os.path.basename(path),
        "phase": phase,
        "phase_step": phase_step,
        "global_step": global_step,
        "world_size": len(rng_states),
        "git_commit": getattr(args, "git_commit", ""),
        "size_bytes": size_bytes,
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


def checkpoint_target(
    out: str,
    phase: str,
    phase_step: int,
    global_step: int,
    phase_steps: int,
    save_every: int,
    recovery_every: int,
) -> tuple[str, str] | None:
    phase_complete = phase_step == phase_steps
    milestone_due = save_every > 0 and global_step % save_every == 0
    recovery_due = recovery_every > 0 and (
        global_step == 1 or global_step % recovery_every == 0
    )
    if not (phase_complete or milestone_due or recovery_due):
        return None
    if phase_complete or milestone_due:
        return os.path.join(out, f"{phase}_{phase_step:07d}.pt"), "milestone"
    return os.path.join(out, f"{phase}_recovery.pt"), "recovery"


def validate_resume(checkpoint: dict, args, world_size: int) -> None:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"resume requires a version-{CHECKPOINT_VERSION} checkpoint; "
            "use --init_from for older checkpoints"
        )
    if checkpoint.get("parallelism") != "ddp_full_state_dict":
        raise ValueError("resume requires a DDP full-state-dict checkpoint")
    if checkpoint.get("git_commit") != getattr(args, "git_commit", ""):
        raise ValueError("resume checkpoint git commit differs")
    required_sections = (
        "model",
        "optimizer",
        "scheduler",
        "config",
        "args",
        "world_size",
        "rng_states",
    )
    missing_sections = [name for name in required_sections if name not in checkpoint]
    if missing_sections:
        raise ValueError(f"resume checkpoint is missing {missing_sections}")
    if (
        checkpoint["world_size"] != world_size
        or len(checkpoint["rng_states"]) != world_size
    ):
        raise ValueError("resume checkpoint world size differs")
    saved = checkpoint["args"]
    immutable = (
        "data",
        "dino",
        "data_format",
        "feature_source",
        "jit_dino_batch",
        "history_frames",
        "future_frames",
        "sequence_anchors",
        "sequence_data_sha256",
        "condition_cache",
        "condition_feature_sha256",
        "teacher_sidecar",
        "teacher_sidecar_sha256",
        "architecture",
        "training_stage",
        "profile",
        "representation_steps",
        "joint_steps",
        "batch",
        "grad_accum",
        "max_train_items",
        "lr",
        "lr_floor",
        "core_lr",
        "action_lr",
        "readout_lr",
        "readout_scope",
        "current_readout_weight",
        "readout_regularization_weight",
        "carrier_support_weight",
        "carrier_compact_weight",
        "gaussian_children",
        "basis_gate_report",
        "carrier_preflight_report",
        "dense_preflight_report",
        "dense_preflight_report_sha256",
        "readout_gate_report",
        "readout_gate_report_sha256",
        "representation_gate_report",
        "representation_gate_report_sha256",
        "gate_report",
        "gate_report_sha256",
        "target_global_batch",
        "warmup_steps",
        "warmup_fraction",
        "weight_decay",
        "seed",
        "amp",
        "language_condition",
        "rgb_supervision",
        "rgb_short_side",
        "rgb_pad_multiple",
        "rgb_render_chunk",
        "rgb_loss_weight",
        "rgb_ssim_weight",
        "rgb_change_loss_weight",
        "rgb_change_threshold",
        "language_effect_weight",
        "zero_action_margin_weight",
        "zero_action_relative_margin",
        "aggregation_mode",
        "density_mode",
        "joint_flow",
        "posterior_dynamics_gate",
        "posterior_core_training",
        "posterior_update_scope",
        "action_anchor",
        "canonical_center_gate",
        "canonical_activity_gate",
        "canonical_activity_power",
        "action_residual_dim",
        "action_residual_gate",
        "action_residual_dropout",
        "semantic_action_basis",
    )
    mismatches = {}
    for name in immutable:
        if name not in saved:
            if name == "posterior_dynamics_gate":
                previous = False
            else:
                mismatches[name] = {
                    "checkpoint": "<missing>",
                    "current": getattr(args, name),
                }
                continue
        else:
            previous = saved[name]
        current = getattr(args, name)
        if (
            name
            in (
                "data",
                "dino",
                "condition_cache",
                "teacher_sidecar",
            )
            and current
        ):
            current = os.path.abspath(current)
            previous = os.path.abspath(previous)
        if current != previous:
            mismatches[name] = {"checkpoint": previous, "current": current}
    if mismatches:
        raise ValueError(
            "resume-critical arguments differ: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _copy_axis_prefix(
    source: torch.Tensor,
    target: torch.Tensor,
    axis: int,
) -> torch.Tensor | None:
    if source.ndim != target.ndim:
        return None
    if any(
        source.shape[index] != target.shape[index]
        for index in range(source.ndim)
        if index != axis
    ):
        return None
    output = target.clone()
    count = min(source.shape[axis], target.shape[axis])
    source_slice = [slice(None)] * source.ndim
    target_slice = [slice(None)] * target.ndim
    source_slice[axis] = slice(0, count)
    target_slice[axis] = slice(0, count)
    output[tuple(target_slice)] = source[tuple(source_slice)]
    return output


def _resize_action_tensor(
    name: str,
    source: torch.Tensor,
    target: torch.Tensor,
    checkpoint: dict,
    target_action_dim: int,
) -> tuple[torch.Tensor, str] | None:
    source_config = checkpoint.get("config", {})
    source_action_dim = int(source_config.get("action_dim", 0))
    if source_action_dim <= 0:
        return None
    posterior_residual = {
        "latent_actions.posterior.output.3.weight",
        "latent_actions.posterior.output.3.bias",
        "latent_actions.posterior.output_norm.weight",
        "latent_actions.posterior.output_norm.bias",
    }
    if name in posterior_residual:
        source_canonical_dim = (
            6 if source_config.get("canonical_semantic_action", False) else 0
        )
        source_is_residual_head = (
            checkpoint.get("checkpoint_version", 1) >= 14 and source_canonical_dim > 0
        )
        offset = 0 if source_is_residual_head else source_canonical_dim
        resized = _copy_axis_prefix(source[offset:], target, 0)
        return (
            (resized, f"posterior_residual_slice_{offset}")
            if resized is not None
            else None
        )
    prefix_axis = {
        "latent_actions.effect_head.0.weight": 0,
        "latent_actions.effect_head.0.bias": 0,
        "latent_actions.effect_head.1.weight": 1,
        "latent_actions.center_effect_head.0.weight": 0,
        "latent_actions.center_effect_head.0.bias": 0,
        "latent_actions.center_effect_head.1.weight": 1,
        "latent_actions.prior.action_identity": 1,
        "latent_actions.prior.output_projection.weight": 0,
        "latent_actions.prior.output_projection.bias": 0,
        "dynamics.action_input.weight": 1,
        "dynamics.residual_action_input.weight": 1,
    }
    if name in prefix_axis:
        resized = _copy_axis_prefix(source, target, prefix_axis[name])
        return (resized, "action_prefix_copy") if resized is not None else None
    if name == "latent_actions.prior.input_projection.weight":
        source_tail = source.shape[1] - 2 * source_action_dim
        target_tail = target.shape[1] - 2 * target_action_dim
        if source.shape[0] != target.shape[0] or source_tail != target_tail:
            return None
        count = min(source_action_dim, target_action_dim)
        output = target.clone()
        output[:, :count] = source[:, :count]
        output[:, target_action_dim : target_action_dim + count] = source[
            :, source_action_dim : source_action_dim + count
        ]
        output[:, 2 * target_action_dim :] = source[:, 2 * source_action_dim :]
        return output, "flow_input_action_prefix_and_context_copy"
    return None


def warm_start_model(model, checkpoint: dict) -> dict:
    source = checkpoint["model"]
    source_config = checkpoint.get("config", {})
    target = model.state_dict()
    compatible = {}
    transformed = {}
    dropped = {}
    unexpected = []
    shape_mismatch = {}
    rgb_semantics_changed = model.config.rgb_semantic_action and not source_config.get(
        "rgb_semantic_action", False
    )
    semantic_input_weights = {
        "dynamics.action_input.weight",
        "latent_actions.effect_head.1.weight",
        "latent_actions.center_effect_head.1.weight",
    }
    for name, value in source.items():
        if name not in target:
            if rgb_semantics_changed and name == (
                "latent_actions.posterior.semantic_projection"
            ):
                dropped[name] = "rgb_semantic_action_removed_slot_basis"
            elif (
                model.config.canonical_semantic_action
                and model.config.action_residual_dim == 0
                and (
                    name.startswith("latent_actions.posterior.output.")
                    or name.startswith("latent_actions.posterior.output_norm.")
                )
            ):
                dropped[name] = "canonical_only_removed_residual_head"
            elif (
                model.config.action_residual_dim == 0
                and name == "dynamics.residual_action_input.weight"
            ):
                dropped[name] = "canonical_only_removed_residual_projection"
            elif (
                model.config.bounded_residual_action
                and name == "dynamics.action_input.bias"
            ):
                dropped[name] = "bounded_action_projection_removed_bias"
            else:
                unexpected.append(name)
        elif target[name].shape != value.shape:
            action_resize = _resize_action_tensor(
                name,
                value,
                target[name],
                checkpoint,
                model.config.action_dim,
            )
            if action_resize is not None:
                compatible[name], transform = action_resize
                transformed[name] = {
                    "source": list(value.shape),
                    "target": list(target[name].shape),
                    "transform": transform,
                }
                continue
            if (
                name
                in (
                    "latent_actions.posterior.queries",
                    "latent_actions.prior.action_identity",
                )
                and value.ndim == 2
                and target[name].shape[1:] == value.shape[1:]
                and target[name].shape[0] % value.shape[0] == 0
            ):
                repeats = target[name].shape[0] // value.shape[0]
                compatible[name] = value.repeat(repeats, 1)
                transformed[name] = {
                    "source": list(value.shape),
                    "target": list(target[name].shape),
                    "transform": "cyclic_repeat",
                }
                continue
            shape_mismatch[name] = {
                "source": list(value.shape),
                "target": list(target[name].shape),
            }
        elif rgb_semantics_changed and name in semantic_input_weights:
            output = value.clone()
            output[:, 3:6] = target[name][:, 3:6]
            compatible[name] = output
            transformed[name] = {
                "source": list(value.shape),
                "target": list(output.shape),
                "transform": "reinitialize_rgb_semantic_columns_3_6",
            }
        else:
            compatible[name] = value
    residual_name = "dynamics.residual_action_input.weight"
    if (
        model.config.bounded_residual_action
        and residual_name in target
        and residual_name not in compatible
    ):
        source_weight = source.get("dynamics.action_input.weight")
        if (
            source_weight is None
            or source_weight.shape[0] != target[residual_name].shape[0]
        ):
            raise ValueError("cannot warm-start bounded residual projection")
        output = target[residual_name].clone()
        count = min(
            model.config.action_residual_dim,
            source_weight.shape[1] - 6,
        )
        output[:, :count] = source_weight[:, 6 : 6 + count]
        compatible[residual_name] = output
        transformed[residual_name] = {
            "source": list(source_weight.shape),
            "target": list(output.shape),
            "transform": "residual_action_slice_6",
        }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "source_checkpoint_version": checkpoint.get("checkpoint_version", 1),
        "loaded": len(compatible),
        "transformed": transformed,
        "dropped": dropped,
        "missing": sorted(result.missing_keys),
        "unexpected": sorted(unexpected),
        "shape_mismatch": shape_mismatch,
    }
