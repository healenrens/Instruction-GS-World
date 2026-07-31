"""Strict training and launch contracts for Object Memory JEPA v39."""

from __future__ import annotations

import json
import os
import subprocess

import torch

from .checkpointing import CHECKPOINT_VERSION
from .dense_readout_contracts import (
    file_sha256,
)
from .loss_weights import AdaptiveGaussianLossWeights
from .v39_stage_contracts import validate_v39_initialization


ARCHITECTURE = "object_memory_v1"
DEFAULT_TARGET_GLOBAL_BATCH = 256


def add_v28_arguments(parser) -> None:
    parser.add_argument(
        "--architecture",
        choices=("legacy", ARCHITECTURE),
        default="legacy",
    )
    parser.add_argument(
        "--training_stage",
        choices=("legacy", "representation", "readout", "posterior", "prior"),
        default="legacy",
    )
    parser.add_argument("--core_lr", type=float, default=2e-4)
    parser.add_argument("--action_lr", type=float, default=2e-4)
    parser.add_argument("--readout_lr", type=float, default=2e-4)
    parser.add_argument(
        "--readout_scope",
        choices=("off", "isolated", "joint"),
        default="off",
    )
    parser.add_argument("--current_readout_weight", type=float, default=0.0)
    parser.add_argument("--readout_regularization_weight", type=float, default=0.0)
    parser.add_argument("--carrier_support_weight", type=float, default=0.0)
    parser.add_argument("--carrier_compact_weight", type=float, default=0.0)
    parser.add_argument(
        "--gaussian_children", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument("--basis_gate_report", default="")
    parser.add_argument("--carrier_preflight_report", default="")
    parser.add_argument("--dense_preflight_report", default="")
    parser.add_argument("--readout_gate_report", default="")
    parser.add_argument("--dense_preflight_report_sha256", default="")
    parser.add_argument("--readout_gate_report_sha256", default="")
    parser.add_argument("--representation_gate_report", default="")
    parser.add_argument("--representation_gate_report_sha256", default="")
    parser.add_argument("--posterior_gate_report", default="")
    parser.add_argument("--posterior_gate_report_sha256", default="")
    parser.add_argument("--goal_rollout_weight", type=float, default=1.0)
    parser.add_argument("--path_consistency_weight", type=float, default=0.25)
    parser.add_argument("--gate_report_sha256", default="")
    parser.add_argument(
        "--target_global_batch",
        type=int,
        default=DEFAULT_TARGET_GLOBAL_BATCH,
    )
    parser.add_argument("--gate_report", default="")


def is_v28(args) -> bool:
    return args.architecture == ARCHITECTURE


def validate_v28_initialization(checkpoint: dict, args) -> None:
    if is_v28(args):
        validate_v39_initialization(checkpoint, args)


def resolve_v28_gradient_accumulation(args, world_size: int) -> None:
    if not is_v28(args):
        return
    if world_size <= 0 or args.target_global_batch <= 0:
        raise ValueError("v39 world size and target global batch must be positive")
    if args.grad_accum == 0:
        samples_per_micro_step = args.batch * world_size
        args.grad_accum = max(
            1,
            (args.target_global_batch + samples_per_micro_step // 2)
            // samples_per_micro_step,
        )
    args.effective_global_batch = args.batch * args.grad_accum * world_size


def validate_v28_arguments(args, world_size: int) -> None:
    if not is_v28(args):
        if args.training_stage != "legacy" or args.teacher_sidecar:
            raise ValueError("staged training and sidecars require object_memory_v1")
        return
    if args.training_stage not in ("representation", "posterior", "prior"):
        raise ValueError("object_memory_v1 requires an explicit training stage")
    if args.profile != "full" or args.data_format != "sequence":
        raise ValueError("object_memory_v1 requires full sequence training")
    if args.feature_source != "jit":
        raise ValueError("v39 object_memory_v1 requires per-rank JIT DINO")
    if args.jit_dino_batch < 1:
        raise ValueError("v39 JIT DINO frame batch must be positive")
    if args.temporal_contract != "dynamic_dual_horizon_v1":
        raise ValueError("v39 requires the dynamic dual-horizon temporal contract")
    if (args.history_frames_min, args.history_frames_max, args.future_frames) != (
        1,
        4,
        2,
    ):
        raise ValueError("v39 requires H=1..4 and exactly two future targets")
    if args.short_horizon_frames != 30:
        raise ValueError("v39 short target must be +30 native 30 Hz frames")
    if args.goal_query_seconds <= 1.0 or args.goal_stability_threshold <= 0.0:
        raise ValueError("v39 goal query and stability contracts are invalid")
    if args.language_condition != "off" or args.rgb_supervision != "off":
        raise ValueError("object_memory_v1 forces language and RGB supervision off")
    if args.condition_cache:
        raise ValueError("object_memory_v1 forbids condition caches")
    if args.action_anchor != "global" or args.semantic_action_basis != "fixed":
        raise ValueError("object_memory_v1 forbids canonical action overrides")
    if args.posterior_dynamics_gate or args.posterior_core_training:
        raise ValueError("object_memory_v1 uses training_stage, not legacy modes")
    if min(args.core_lr, args.action_lr, args.readout_lr) <= 0.0:
        raise ValueError("v39 learning rates must be positive")
    if args.goal_rollout_weight <= 0.0 or args.path_consistency_weight <= 0.0:
        raise ValueError("v39 requires positive rollout and path objectives")
    if (
        min(
            args.current_readout_weight,
            args.readout_regularization_weight,
            args.carrier_support_weight,
            args.carrier_compact_weight,
        )
        < 0.0
    ):
        raise ValueError("v39 readout weights must be non-negative")
    if args.gaussian_children != 1:
        raise ValueError("v39 change residual readout requires gaussian_children=1")
    if (
        args.basis_gate_report
        or args.carrier_preflight_report
        or args.carrier_support_weight != 0.0
        or args.carrier_compact_weight != 0.0
    ):
        raise ValueError("v39 forbids hierarchical Gaussian carrier inputs")
    if args.grad_accum <= 0:
        raise ValueError("v39 gradient accumulation did not resolve")
    if args.lr != args.core_lr or abs(args.lr_floor / args.core_lr - 0.1) > 1e-9:
        raise ValueError("v39 legacy LR fields must mirror core LR and its 0.1 floor")
    if args.warmup_steps != 0 or abs(args.warmup_fraction - 0.05) > 1e-9:
        raise ValueError("v39 warmup is fixed at five percent")
    if args.training_stage == "representation":
        if args.representation_steps <= 0 or args.joint_steps != 0:
            raise ValueError("representation stage must only set representation_steps")
    elif args.representation_steps != 0 or args.joint_steps <= 0:
        raise ValueError("posterior/prior stage must only set joint_steps")
    if (
        args.readout_scope != "off"
        or args.current_readout_weight != 0.0
        or args.readout_regularization_weight != 0.0
        or args.carrier_support_weight != 0.0
        or args.carrier_compact_weight != 0.0
        or args.basis_gate_report
        or args.carrier_preflight_report
        or args.dense_preflight_report
        or args.readout_gate_report
    ):
        raise ValueError("v39 has no separate readout stage or legacy readout gates")
    if args.training_stage in ("posterior", "prior") and not (
        args.init_from or args.resume
    ):
        raise ValueError(
            "posterior/prior stage requires a completed-stage init or strict resume"
        )
    if args.training_stage == "representation" and args.representation_gate_report:
        raise ValueError("representation training cannot consume its own held gate")
    if args.training_stage == "posterior" and not (
        args.representation_gate_report or args.resume
    ):
        raise ValueError("posterior training requires a passed representation gate")
    if args.training_stage == "prior" and not (
        args.posterior_gate_report or args.resume
    ):
        raise ValueError("prior training requires a passed posterior gate")
    if not args.validate_only:
        if args.batch not in (2, 4, 8):
            raise ValueError("v39 per-rank batch must be 2, 4, or 8")
        if not args.gate_report:
            raise ValueError("v39 training requires --gate_report")


def validate_v28_gate(args, dataset, project_root: str) -> dict:
    if not is_v28(args):
        return {}
    if not hasattr(dataset, "teacher_sidecar"):
        raise ValueError("object_memory_v1 requires the dense episode backend")
    if args.validate_only:
        return {}
    with open(args.gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        text=True,
    )
    if worktree.strip():
        raise ValueError("v39 training rejects tracked worktree changes")
    expected = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": "change_only_object_residual",
        "feature_contract": "jit_backbone_native_dinov2_l_1024",
        "jit_dino_model": "vit_large_patch14_dinov2.lvd142m",
        "jit_dino_image_size": 518,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "control_hz": float(dataset.control_hz),
        "temporal_contract": "dynamic_dual_horizon_v1",
        "history_lengths": [1, 2, 3, 4],
        "history_span_frames": list(dataset.history_span_frames),
        "short_horizon_frames": 30,
        "goal_query_seconds": args.goal_query_seconds,
        "goal_tail_guard_frames": args.goal_tail_guard_frames,
        "goal_probe_frames": args.goal_probe_frames,
        "goal_stability_threshold": args.goal_stability_threshold,
        "goal_rollout_weight": args.goal_rollout_weight,
        "path_consistency_weight": args.path_consistency_weight,
        "git_commit": current_commit,
        "data_manifest_sha256": dataset.data_sha256,
        "teacher_sidecar_sha256": getattr(dataset, "teacher_sidecar_sha256", ""),
    }
    mismatch = {
        name: {"gate": report.get(name), "current": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    if mismatch:
        raise ValueError(f"v39 verifier gate differs: {mismatch}")
    args.gate_report_sha256 = file_sha256(args.gate_report)
    args.dense_preflight_report_sha256 = (
        file_sha256(args.dense_preflight_report) if args.dense_preflight_report else ""
    )
    args.readout_gate_report_sha256 = (
        file_sha256(args.readout_gate_report) if args.readout_gate_report else ""
    )
    args.representation_gate_report_sha256 = (
        file_sha256(args.representation_gate_report)
        if args.representation_gate_report
        else ""
    )
    args.posterior_gate_report_sha256 = (
        file_sha256(args.posterior_gate_report) if args.posterior_gate_report else ""
    )
    return report


def _action_modules(model) -> tuple:
    dynamics = model.dynamics
    return (
        model.latent_actions.posterior,
        model.latent_actions.effect_head,
        dynamics.routing_query,
        dynamics.action_input,
        dynamics.action_slot_basis,
        dynamics.action_slot_gate,
        dynamics.action_geometry_basis,
        dynamics.action_geometry_gate,
        model.effect_composer,
    )


def configure_v28_stage(model, args) -> None:
    if not is_v28(args):
        return
    if args.training_stage == "prior":
        model.requires_grad_(False)
        model.latent_actions.prior.requires_grad_(True)
        for parameter in model.latent_actions.prior_condition_parameters():
            parameter.requires_grad_(True)
        return
    model.latent_actions.prior.requires_grad_(False)
    for parameter in model.latent_actions.prior_condition_parameters():
        parameter.requires_grad_(False)
    if args.training_stage == "representation":
        for module in _action_modules(model):
            module.requires_grad_(False)
        model.dynamics.factor_keys.requires_grad_(False)


def build_optimizer(model, args) -> torch.optim.AdamW:
    if not is_v28(args):
        parameters = [value for value in model.parameters() if value.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    action_prefixes = (
        "latent_actions.posterior.",
        "latent_actions.effect_head.",
        "latent_actions.prior.",
        "latent_actions.prior_",
        "dynamics.factor_keys",
        "dynamics.routing_query.",
        "dynamics.action_",
        "effect_composer.",
    )
    core = []
    action = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = action if name.startswith(action_prefixes) else core
        target.append(parameter)
    groups = []
    if core:
        groups.append({"params": core, "lr": args.core_lr, "group_name": "core"})
    if action:
        groups.append({"params": action, "lr": args.action_lr, "group_name": "action"})
    if not groups:
        raise ValueError("v39 optimizer has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def v28_loss_weights(args) -> AdaptiveGaussianLossWeights | None:
    if not is_v28(args):
        return None
    common = dict(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=1.0,
        allocator=0.2,
        slot=0.2,
        geometry=0.25,
        rgb=0.0,
        current_readout=0.0,
        readout_regularization=0.0,
        carrier_support=0.0,
        carrier_compact=0.0,
    )
    if args.training_stage == "representation":
        return AdaptiveGaussianLossWeights(
            action=0.0,
            action_specificity=0.0,
            **common,
        )
    if args.training_stage == "posterior":
        return AdaptiveGaussianLossWeights(
            action=0.0,
            action_specificity=1.0,
            **common,
        )
    return AdaptiveGaussianLossWeights(
        future=0.0,
        history=0.0,
        flow=1.0,
        feature=0.0,
        allocator=0.0,
        slot=0.0,
        action=0.0,
        action_specificity=0.0,
        geometry=0.0,
        rgb=0.0,
    )


def v28_runtime_metadata(args, dataset, gate: dict) -> dict:
    if not is_v28(args):
        return {}
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "architecture": args.architecture,
        "training_stage": args.training_stage,
        "diagnostics_contract": "dynamic_dual_horizon_training_v1",
        "language_condition": "off",
        "rgb_supervision": "off",
        "latent_action_shape": [4, 32],
        "temporal_contract": "dynamic_dual_horizon_v1",
        "history_lengths": [1, 2, 3, 4],
        "history_span_frames": list(dataset.history_span_frames),
        "short_horizon_frames": args.short_horizon_frames,
        "goal_query_seconds": args.goal_query_seconds,
        "goal_tail_guard_frames": args.goal_tail_guard_frames,
        "goal_probe_frames": args.goal_probe_frames,
        "goal_stability_threshold": args.goal_stability_threshold,
        "goal_rollout_weight": args.goal_rollout_weight,
        "path_consistency_weight": args.path_consistency_weight,
        "data_manifest_sha256": dataset.data_sha256,
        "feature_source": args.feature_source,
        "feature_contract": "jit_backbone_native_dinov2_l_1024",
        "jit_dino_model": "vit_large_patch14_dinov2.lvd142m",
        "jit_dino_image_size": 518,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "control_hz": float(dataset.control_hz),
        "core_lr": args.core_lr,
        "action_lr": args.action_lr,
        "readout_lr": args.readout_lr,
        "readout_scope": args.readout_scope,
        "current_readout_weight": args.current_readout_weight,
        "readout_regularization_weight": args.readout_regularization_weight,
        "carrier_support_weight": args.carrier_support_weight,
        "carrier_compact_weight": args.carrier_compact_weight,
        "gaussian_children": args.gaussian_children,
        "readout_backend": "change_only_object_residual",
        "basis_gate_report": (
            os.path.abspath(args.basis_gate_report) if args.basis_gate_report else ""
        ),
        "carrier_preflight_report": (
            os.path.abspath(args.carrier_preflight_report)
            if args.carrier_preflight_report
            else ""
        ),
        "dense_preflight_report": (
            os.path.abspath(args.dense_preflight_report)
            if args.dense_preflight_report
            else ""
        ),
        "dense_preflight_report_sha256": args.dense_preflight_report_sha256,
        "readout_gate_report": (
            os.path.abspath(args.readout_gate_report)
            if args.readout_gate_report
            else ""
        ),
        "readout_gate_report_sha256": args.readout_gate_report_sha256,
        "representation_gate_report": (
            os.path.abspath(args.representation_gate_report)
            if args.representation_gate_report
            else ""
        ),
        "representation_gate_report_sha256": (args.representation_gate_report_sha256),
        "posterior_gate_report": (
            os.path.abspath(args.posterior_gate_report)
            if args.posterior_gate_report
            else ""
        ),
        "posterior_gate_report_sha256": args.posterior_gate_report_sha256,
        "gpu_policy": "auto",
        "target_global_batch": args.target_global_batch,
        "effective_global_batch": args.effective_global_batch,
        "teacher_sidecar": "enabled" if args.teacher_sidecar else "disabled",
        "teacher_sidecar_sha256": getattr(dataset, "teacher_sidecar_sha256", ""),
        "disabled_teacher_losses": (
            [] if args.teacher_sidecar else ["relative_disparity", "visibility"]
        ),
        "gate_report": os.path.abspath(args.gate_report) if args.gate_report else "",
        "gate_report_sha256": args.gate_report_sha256,
        "gate_git_commit": gate.get("git_commit", ""),
    }
