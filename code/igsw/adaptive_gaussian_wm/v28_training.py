"""Strict training and launch contracts for Object Memory JEPA v29."""
from __future__ import annotations

import json
import os
import subprocess

import torch

from .checkpointing import CHECKPOINT_VERSION
from .carrier_contracts import (
    validate_basis_gate,
    validate_carrier_preflight,
    validate_readout_gate,
)
from .loss_weights import AdaptiveGaussianLossWeights


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
        choices=("legacy", "representation", "readout", "posterior"),
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
    parser.add_argument(
        "--readout_regularization_weight", type=float, default=0.0
    )
    parser.add_argument("--carrier_support_weight", type=float, default=0.0)
    parser.add_argument("--carrier_compact_weight", type=float, default=0.0)
    parser.add_argument(
        "--gaussian_children", type=int, choices=(1, 2, 4, 8), default=4
    )
    parser.add_argument("--basis_gate_report", default="")
    parser.add_argument("--carrier_preflight_report", default="")
    parser.add_argument("--readout_gate_report", default="")
    parser.add_argument(
        "--target_global_batch",
        type=int,
        default=DEFAULT_TARGET_GLOBAL_BATCH,
    )
    parser.add_argument("--gate_report", default="")


def is_v28(args) -> bool:
    return args.architecture == ARCHITECTURE


def resolve_v28_gradient_accumulation(args, world_size: int) -> None:
    if not is_v28(args):
        return
    if world_size <= 0 or args.target_global_batch <= 0:
        raise ValueError("v29 world size and target global batch must be positive")
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
    if args.training_stage not in ("representation", "readout", "posterior"):
        raise ValueError("object_memory_v1 requires an explicit training stage")
    if args.profile != "full" or args.data_format != "sequence":
        raise ValueError("object_memory_v1 requires full sequence training")
    if args.language_condition != "off" or args.rgb_supervision != "off":
        raise ValueError("object_memory_v1 forces language and RGB supervision off")
    if args.condition_cache:
        raise ValueError("object_memory_v1 forbids condition caches")
    if args.action_anchor != "global" or args.semantic_action_basis != "fixed":
        raise ValueError("object_memory_v1 forbids canonical action overrides")
    if args.posterior_dynamics_gate or args.posterior_core_training:
        raise ValueError("object_memory_v1 uses training_stage, not legacy modes")
    if min(args.core_lr, args.action_lr, args.readout_lr) <= 0.0:
        raise ValueError("v29 learning rates must be positive")
    if min(
        args.current_readout_weight,
        args.readout_regularization_weight,
        args.carrier_support_weight,
        args.carrier_compact_weight,
    ) < 0.0:
        raise ValueError("v29 carrier weights must be non-negative")
    if args.gaussian_children <= 1:
        raise ValueError("v29 requires a hierarchical Gaussian carrier")
    if args.grad_accum <= 0:
        raise ValueError("v29 gradient accumulation did not resolve")
    if args.lr != args.core_lr or abs(args.lr_floor / args.core_lr - 0.1) > 1e-9:
        raise ValueError("v29 legacy LR fields must mirror core LR and its 0.1 floor")
    if args.warmup_steps != 0 or abs(args.warmup_fraction - 0.05) > 1e-9:
        raise ValueError("v29 warmup is fixed at five percent")
    if args.training_stage in ("representation", "readout"):
        if args.representation_steps <= 0 or args.joint_steps != 0:
            raise ValueError(
                "representation/readout stage must only set representation_steps"
            )
    elif args.representation_steps != 0 or args.joint_steps <= 0:
        raise ValueError("posterior stage must only set joint_steps")
    if args.training_stage == "readout":
        if args.readout_scope == "off" or args.current_readout_weight <= 0.0:
            raise ValueError("readout stage requires a scope and current anchor")
        if not (args.init_from or args.resume):
            raise ValueError("readout stage requires checkpoint initialization")
    elif (
        args.readout_scope != "off"
        or args.current_readout_weight != 0.0
        or args.readout_regularization_weight != 0.0
        or args.carrier_support_weight != 0.0
        or args.carrier_compact_weight != 0.0
        or args.basis_gate_report
        or args.carrier_preflight_report
    ):
        raise ValueError("readout scope and anchor belong to readout stage")
    if args.training_stage == "posterior" and not (args.init_from or args.resume):
        raise ValueError("posterior stage requires readout init or strict resume")
    needs_readout_gate = (
        args.training_stage == "posterior"
        or (
            args.training_stage == "readout"
            and args.readout_scope == "joint"
        )
    )
    if needs_readout_gate and not (args.readout_gate_report or args.resume):
        raise ValueError("this stage requires a passed held readout gate")
    if args.readout_gate_report and not needs_readout_gate:
        raise ValueError("readout gate is only valid for joint readout or posterior")
    if not args.validate_only:
        if args.batch not in (2, 4, 8):
            raise ValueError("v29 per-rank batch must be 2, 4, or 8")
        if not args.gate_report:
            raise ValueError("v29 training requires --gate_report")
        if (
            args.training_stage == "readout"
            and args.readout_scope == "isolated"
            and not args.resume
            and not args.basis_gate_report
        ):
            raise ValueError("isolated carrier training requires --basis_gate_report")
        if (
            args.training_stage == "readout"
            and args.readout_scope == "isolated"
            and not args.resume
            and not args.carrier_preflight_report
        ):
            raise ValueError(
                "isolated carrier training requires --carrier_preflight_report"
            )


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
        ["git", "status", "--porcelain"],
        cwd=project_root,
        text=True,
    )
    if worktree.strip():
        raise ValueError("v29 training requires a clean worktree")
    expected = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "git_commit": current_commit,
        "data_manifest_sha256": dataset.data_sha256,
        "teacher_sidecar_sha256": getattr(
            dataset, "teacher_sidecar_sha256", ""
        ),
    }
    mismatch = {
        name: {"gate": report.get(name), "current": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    if mismatch:
        raise ValueError(f"v29 verifier gate differs: {mismatch}")
    return report


def _stage_complete(checkpoint: dict) -> bool:
    saved = checkpoint.get("args", {})
    expected = (
        saved.get("representation_steps")
        if checkpoint.get("phase") in ("representation", "readout")
        else saved.get("joint_steps")
    )
    return expected is not None and int(checkpoint.get("phase_step", -1)) == int(
        expected
    )


def validate_v28_initialization(checkpoint: dict, args) -> None:
    if not is_v28(args):
        return
    version = int(checkpoint.get("checkpoint_version", 0))
    if version > CHECKPOINT_VERSION:
        raise ValueError("cannot initialize v29 from a newer checkpoint")
    if args.training_stage == "representation":
        if version != 27:
            raise ValueError(
                "representation init_from accepts v27 only; resume v29 strictly"
            )
        return
    if checkpoint.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint initialization architecture differs")
    saved = checkpoint.get("args", {})
    if args.training_stage == "readout":
        if args.readout_scope == "isolated":
            if (
                version not in (28, CHECKPOINT_VERSION)
                or
                checkpoint.get("phase") != "representation"
                or saved.get("training_stage") != "representation"
            ):
                raise ValueError(
                    "isolated carrier must start from v28/v29 representation"
                )
            validate_basis_gate(args, args.init_from)
            validate_carrier_preflight(args, args.init_from)
        else:
            if (
                version != CHECKPOINT_VERSION
                or
                checkpoint.get("phase") != "readout"
                or saved.get("training_stage") != "readout"
                or saved.get("readout_scope") != "isolated"
                or not _stage_complete(checkpoint)
            ):
                raise ValueError("joint readout must start from completed isolation")
            validate_readout_gate(args, args.init_from, "isolated")
        return
    if (
        version != CHECKPOINT_VERSION
        or checkpoint.get("phase") != "readout"
        or saved.get("training_stage") != "readout"
        or saved.get("readout_scope") != "joint"
        or not _stage_complete(checkpoint)
    ):
        raise ValueError("posterior must start from completed joint readout repair")
    validate_readout_gate(args, args.init_from, "joint")


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
    )


def configure_v28_stage(model, args) -> None:
    if not is_v28(args):
        return
    model.latent_actions.prior.requires_grad_(False)
    for parameter in model.latent_actions.prior_condition_parameters():
        parameter.requires_grad_(False)
    if args.training_stage in ("representation", "readout"):
        for module in _action_modules(model):
            module.requires_grad_(False)
        model.dynamics.factor_keys.requires_grad_(False)
    if args.training_stage == "readout" and args.readout_scope == "isolated":
        model.requires_grad_(False)
    if args.training_stage == "readout":
        if model.gaussian_readout.hierarchical is None:
            model.gaussian_readout.requires_grad_(True)
        else:
            model.gaussian_readout.requires_grad_(False)
            model.gaussian_readout.hierarchical.requires_grad_(True)


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
        "dynamics.factor_keys",
        "dynamics.routing_query.",
        "dynamics.action_",
    )
    readout_prefixes = ("gaussian_readout.",)
    core = []
    action = []
    readout = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if args.training_stage == "readout" and name.startswith(readout_prefixes):
            target = readout
        else:
            target = action if name.startswith(action_prefixes) else core
        target.append(parameter)
    groups = []
    if core:
        groups.append({"params": core, "lr": args.core_lr, "group_name": "core"})
    if action:
        groups.append(
            {"params": action, "lr": args.action_lr, "group_name": "action"}
        )
    if readout:
        groups.append(
            {
                "params": readout,
                "lr": args.readout_lr,
                "group_name": "readout",
            }
        )
    if not groups:
        raise ValueError("v29 optimizer has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def v28_loss_weights(args) -> AdaptiveGaussianLossWeights | None:
    if not is_v28(args):
        return None
    common = dict(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=0.5,
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
    if args.training_stage == "readout":
        if args.readout_scope == "isolated":
            common.update(
                future=0.0,
                history=0.0,
                feature=0.0,
                allocator=0.0,
                slot=0.0,
                geometry=0.0,
            )
        common.update(
            current_readout=args.current_readout_weight,
            readout_regularization=args.readout_regularization_weight,
            carrier_support=args.carrier_support_weight,
            carrier_compact=args.carrier_compact_weight,
        )
        return AdaptiveGaussianLossWeights(
            action=0.0,
            action_specificity=0.0,
            **common,
        )
    return AdaptiveGaussianLossWeights(
        action=0.5,
        action_specificity=1.0,
        **common,
    )


def v28_runtime_metadata(args, dataset, gate: dict) -> dict:
    if not is_v28(args):
        return {}
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "architecture": args.architecture,
        "training_stage": args.training_stage,
        "diagnostics_contract": "object_memory_training_v5",
        "language_condition": "off",
        "rgb_supervision": "off",
        "latent_action_shape": [4, 32],
        "data_manifest_sha256": dataset.data_sha256,
        "core_lr": args.core_lr,
        "action_lr": args.action_lr,
        "readout_lr": args.readout_lr,
        "readout_scope": args.readout_scope,
        "current_readout_weight": args.current_readout_weight,
        "readout_regularization_weight": args.readout_regularization_weight,
        "carrier_support_weight": args.carrier_support_weight,
        "carrier_compact_weight": args.carrier_compact_weight,
        "gaussian_children": args.gaussian_children,
        "basis_gate_report": (
            os.path.abspath(args.basis_gate_report)
            if args.basis_gate_report
            else ""
        ),
        "carrier_preflight_report": (
            os.path.abspath(args.carrier_preflight_report)
            if args.carrier_preflight_report
            else ""
        ),
        "readout_gate_report": (
            os.path.abspath(args.readout_gate_report)
            if args.readout_gate_report
            else ""
        ),
        "gpu_policy": "auto",
        "target_global_batch": args.target_global_batch,
        "effective_global_batch": args.effective_global_batch,
        "teacher_sidecar": "enabled" if args.teacher_sidecar else "disabled",
        "teacher_sidecar_sha256": getattr(dataset, "teacher_sidecar_sha256", ""),
        "disabled_teacher_losses": (
            [] if args.teacher_sidecar else ["relative_disparity", "visibility"]
        ),
        "gate_report": os.path.abspath(args.gate_report) if args.gate_report else "",
        "gate_git_commit": gate.get("git_commit", ""),
    }
