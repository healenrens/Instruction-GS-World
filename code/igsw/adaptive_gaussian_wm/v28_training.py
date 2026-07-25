"""Strict training and launch contracts for Object Memory JEPA v28."""
from __future__ import annotations

import json
import os
import subprocess

import torch

from .checkpointing import CHECKPOINT_VERSION
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
        choices=("legacy", "representation", "posterior"),
        default="legacy",
    )
    parser.add_argument("--core_lr", type=float, default=2e-4)
    parser.add_argument("--action_lr", type=float, default=2e-4)
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
        raise ValueError("v28 world size and target global batch must be positive")
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
            raise ValueError("v28 stage and sidecar require object_memory_v1")
        return
    if args.training_stage not in ("representation", "posterior"):
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
    if args.core_lr <= 0.0 or args.action_lr <= 0.0:
        raise ValueError("v28 learning rates must be positive")
    if args.grad_accum <= 0:
        raise ValueError("v28 gradient accumulation did not resolve")
    if args.lr != args.core_lr or abs(args.lr_floor / args.core_lr - 0.1) > 1e-9:
        raise ValueError("v28 legacy LR fields must mirror core LR and its 0.1 floor")
    if args.warmup_steps != 0 or abs(args.warmup_fraction - 0.05) > 1e-9:
        raise ValueError("v28 warmup is fixed at five percent")
    if args.training_stage == "representation":
        if args.representation_steps <= 0 or args.joint_steps != 0:
            raise ValueError("representation stage must only set representation_steps")
    elif args.representation_steps != 0 or args.joint_steps <= 0:
        raise ValueError("posterior stage must only set joint_steps")
    if args.training_stage == "posterior" and not (args.init_from or args.resume):
        raise ValueError("posterior stage requires representation init or strict resume")
    if not args.validate_only:
        if args.batch not in (2, 4, 8):
            raise ValueError("v28 per-rank batch must be 2, 4, or 8")
        if not args.gate_report:
            raise ValueError("v28 training requires --gate_report")


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
        raise ValueError("v28 training requires a clean worktree")
    expected = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
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
        raise ValueError(f"v28 verifier gate differs: {mismatch}")
    return report


def validate_v28_initialization(checkpoint: dict, args) -> None:
    if not is_v28(args):
        return
    version = int(checkpoint.get("checkpoint_version", 0))
    if version > CHECKPOINT_VERSION:
        raise ValueError("cannot initialize v28 from a newer checkpoint")
    if args.training_stage == "representation":
        if version != 27:
            raise ValueError(
                "representation init_from accepts v27 only; resume v28 strictly"
            )
        return
    if version != CHECKPOINT_VERSION:
        raise ValueError("posterior stage requires a v28 representation checkpoint")
    if checkpoint.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError("posterior initialization architecture differs")
    if checkpoint.get("phase") != "representation":
        raise ValueError("posterior initialization must come from representation")
    saved = checkpoint.get("args", {})
    if saved.get("training_stage") != "representation":
        raise ValueError("posterior initialization has no representation contract")
    if int(checkpoint.get("phase_step", -1)) != int(
        saved.get("representation_steps", -2)
    ):
        raise ValueError("representation checkpoint is not stage-complete")


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
        "dynamics.factor_keys",
        "dynamics.routing_query.",
        "dynamics.action_",
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
        groups.append(
            {"params": action, "lr": args.action_lr, "group_name": "action"}
        )
    if not groups:
        raise ValueError("v28 optimizer has no trainable parameters")
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
    )
    if args.training_stage == "representation":
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
        "architecture": args.architecture,
        "training_stage": args.training_stage,
        "language_condition": "off",
        "rgb_supervision": "off",
        "latent_action_shape": [4, 32],
        "data_manifest_sha256": dataset.data_sha256,
        "core_lr": args.core_lr,
        "action_lr": args.action_lr,
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
