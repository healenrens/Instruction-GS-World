"""DDP trainer for strict-causal DINO pairs and adaptive GPSToken Object-JEPA."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
    restore_rng_state,
    validate_resume,
    warm_start_model,
)
from igsw.adaptive_gaussian_wm.architecture_args import (  # noqa: E402
    apply_architecture_args,
    build_config,
)
from igsw.adaptive_gaussian_wm.dataset_factory import add_dataset_arguments, build_training_dataset  # noqa: E402
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.experiment_tracking import (  # noqa: E402
    add_wandb_arguments,
    init_wandb_tracker,
    validate_wandb_arguments,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    cosine_schedule,
    validate_data_model_contract,
)
from igsw.adaptive_gaussian_wm.training_loop import train_phase  # noqa: E402
from igsw.adaptive_gaussian_wm.training_modes import (  # noqa: E402
    configure_posterior_core_training,
    configure_posterior_dynamics_gate,
    staged_loss_weights,
)
from igsw.adaptive_gaussian_wm.v28_training import (  # noqa: E402
    add_v28_arguments,
    build_optimizer,
    configure_v28_stage,
    is_v28,
    resolve_v28_gradient_accumulation,
    v28_loss_weights,
    v28_runtime_metadata,
    validate_v28_arguments,
    validate_v28_gate,
    validate_v28_initialization,
)
from igsw.distributed import assert_same_paths, init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", default="")
    add_dataset_arguments(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--init_from", default="")
    parser.add_argument("--condition_cache", default="")
    parser.add_argument("--profile", choices=("tiny", "probe", "full"), default="probe")
    parser.add_argument("--representation_steps", type=int, default=1000)
    parser.add_argument("--joint_steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_floor", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--recovery_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--language_condition", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--rgb_supervision", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--rgb_short_side", type=int, default=256)
    parser.add_argument("--rgb_pad_multiple", type=int, default=16)
    parser.add_argument("--rgb_render_chunk", type=int, default=8192)
    parser.add_argument("--rgb_loss_weight", type=float, default=0.5)
    parser.add_argument("--rgb_ssim_weight", type=float, default=0.2)
    parser.add_argument("--rgb_change_loss_weight", type=float, default=0.0)
    parser.add_argument("--rgb_change_threshold", type=float, default=0.04)
    parser.add_argument("--language_effect_weight", type=float, default=0.0)
    parser.add_argument("--zero_action_margin_weight", type=float, default=0.0)
    parser.add_argument("--zero_action_relative_margin", type=float, default=0.01)
    parser.add_argument("--posterior_dynamics_gate", action="store_true")
    parser.add_argument("--posterior_core_training", action="store_true")
    parser.add_argument("--posterior_update_scope", choices=("full", "action_projection"), default="full")
    parser.add_argument("--canonical_activity_gate", action="store_true")
    parser.add_argument("--canonical_activity_power", type=float, default=0.5)
    parser.add_argument(
        "--aggregation_mode",
        choices=("auto", "competitive", "independent", "global"),
        default="auto",
    )
    parser.add_argument(
        "--density_mode",
        choices=("auto", "legacy", "adaptive", "fixed"),
        default="auto",
    )
    parser.add_argument(
        "--joint_flow",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--action_anchor",
        choices=("global", "object_slot"),
        default="global",
    )
    parser.add_argument("--canonical_center_gate", type=float,
                        choices=(1.0, 0.5, 0.25, 0.1), default=1.0)
    parser.add_argument(
        "--action_residual_dim",
        type=int,
        choices=(-1, 0, 8, 16, 58),
        default=-1,
        help="-1 preserves the profile action width; otherwise use 6+R dimensions",
    )
    parser.add_argument(
        "--action_residual_gate",
        type=float,
        choices=(1.0, 0.25, 0.1),
        default=1.0,
    )
    parser.add_argument(
        "--action_residual_dropout",
        type=float,
        choices=(0.0, 0.5, 0.75),
        default=0.0,
    )
    parser.add_argument(
        "--semantic_action_basis",
        choices=("fixed", "learned", "rgb"),
        default="fixed",
    )
    add_v28_arguments(parser)
    add_wandb_arguments(parser)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if min(args.representation_steps, args.joint_steps) < 0:
        raise ValueError("training steps cannot be negative")
    if not args.validate_only and args.representation_steps + args.joint_steps == 0:
        raise ValueError("training requires at least one optimizer step")
    if args.batch <= 0 or args.grad_accum < 0:
        raise ValueError("batch must be positive and grad_accum non-negative")
    if args.grad_accum == 0 and not is_v28(args):
        raise ValueError("automatic grad_accum is only available for v28")
    if args.save_every < 0 or args.recovery_every < 0:
        raise ValueError("checkpoint intervals must be non-negative")
    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive")
    if not 0.0 < args.lr_floor <= args.lr:
        raise ValueError("lr_floor must be in (0, lr]")
    if args.warmup_steps < 0 or not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("invalid warmup configuration")
    if args.language_condition == "on" and not args.condition_cache:
        raise ValueError("--language_condition on requires --condition_cache")
    if args.posterior_dynamics_gate and args.representation_steps != 0:
        raise ValueError("posterior Dynamics gate requires representation_steps=0")
    if args.posterior_core_training and args.representation_steps != 0:
        raise ValueError("posterior core training requires representation_steps=0")
    if args.posterior_dynamics_gate and args.posterior_core_training:
        raise ValueError("posterior training modes are mutually exclusive")
    if args.posterior_update_scope != "full" and not args.posterior_dynamics_gate:
        raise ValueError(
            "restricted posterior update scope requires --posterior_dynamics_gate"
        )
    validate_wandb_arguments(args)
    language_enabled = (
        bool(args.condition_cache)
        if args.language_condition == "auto"
        else args.language_condition == "on"
    )
    rgb_enabled = args.rgb_supervision != "off"
    context = init_torchrun()
    resolve_v28_gradient_accumulation(args, context.world_size)
    validate_v28_arguments(args, context.world_size)
    device = torch.device(context.device)
    seed = args.seed + context.rank
    random.seed(seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = build_training_dataset(
        args,
        "train",
        language_enabled,
        rgb_enabled,
    )
    args.condition_feature_sha256 = (
        dataset.condition_store.feature_sha256
        if dataset.condition_store is not None
        else ""
    )
    args.sequence_data_sha256 = getattr(dataset, "data_sha256", "")
    args.teacher_sidecar_sha256 = getattr(dataset, "teacher_sidecar_sha256", "")
    assert_same_paths(dataset.paths, context, dataset.contract_label)
    gate_report = validate_v28_gate(args, dataset, PROJECT_ROOT)
    if args.validate_only:
        if context.is_main:
            sample = dataset[0]
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "examples": len(dataset),
                        "feature_dim": dataset.feature_dim,
                        "history_shape": list(sample["history_features"].shape),
                        "future_shape": list(sample["future_features"].shape),
                        "condition_dim": dataset.condition_dim,
                        "teacher_sidecar": bool(args.teacher_sidecar),
                        "rgb_shape": (
                            list(sample["history_rgb"].shape)
                            if rgb_enabled
                            else None
                        ),
                    },
                    sort_keys=True,
                )
            )
        if context.distributed:
            dist.destroy_process_group()
        return

    if context.is_main:
        os.makedirs(args.out, exist_ok=True)
        if not args.resume and os.path.lexists(os.path.join(args.out, "latest.pt")):
            raise ValueError(f"output already contains a run: {args.out}")
    if context.distributed:
        dist.barrier()

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        validate_resume(checkpoint, args, context.world_size)
        config = AdaptiveGaussianWMConfig(**checkpoint["config"])
        requested = apply_architecture_args(config, args)
        if requested != config:
            raise ValueError("resume architecture overrides differ from checkpoint")
    else:
        config = apply_architecture_args(
            build_config(
                args.profile,
                dataset.feature_dim,
                dataset.condition_dim,
                rgb_enabled,
                args,
            ),
            args,
        )
    validate_data_model_contract(
        config,
        dataset,
        language_enabled,
        rgb_enabled,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    elif args.init_from:
        init_checkpoint = torch.load(
            args.init_from,
            map_location="cpu",
            weights_only=False, mmap=True,
        )
        validate_v28_initialization(init_checkpoint, args)
        warm_start_report = warm_start_model(model, init_checkpoint)
        if context.is_main:
            report_path = os.path.join(args.out, "warm_start_report.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(warm_start_report, handle, indent=2, sort_keys=True)
    if is_v28(args):
        configure_v28_stage(model, args)
    elif args.posterior_dynamics_gate:
        configure_posterior_dynamics_gate(model, args.posterior_update_scope)
    elif args.posterior_core_training:
        configure_posterior_core_training(model)
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if context.distributed
        else model
    )
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    sampler = build_training_sampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        seed=args.seed,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    if len(loader) < args.grad_accum:
        raise ValueError("not enough batches per rank for one optimizer step")
    balanced_updates = (
        sampler.num_samples // (args.batch * args.grad_accum)
        if getattr(dataset, "balance_sampling", False)
        else 0
    )
    if (
        args.posterior_core_training
        and args.joint_steps < balanced_updates
    ):
        raise ValueError(
            "posterior Core steps do not cover one balanced data epoch: "
            f"{args.joint_steps} < {balanced_updates}"
        )
    optimizer = build_optimizer(model, args)
    total_steps = args.representation_steps + args.joint_steps
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else round(total_steps * args.warmup_fraction)
    )
    scheduler = cosine_schedule(
        optimizer,
        warmup_steps,
        total_steps,
        0.1 if is_v28(args) else args.lr_floor / args.lr,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint, context)
    weights = v28_loss_weights(args) or staged_loss_weights(
        args.posterior_dynamics_gate, args.posterior_core_training
    )
    representation_step = 0
    joint_step = 0
    global_step = 0
    if checkpoint is not None:
        global_step = int(checkpoint["global_step"])
        if checkpoint["phase"] == "representation":
            representation_step = int(checkpoint["phase_step"])
        else:
            representation_step = args.representation_steps
            joint_step = int(checkpoint["phase_step"])
    if context.is_main:
        effective_batch = args.batch * context.world_size * args.grad_accum
        runtime_metadata = v28_runtime_metadata(args, dataset, gate_report)
        if is_v28(args):
            with open(
                os.path.join(args.out, "run_contract.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(runtime_metadata, handle, indent=2, sort_keys=True)
        print(
            f"[adaptive-wm] architecture={args.architecture} "
            f"stage={args.training_stage} profile={args.profile} "
            f"world={context.world_size} "
            f"effective_batch={effective_batch} examples={len(dataset)} "
            f"data_format={args.data_format} "
            f"language={language_enabled} rgb={rgb_enabled} "
            f"posterior_gate={args.posterior_dynamics_gate} "
            f"posterior_core={args.posterior_core_training} "
            f"posterior_update_scope={args.posterior_update_scope} "
            f"action_dim={config.action_dim} "
            f"canonical_center_gate={config.canonical_center_gate} "
            f"action_residual_dim={config.action_residual_dim} "
            f"action_residual_gate={config.action_residual_gate} "
            f"action_residual_dropout={config.action_residual_dropout} "
            f"semantic_action_basis={args.semantic_action_basis} "
            f"zero_action_margin={config.zero_action_margin_weight} "
            f"warmup_steps={warmup_steps} "
            f"recovery_every={args.recovery_every} save_every={args.save_every} "
            f"sampler={type(sampler).__name__} "
            f"balanced_updates_per_epoch={balanced_updates} "
            f"parallelism=DDP checkpoint=v{CHECKPOINT_VERSION}_full_state_dict",
            flush=True,
        )
    experiment_tracker = init_wandb_tracker(
        args,
        context,
        {
            "arguments": vars(args),
            "model": config.to_dict(),
            "dataset": {
                "examples": len(dataset),
                "feature_dim": dataset.feature_dim,
                "condition_dim": dataset.condition_dim,
                "contract": dataset.contract_label,
                "sha256": args.sequence_data_sha256,
                "teacher_sidecar_sha256": args.teacher_sidecar_sha256,
            },
            "runtime": {
                "checkpoint_version": CHECKPOINT_VERSION,
                "parallelism": "ddp_full_state_dict",
                "world_size": context.world_size,
                "effective_batch": (
                    args.batch * context.world_size * args.grad_accum
                ),
                "warmup_steps": warmup_steps,
                "balanced_updates_per_epoch": balanced_updates,
                **v28_runtime_metadata(args, dataset, gate_report),
            },
        },
    )
    representation_step, global_step = train_phase(
        "representation",
        args.representation_steps,
        representation_step,
        global_step,
        model,
        wrapped,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        weights,
        experiment_tracker,
    )
    joint_step, global_step = train_phase(
        "posterior" if args.training_stage == "posterior" else "joint",
        args.joint_steps,
        joint_step,
        global_step,
        model,
        wrapped,
        loader,
        sampler,
        optimizer,
        scheduler,
        context,
        args,
        weights,
        experiment_tracker,
    )
    if experiment_tracker is not None:
        experiment_tracker.finish()
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
