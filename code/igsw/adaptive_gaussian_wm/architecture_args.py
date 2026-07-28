"""CLI architecture overrides shared by adaptive world-model trainers."""
from __future__ import annotations

from dataclasses import replace

from .config import AdaptiveGaussianWMConfig


def build_config(
    profile: str,
    feature_dim: int,
    condition_dim: int,
    rgb_supervision: bool,
    args,
) -> AdaptiveGaussianWMConfig:
    constructors = {
        "tiny": AdaptiveGaussianWMConfig.tiny,
        "probe": AdaptiveGaussianWMConfig.probe,
        "full": AdaptiveGaussianWMConfig.full,
    }
    constructor = (
        AdaptiveGaussianWMConfig.object_memory_full
        if args.architecture == "object_memory_v1"
        else constructors[profile]
    )
    return replace(
        constructor(feature_dim),
        condition_dim=condition_dim,
        rgb_supervision=rgb_supervision,
        rgb_short_side=args.rgb_short_side,
        rgb_pad_multiple=args.rgb_pad_multiple,
        rgb_render_chunk=args.rgb_render_chunk,
        rgb_loss_weight=args.rgb_loss_weight,
        rgb_ssim_weight=args.rgb_ssim_weight,
        rgb_change_loss_weight=args.rgb_change_loss_weight,
        rgb_change_threshold=args.rgb_change_threshold,
        language_effect_weight=args.language_effect_weight,
        zero_action_margin_weight=args.zero_action_margin_weight,
        zero_action_relative_margin=args.zero_action_relative_margin,
    )


def apply_architecture_args(
    config: AdaptiveGaussianWMConfig,
    args,
) -> AdaptiveGaussianWMConfig:
    overrides = {}
    if args.architecture == "object_memory_v1":
        overrides.update(
            gaussian_children=args.gaussian_children,
            hierarchical_gaussian_carrier=args.gaussian_children > 1,
        )
    if args.aggregation_mode != "auto":
        overrides["aggregation_mode"] = args.aggregation_mode
    if args.density_mode != "auto":
        overrides["density_mode"] = args.density_mode
    if args.joint_flow != "auto":
        overrides["joint_flow"] = args.joint_flow == "on"
    if args.posterior_dynamics_gate:
        overrides["action_query_modulation"] = True
    if args.action_anchor == "object_slot":
        residual_dim = (
            config.action_dim - 6
            if args.action_residual_dim < 0
            else args.action_residual_dim
        )
        overrides.update(
            action_tokens=config.object_slots,
            action_dim=6 + residual_dim,
            object_aligned_actions=True,
            canonical_center_action=True,
            canonical_center_gate=args.canonical_center_gate,
            canonical_semantic_action=True,
            canonical_activity_gate=args.canonical_activity_gate,
            canonical_activity_power=args.canonical_activity_power,
            bounded_residual_action=True,
            action_query_modulation=True,
            prior_query_residual=True,
        )
        if args.semantic_action_basis == "learned":
            overrides.update(
                learned_semantic_action_basis=True,
                semantic_action_basis_weight=0.1,
            )
        elif args.semantic_action_basis == "rgb":
            overrides["rgb_semantic_action"] = True
        if residual_dim == 0 and (
            args.action_residual_gate < 1.0
            or args.action_residual_dropout > 0.0
        ):
            raise ValueError("canonical-only actions cannot gate or drop a residual")
        overrides.update(
            action_residual_gate=args.action_residual_gate,
            action_residual_dropout=args.action_residual_dropout,
        )
    elif args.action_residual_dim >= 0:
        raise ValueError(
            "--action_residual_dim requires --action_anchor object_slot"
        )
    elif args.action_residual_gate != 1.0:
        raise ValueError(
            "--action_residual_gate requires --action_anchor object_slot"
        )
    elif args.action_residual_dropout != 0.0:
        raise ValueError(
            "--action_residual_dropout requires --action_anchor object_slot"
        )
    elif args.canonical_center_gate != 1.0:
        raise ValueError(
            "--canonical_center_gate requires --action_anchor object_slot"
        )
    elif args.canonical_activity_gate:
        raise ValueError(
            "--canonical_activity_gate requires --action_anchor object_slot"
        )
    elif args.canonical_activity_power != 0.5:
        raise ValueError(
            "--canonical_activity_power requires --action_anchor object_slot"
        )
    elif args.semantic_action_basis != "fixed":
        raise ValueError(
            "--semantic_action_basis requires --action_anchor object_slot"
        )
    return replace(config, **overrides)
