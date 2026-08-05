"""Runtime metadata shared by Object Memory training generations."""
from __future__ import annotations

import os

from .checkpointing import CHECKPOINT_VERSION
from .v42_runtime_contracts import runtime_contract_fields


OBJECT_REGION_ARCHITECTURE = "object_region_memory_v1"


def build_v28_runtime_metadata(args, dataset, gate: dict, enabled: bool) -> dict:
    if not enabled:
        return {}
    is_region = args.architecture == OBJECT_REGION_ARCHITECTURE
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "architecture": args.architecture,
        "training_stage": args.training_stage,
        **runtime_contract_fields(args.architecture),
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
        "feature_contract": (
            "model_owned_trainable_dinov2_l_1024_region_768"
            if is_region
            else "jit_backbone_native_dinov2_l_1024"
        ),
        "jit_dino_model": "vit_large_patch14_dinov2.lvd142m",
        "jit_dino_image_size": 518,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "dino_trainable_blocks": 12 if is_region else 0,
        "dino_target": "ema_no_grad" if is_region else "external_frozen",
        "region_feature_dim": 768 if is_region else 0,
        "curriculum_boundaries": [5000, 20000, 50000] if is_region else [],
        "control_hz": float(dataset.control_hz),
        "core_lr": args.core_lr,
        "action_lr": args.action_lr,
        "dino_lr": args.dino_lr,
        "new_module_lr": args.new_module_lr,
        "readout_lr": args.readout_lr,
        "readout_scope": args.readout_scope,
        "current_readout_weight": args.current_readout_weight,
        "readout_regularization_weight": args.readout_regularization_weight,
        "carrier_support_weight": args.carrier_support_weight,
        "carrier_compact_weight": args.carrier_compact_weight,
        "gaussian_children": args.gaussian_children,
        "readout_backend": (
            "offline_probe_only" if is_region else "change_only_object_residual"
        ),
        "basis_gate_report": _absolute(args.basis_gate_report),
        "carrier_preflight_report": _absolute(args.carrier_preflight_report),
        "dense_preflight_report": _absolute(args.dense_preflight_report),
        "dense_preflight_report_sha256": args.dense_preflight_report_sha256,
        "readout_gate_report": _absolute(args.readout_gate_report),
        "readout_gate_report_sha256": args.readout_gate_report_sha256,
        "representation_gate_report": _absolute(args.representation_gate_report),
        "representation_gate_report_sha256": args.representation_gate_report_sha256,
        "posterior_gate_report": _absolute(args.posterior_gate_report),
        "posterior_gate_report_sha256": args.posterior_gate_report_sha256,
        "gpu_policy": "auto",
        "target_global_batch": args.target_global_batch,
        "effective_global_batch": args.effective_global_batch,
        "teacher_sidecar": (
            "audit_only" if is_region and args.teacher_sidecar
            else "enabled" if args.teacher_sidecar
            else "disabled"
        ),
        "teacher_sidecar_sha256": getattr(dataset, "teacher_sidecar_sha256", ""),
        "disabled_teacher_losses": (
            ["relative_disparity_teacher", "visibility_teacher"]
            if is_region
            else [] if args.teacher_sidecar
            else ["relative_disparity", "visibility"]
        ),
        "gate_report": _absolute(args.gate_report),
        "gate_report_sha256": args.gate_report_sha256,
        "gate_git_commit": gate.get("git_commit", ""),
    }


def _absolute(path: str) -> str:
    return os.path.abspath(path) if path else ""
