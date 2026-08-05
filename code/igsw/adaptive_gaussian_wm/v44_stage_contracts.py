"""Initialization and runtime contracts for dual-encoder v44."""
from __future__ import annotations

import os

from .video_vae_contract import (
    VIDEO_VAE_CONTRACT,
    VIDEO_VAE_DIFFUSERS_VERSION,
    VIDEO_VAE_MODEL_ID,
    file_sha256,
    validate_video_vae_artifact,
)


ARCHITECTURE = "object_region_dual_encoder_v1"
SOURCE_ARCHITECTURE = "object_region_memory_v1"
SOURCE_CHECKPOINT_VERSION = 43
TEMPORAL_CONTRACT = "dynamic_dual_horizon_video_v2"


def validate_v44_initialization(checkpoint: dict, args) -> None:
    if checkpoint.get("checkpoint_version") != SOURCE_CHECKPOINT_VERSION:
        raise ValueError("v44 init_from requires a version-43 checkpoint")
    if checkpoint.get("config", {}).get("architecture") != SOURCE_ARCHITECTURE:
        raise ValueError("v44 init_from requires object_region_memory_v1")
    if checkpoint.get("phase") != "representation":
        raise ValueError("v44 init_from requires a representation checkpoint")
    if args.architecture != ARCHITECTURE or args.training_stage != "representation":
        raise ValueError("v44 is a single-run representation-stage architecture")


def validate_v44_warm_start_report(report: dict, args) -> None:
    if args.architecture != ARCHITECTURE:
        raise ValueError("v44 warm-start report used by another architecture")
    allowed_missing = ("region_memory.video_", "target_region_memory.video_")
    invalid_missing = [
        name for name in report["missing"] if not name.startswith(allowed_missing)
    ]
    if invalid_missing or report["unexpected"] or report["shape_mismatch"]:
        raise ValueError(
            "v44 warm start has incompatible parameters: "
            f"missing={invalid_missing}, unexpected={report['unexpected']}, "
            f"shape_mismatch={report['shape_mismatch']}"
        )


def _validate_diffusers_runtime(args) -> None:
    if not os.path.isabs(args.video_vae_pythonpath):
        raise ValueError("v44 --video_vae_pythonpath must be absolute")
    if not os.path.isdir(args.video_vae_pythonpath):
        raise ValueError("v44 video VAE Python package directory is missing")
    import diffusers

    if diffusers.__version__ != VIDEO_VAE_DIFFUSERS_VERSION:
        raise ValueError(
            "v44 diffusers version differs: "
            f"{diffusers.__version__} != {VIDEO_VAE_DIFFUSERS_VERSION}"
        )
    package = os.path.realpath(diffusers.__file__)
    runtime = os.path.realpath(args.video_vae_pythonpath)
    if os.path.commonpath((package, runtime)) != runtime:
        raise ValueError("v44 diffusers was not imported from video_vae_pythonpath")


def validate_v44_arguments(args) -> None:
    if args.temporal_contract != TEMPORAL_CONTRACT:
        raise ValueError("v44 requires five-frame dual-encoder temporal samples")
    for name in ("video_vae_model", "video_vae_contract"):
        path = getattr(args, name)
        if not os.path.isabs(path):
            raise ValueError(f"v44 --{name} must be absolute")
    if (args.video_vae_short_side, args.video_vae_clip_frames) != (256, 5):
        raise ValueError("v44 requires 256px, five-frame Wan VAE clips")
    if args.video_vae_batch < 1:
        raise ValueError("v44 video VAE batch must be positive")
    _validate_diffusers_runtime(args)
    validate_video_vae_artifact(
        args.video_vae_model,
        args.video_vae_contract,
        verify_hashes=False,
    )


def v44_gate_fields(args) -> dict[str, object]:
    _validate_diffusers_runtime(args)
    args.video_vae_contract_sha256 = file_sha256(args.video_vae_contract)
    contract = validate_video_vae_artifact(
        args.video_vae_model,
        args.video_vae_contract,
        verify_hashes=False,
    )
    return {
        "video_vae_contract": VIDEO_VAE_CONTRACT,
        "video_vae_model_id": VIDEO_VAE_MODEL_ID,
        "video_vae_revision": contract["revision"],
        "video_vae_contract_sha256": args.video_vae_contract_sha256,
        "video_vae_model": os.path.abspath(args.video_vae_model),
        "video_vae_pythonpath": os.path.abspath(args.video_vae_pythonpath),
        "video_vae_diffusers_version": VIDEO_VAE_DIFFUSERS_VERSION,
        "video_vae_latent_dim": 48,
        "video_vae_feature_dim": 96,
        "video_vae_clip_frames": args.video_vae_clip_frames,
        "video_vae_short_side": args.video_vae_short_side,
        "video_vae_batch": args.video_vae_batch,
        "video_vae_trainable": False,
        "video_vae_checkpointed": False,
        "dual_encoder_contract": "dino_semantic_wan_detail_motion_v1",
        "minimum_goal_tail_frames": 4,
    }
