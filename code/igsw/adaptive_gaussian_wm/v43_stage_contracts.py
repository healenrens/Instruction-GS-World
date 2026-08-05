"""Warm-start boundary from v42 roots into the v43 object-region model."""
from __future__ import annotations


ARCHITECTURE = "object_region_memory_v1"
SOURCE_ARCHITECTURE = "object_memory_v3"
SOURCE_CHECKPOINT_VERSION = 42


def validate_v43_initialization(checkpoint: dict, args) -> None:
    if checkpoint.get("checkpoint_version") != SOURCE_CHECKPOINT_VERSION:
        raise ValueError("v43 init_from requires a version-42 checkpoint")
    if checkpoint.get("config", {}).get("architecture") != SOURCE_ARCHITECTURE:
        raise ValueError("v43 init_from requires an object_memory_v3 source")
    if checkpoint.get("phase") != "representation":
        raise ValueError("v43 init_from requires a v42 representation checkpoint")
    if args.architecture != ARCHITECTURE or args.training_stage != "representation":
        raise ValueError("v43 is a single-run representation-stage architecture")


def validate_v43_warm_start_report(report: dict, args) -> None:
    if args.architecture != ARCHITECTURE:
        raise ValueError("v43 warm-start report used by another architecture")
    allowed_missing = (
        "curriculum_step",
        "online_dino.",
        "target_dino.",
        "region_transformer.",
        "target_region_transformer.",
        "region_memory.",
        "target_region_memory.",
        "region_dynamics.",
        "region_effect_posterior.",
    )
    invalid_missing = [
        name
        for name in report["missing"]
        if name != allowed_missing[0]
        and not name.startswith(allowed_missing[1:])
    ]
    invalid_unexpected = [
        name
        for name in report["unexpected"]
        if not name.startswith(("change_readout.", "latent_actions."))
    ]
    if invalid_missing or invalid_unexpected or report["shape_mismatch"]:
        raise ValueError(
            "v43 warm start has incompatible parameters: "
            f"missing={invalid_missing}, unexpected={invalid_unexpected}, "
            f"shape_mismatch={report['shape_mismatch']}"
        )
