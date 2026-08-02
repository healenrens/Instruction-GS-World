"""Initialization and promotion contracts for Object Memory JEPA v40."""

from __future__ import annotations

import json
import os

from .checkpointing import CHECKPOINT_VERSION
from .dense_readout_contracts import file_sha256


ARCHITECTURE = "object_memory_v2"
SOURCE_ARCHITECTURE = "object_memory_v1"
SOURCE_CHECKPOINT_VERSION = 39
V40_NEW_PARAMETERS = {
    "dynamics.action_observability_basis.0.bias",
    "dynamics.action_observability_basis.0.weight",
    "dynamics.action_observability_basis.1.bias",
    "dynamics.action_observability_basis.1.weight",
    "dynamics.action_observability_basis.3.bias",
    "dynamics.action_observability_basis.3.weight",
    "dynamics.action_observability_gate.weight",
    "dynamics.motion_input.0.bias",
    "dynamics.motion_input.0.weight",
    "dynamics.motion_input.2.bias",
    "dynamics.motion_input.2.weight",
    "dynamics.observability_output.bias",
    "dynamics.observability_output.weight",
    "object_aggregator.identity_anchor_projection.weight",
    "target_object_aggregator.identity_anchor_projection.weight",
}
V40_REINITIALIZED_PARAMETERS = {
    "dynamics.base_geometry_output.bias",
    "dynamics.base_geometry_output.weight",
    "dynamics.base_lifecycle_output.bias",
    "dynamics.base_lifecycle_output.weight",
    "object_memory.motion_head.3.bias",
    "object_memory.motion_head.3.weight",
    "target_object_memory.motion_head.3.bias",
    "target_object_memory.motion_head.3.weight",
}


def _stage_complete(checkpoint: dict) -> bool:
    saved = checkpoint.get("args", {})
    phase = checkpoint.get("phase")
    expected = (
        saved.get("representation_steps")
        if phase == "representation"
        else saved.get("joint_steps") if phase in ("posterior", "prior") else None
    )
    return expected is not None and int(checkpoint.get("phase_step", -1)) == int(
        expected
    )


def _validate_stage_gate(args, stage: str) -> None:
    report_path = (
        args.representation_gate_report
        if stage == "representation"
        else args.posterior_gate_report
    )
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "contract": f"object_memory_v40_{stage}_held_v1",
        "git_commit": args.git_commit,
        "data_manifest_sha256": args.sequence_data_sha256,
        "source_checkpoint": os.path.realpath(args.init_from),
        "source_checkpoint_sha256": file_sha256(args.init_from),
    }
    mismatch = {
        name: {"gate": report.get(name), "current": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    if mismatch:
        raise ValueError(f"v40 {stage} gate differs: {mismatch}")
    digest = file_sha256(report_path)
    if stage == "representation":
        args.representation_gate_report_sha256 = digest
    else:
        args.posterior_gate_report_sha256 = digest


def _validate_v39_representation_source(checkpoint: dict) -> None:
    saved = checkpoint.get("args", {})
    if int(checkpoint.get("checkpoint_version", 0)) != SOURCE_CHECKPOINT_VERSION:
        raise ValueError("v40 representation warm start requires checkpoint v39")
    if checkpoint.get("config", {}).get("architecture") != SOURCE_ARCHITECTURE:
        raise ValueError("v40 representation warm start requires object_memory_v1")
    if checkpoint.get("phase") != "representation":
        raise ValueError("v40 representation warm start requires representation state")
    if saved.get("training_stage") != "representation":
        raise ValueError("v39 source does not declare representation training")
    if int(checkpoint.get("phase_step", 0)) <= 0:
        raise ValueError("v39 representation source has no completed update")


def validate_v40_initialization(checkpoint: dict, args) -> None:
    if args.training_stage == "representation":
        _validate_v39_representation_source(checkpoint)
        return
    version = int(checkpoint.get("checkpoint_version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError("v40 posterior/prior initialization requires checkpoint v40")
    if checkpoint.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError("v40 initialization architecture differs")
    source_stage = (
        "representation" if args.training_stage == "posterior" else "posterior"
    )
    saved = checkpoint.get("args", {})
    if (
        checkpoint.get("phase") != source_stage
        or saved.get("training_stage") != source_stage
        or not _stage_complete(checkpoint)
    ):
        raise ValueError(
            f"{args.training_stage} must start from completed v40 {source_stage}"
        )
    _validate_stage_gate(args, source_stage)


def validate_v40_warm_start_report(report: dict, args) -> None:
    if report["unexpected"] or report["shape_mismatch"] or report["dropped"]:
        raise ValueError("v40 warm start has unexpected or incompatible parameters")
    missing = set(report["missing"])
    if args.training_stage == "representation":
        if missing != V40_NEW_PARAMETERS:
            raise ValueError(
                f"v39 to v40 warm start missing parameters differ: {sorted(missing)}"
            )
        if set(report["transformed"]) != V40_REINITIALIZED_PARAMETERS:
            raise ValueError("v39 to v40 semantic transforms differ")
        return
    if missing or report["transformed"]:
        raise ValueError("v40 stage promotion must load the complete model state")
