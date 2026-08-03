"""Initialization and promotion contracts for Object Memory JEPA v42."""

from __future__ import annotations

import json
import os

from .checkpointing import CHECKPOINT_VERSION
from .dense_readout_contracts import file_sha256
from .v40_stage_contracts import (
    V40_NEW_PARAMETERS,
    V40_REINITIALIZED_PARAMETERS,
)
from .v41_stage_contracts import V41_NEW_PARAMETERS


ARCHITECTURE = "object_memory_v3"


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
        "contract": f"object_memory_v42_{stage}_held_v1",
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
        raise ValueError(f"v42 {stage} gate differs: {mismatch}")
    digest = file_sha256(report_path)
    if stage == "representation":
        args.representation_gate_report_sha256 = digest
    else:
        args.posterior_gate_report_sha256 = digest


def _validate_representation_source(checkpoint: dict) -> None:
    version = int(checkpoint.get("checkpoint_version", 0))
    architecture = checkpoint.get("config", {}).get("architecture")
    expected_architecture = {
        39: "object_memory_v1",
        40: "object_memory_v2",
    }.get(version)
    if expected_architecture is None or architecture != expected_architecture:
        raise ValueError(
            "v42 representation requires a clean v39 or v40 representation; "
            "v41 checkpoints are intentionally rejected"
        )
    saved = checkpoint.get("args", {})
    if (
        checkpoint.get("phase") != "representation"
        or saved.get("training_stage") != "representation"
        or int(checkpoint.get("phase_step", 0)) <= 0
    ):
        raise ValueError("v42 representation source has no completed update")


def validate_v42_initialization(checkpoint: dict, args) -> None:
    if args.training_stage == "representation":
        _validate_representation_source(checkpoint)
        return
    if int(checkpoint.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError("v42 posterior/prior initialization requires checkpoint v42")
    if checkpoint.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError("v42 initialization architecture differs")
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
            f"{args.training_stage} must start from completed v42 {source_stage}"
        )
    _validate_stage_gate(args, source_stage)


def validate_v42_warm_start_report(report: dict, args) -> None:
    if report["unexpected"] or report["shape_mismatch"] or report["dropped"]:
        raise ValueError("v42 warm start has unexpected or incompatible parameters")
    missing = set(report["missing"])
    transformed = set(report["transformed"])
    if args.training_stage != "representation":
        if missing or transformed:
            raise ValueError("v42 stage promotion must load the complete model state")
        return
    source_version = int(report["source_checkpoint_version"])
    if source_version == 39:
        expected_missing = V40_NEW_PARAMETERS | V41_NEW_PARAMETERS
        expected_transformed = V40_REINITIALIZED_PARAMETERS
    elif source_version == 40:
        expected_missing = V41_NEW_PARAMETERS
        expected_transformed = set()
    else:
        raise ValueError("v42 warm start source version is unsupported")
    if missing != expected_missing:
        raise ValueError(
            f"v42 warm-start missing parameters differ: {sorted(missing)}"
        )
    if transformed != expected_transformed:
        raise ValueError("v42 warm-start semantic transforms differ")
