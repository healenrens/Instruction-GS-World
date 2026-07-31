"""Promotion gates between v39 representation, posterior, and prior stages."""

from __future__ import annotations

import json
import os

from .checkpointing import CHECKPOINT_VERSION
from .dense_readout_contracts import file_sha256


ARCHITECTURE = "object_memory_v1"


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
        "contract": f"object_memory_v39_{stage}_held_v1",
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
        raise ValueError(f"v39 {stage} gate differs: {mismatch}")
    digest = file_sha256(report_path)
    if stage == "representation":
        args.representation_gate_report_sha256 = digest
    else:
        args.posterior_gate_report_sha256 = digest


def validate_v39_initialization(checkpoint: dict, args) -> None:
    version = int(checkpoint.get("checkpoint_version", 0))
    if version > CHECKPOINT_VERSION:
        raise ValueError("cannot initialize v39 from a newer checkpoint")
    if args.training_stage == "representation":
        raise ValueError("v39 representation starts from scratch; use resume to continue")
    if checkpoint.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint initialization architecture differs")
    source_stage = (
        "representation" if args.training_stage == "posterior" else "posterior"
    )
    saved = checkpoint.get("args", {})
    if (
        version != CHECKPOINT_VERSION
        or checkpoint.get("phase") != source_stage
        or saved.get("training_stage") != source_stage
        or not _stage_complete(checkpoint)
    ):
        raise ValueError(
            f"{args.training_stage} must start from completed v39 {source_stage}"
        )
    _validate_stage_gate(args, source_stage)
