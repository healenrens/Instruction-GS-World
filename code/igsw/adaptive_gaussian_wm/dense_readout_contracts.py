"""Immutable reports for v30 dense object-readout promotion."""

from __future__ import annotations

import hashlib
import json


DENSE_PREFLIGHT_CONTRACT = "dense_object_readout_preflight_v1"
DENSE_HELD_CONTRACT = "dense_object_readout_held_v1"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _require(report: dict, expected: dict, label: str) -> None:
    mismatch = {
        name: {"report": report.get(name), "expected": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    if mismatch:
        raise ValueError(f"{label} differs: {mismatch}")


def validate_dense_preflight(args, initialization_path: str) -> None:
    report = _load(args.dense_preflight_report)
    base_gate = _load(args.gate_report)
    _require(
        report,
        {
            "status": "passed",
            "contract": DENSE_PREFLIGHT_CONTRACT,
            "git_commit": base_gate.get("git_commit"),
            "data_manifest_sha256": args.sequence_data_sha256,
            "teacher_sidecar_sha256": args.teacher_sidecar_sha256,
            "source_checkpoint_sha256": file_sha256(initialization_path),
            "base_gate_sha256": file_sha256(args.gate_report),
        },
        "dense readout preflight",
    )


def validate_dense_held_gate(
    args,
    initialization_path: str,
    mode: str,
) -> None:
    report = _load(args.readout_gate_report)
    base_gate = _load(args.gate_report)
    _require(
        report,
        {
            "status": "passed",
            "contract": DENSE_HELD_CONTRACT,
            "evaluation_mode": mode,
            "git_commit": base_gate.get("git_commit"),
            "data_manifest_sha256": args.sequence_data_sha256,
            "candidate_checkpoint_sha256": file_sha256(initialization_path),
        },
        "dense held gate",
    )
