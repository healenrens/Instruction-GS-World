"""Immutable report contracts for staged hierarchical-carrier training."""
from __future__ import annotations

import hashlib
import json


READOUT_GATE_CONTRACT = "hierarchical_gaussian_carrier_held_v1"
BASIS_GATE_CONTRACT = "hierarchical_gaussian_basis_v1"
CARRIER_PREFLIGHT_CONTRACT = "hierarchical_gaussian_carrier_preflight_v1"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_fields(report: dict, expected: dict, label: str) -> None:
    mismatch = {
        name: {"gate": report.get(name), "current": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    if mismatch:
        raise ValueError(f"{label} differs: {mismatch}")


def validate_readout_gate(args, initialization_path: str, mode: str) -> None:
    with open(args.readout_gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    with open(args.gate_report, encoding="utf-8") as handle:
        base_gate = json.load(handle)
    _require_fields(
        report,
        {
            "status": "passed",
            "contract": READOUT_GATE_CONTRACT,
            "evaluation_mode": mode,
            "git_commit": base_gate.get("git_commit"),
            "data_manifest_sha256": args.sequence_data_sha256,
            "candidate_checkpoint_sha256": file_sha256(initialization_path),
        },
        "held readout gate",
    )


def validate_basis_gate(args, initialization_path: str) -> None:
    with open(args.basis_gate_report, encoding="utf-8") as handle:
        report = json.load(handle)
    with open(args.gate_report, encoding="utf-8") as handle:
        base_gate = json.load(handle)
    _require_fields(
        report,
        {
            "status": "passed",
            "contract": BASIS_GATE_CONTRACT,
            "git_commit": base_gate.get("git_commit"),
            "data_manifest_sha256": args.sequence_data_sha256,
            "source_checkpoint_sha256": file_sha256(initialization_path),
            "selected_children": args.gaussian_children,
        },
        "hierarchical basis gate",
    )


def validate_carrier_preflight(args, initialization_path: str) -> None:
    with open(args.carrier_preflight_report, encoding="utf-8") as handle:
        report = json.load(handle)
    with open(args.gate_report, encoding="utf-8") as handle:
        base_gate = json.load(handle)
    _require_fields(
        report,
        {
            "status": "passed",
            "contract": CARRIER_PREFLIGHT_CONTRACT,
            "git_commit": base_gate.get("git_commit"),
            "data_manifest_sha256": args.sequence_data_sha256,
            "source_checkpoint_sha256": file_sha256(initialization_path),
            "selected_children": args.gaussian_children,
            "basis_gate_sha256": file_sha256(args.basis_gate_report),
        },
        "hierarchical carrier preflight",
    )
