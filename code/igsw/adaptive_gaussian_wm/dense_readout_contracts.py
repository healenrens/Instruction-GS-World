"""Immutable reports for v30 dense object-readout promotion."""

from __future__ import annotations

import hashlib
import json


DENSE_PREFLIGHT_CONTRACT = "dense_object_readout_preflight_v2"
DENSE_HELD_CONTRACT = "dense_object_readout_held_v1"
GAUSSIAN_COMPATIBILITY_PARAMETERS = frozenset(
    {
        "gaussian_readout.feature_residual_head.weight",
        "gaussian_readout.feature_residual_head.bias",
    }
)


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


def validate_dense_warm_start(model, checkpoint: dict, report: dict) -> dict:
    """Require an exact transition from the frozen parent into dense readout."""
    if not model.config.dense_object_readout:
        raise ValueError("dense warm start requires the dense readout backend")
    source_config = checkpoint.get("config", {})
    target_names = set(model.state_dict())
    expected_missing = set()
    if not source_config.get("dense_object_readout", False):
        expected_missing = {
            name for name in target_names if name.startswith("dense_readout.")
        }
    compatibility_missing = set()
    source_gaussian_residual = source_config.get("gaussian_feature_residual", False)
    if model.config.gaussian_feature_residual and not source_gaussian_residual:
        compatibility_missing = GAUSSIAN_COMPATIBILITY_PARAMETERS & target_names
        if compatibility_missing != GAUSSIAN_COMPATIBILITY_PARAMETERS:
            raise ValueError("Gaussian compatibility parameters are incomplete")
        expected_missing.update(compatibility_missing)
    elif source_gaussian_residual and not model.config.gaussian_feature_residual:
        raise ValueError("dense target removed source Gaussian repair parameters")
    problems = {
        "missing": {
            "actual": sorted(report["missing"]),
            "expected": sorted(expected_missing),
        },
        "unexpected": report["unexpected"],
        "shape_mismatch": report["shape_mismatch"],
        "transformed": report["transformed"],
        "dropped": report["dropped"],
    }
    invalid = {
        name: value
        for name, value in problems.items()
        if value and name != "missing"
    }
    if set(report["missing"]) != expected_missing:
        invalid["missing"] = problems["missing"]
    if invalid:
        raise ValueError(f"dense warm start contract differs: {invalid}")
    return {
        "source_dense_object_readout": bool(
            source_config.get("dense_object_readout", False)
        ),
        "source_gaussian_feature_residual": bool(source_gaussian_residual),
        "expected_missing": sorted(expected_missing),
        "zero_initialized_missing": sorted(compatibility_missing),
    }


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
