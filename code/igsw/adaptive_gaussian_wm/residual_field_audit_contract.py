"""Provenance and row contracts for the v33 residual-field audit."""

from __future__ import annotations

import json

import torch

from .object_centered_audit_runtime import append_row, require


V32_COMMIT = "1d8545edcf076c11d515997770c10ce2ec5e67e8"


def load_matching_report(path: str, expected: dict) -> dict:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    mismatch = {
        name: {"report": report.get(name), "expected": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    require(not mismatch, f"baseline report differs: {mismatch}")
    return report


def load_v32_report(
    path: str,
    checkpoint_sha256: str,
    data_sha256: str,
    split: str,
    held_items: int,
) -> dict:
    return load_matching_report(
        path,
        {
            "status": "completed",
            "decision": "reject_attention_carrier_parameterization",
            "contract": "object_centered_attention_carrier_capacity_v1",
            "git_commit": V32_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "data_manifest_sha256": data_sha256,
            "held_split": split,
            "held_items": held_items,
        },
    )


def append_fit_rows(rows: dict, prefix: str, fit, budgets: tuple[int, ...]) -> None:
    append_row(rows, f"{prefix}_global_root", fit.global_root_error)
    append_row(rows, f"{prefix}_oracle_root", fit.oracle_root_error)
    append_row(rows, f"{prefix}_geometric_root", fit.geometric_root_error)
    for budget in budgets:
        append_row(rows, f"{prefix}_global_b{budget}", fit.global_errors[budget])
        append_row(rows, f"{prefix}_oracle_b{budget}", fit.oracle_errors[budget])
        append_row(rows, f"{prefix}_geometric_b{budget}", fit.geometric_errors[budget])
        for method in ("global", "oracle", "geometric"):
            key = f"{method}_b{budget}"
            append_row(
                rows,
                f"{prefix}_{method}_condition_b{budget}",
                fit.condition_numbers[key],
            )
            append_row(
                rows,
                f"{prefix}_{method}_coefficient_rms_b{budget}",
                fit.coefficient_rms[key],
            )
    append_row(rows, f"{prefix}_global_selected", fit.selected_global_carriers)
    append_row(rows, f"{prefix}_object_selected", fit.selected_object_carriers)
    append_row(rows, f"{prefix}_scene_selected", fit.selected_scene_carriers)
    append_row(rows, f"{prefix}_global_halted", float(fit.global_stopped_by_utility))
    append_row(rows, f"{prefix}_object_halted", float(fit.object_stopped_by_utility))


def allocation_standard_deviation(fit) -> torch.Tensor:
    active = fit.local_carriers_per_object[fit.geometry.object_active].float()
    return active.std(unbiased=False) if active.numel() > 1 else active.new_zeros(())


def baseline_comparison(v31: dict, v32: dict, statistics: dict) -> dict:
    means = statistics["means"]
    old31 = v31["means"]
    old32 = v32["means"]
    result = {
        "v31_git_commit": v31["git_commit"],
        "v31_decision": v31["decision"],
        "v32_git_commit": v32["git_commit"],
        "v32_decision": v32["decision"],
    }
    for prefix, old_current, old_future in (
        ("v31", old31["current_full_b64"], old31["future_oracle_b64"]),
        ("v32", old32["current_attention_b64"], old32["future_oracle_b64"]),
    ):
        result[f"{prefix}_current_b64"] = old_current
        result[f"{prefix}_future_b64"] = old_future
        for method in ("global", "oracle", "geometric"):
            current = means[f"current_{method}_b64"]
            future = means[f"future_{method}_b64"]
            result[f"{method}_current_b64_gain_over_{prefix}"] = old_current - current
            result[f"{method}_future_b64_gain_over_{prefix}"] = old_future - future
    return result
