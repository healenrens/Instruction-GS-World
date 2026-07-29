"""Provenance and metric rows for the v35 partitioned-object audit."""

from __future__ import annotations

from .object_centered_audit_runtime import append_row
from .residual_field_audit_contract import load_matching_report


V34_COMMIT = "d78ccd4f219fc40567d535f681651a7c532729e3"


def load_v34_report(
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
            "decision": "reject_ungated_local_residual",
            "contract": "stable_object_residual_transport_capacity_v1",
            "git_commit": V34_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "data_manifest_sha256": data_sha256,
            "held_split": split,
            "held_items": held_items,
        },
    )


def append_partitioned_fit_rows(rows: dict, prefix: str, fit, budgets) -> None:
    reference = fit.reference
    append_row(rows, f"{prefix}_geometric_root", reference.geometric_root_error)
    append_row(rows, f"{prefix}_geometric_support_js", fit.geometric_support_js)
    for budget in budgets:
        partition = fit.budgets[budget]
        append_row(
            rows,
            f"{prefix}_gated_b{budget}",
            reference.geometric_errors[budget],
        )
        append_row(
            rows,
            f"{prefix}_oracle_b{budget}",
            reference.oracle_errors[budget],
        )
        append_row(rows, f"{prefix}_partition_b{budget}", partition.error)
        append_row(
            rows,
            f"{prefix}_partition_support_js_b{budget}",
            partition.support_js,
        )
        append_row(
            rows,
            f"{prefix}_partition_condition_b{budget}",
            partition.maximum_condition,
        )
        append_row(
            rows,
            f"{prefix}_partition_coefficient_rms_b{budget}",
            partition.coefficient_rms,
        )
        append_row(
            rows,
            f"{prefix}_partition_gate_sum_error_b{budget}",
            (partition.gates.sum(dim=-1) - 1.0).abs().max(),
        )
        append_row(
            rows,
            f"{prefix}_partition_gate_minimum_b{budget}",
            partition.gates.min(),
        )


def v34_comparison(v34: dict, statistics: dict, budgets) -> dict:
    old = v34["means"]
    new = statistics["means"]
    differences = []
    result = {
        "v34_git_commit": v34["git_commit"],
        "v34_decision": v34["decision"],
        "v34_corrected_interpretation": "replace_additive_transport_with_normalized_object_experts",
    }
    for prefix in ("current", "future"):
        for name in ("token", "geometric_root"):
            differences.append(abs(old[f"{prefix}_{name}"] - new[f"{prefix}_{name}"]))
        for budget in budgets:
            for method in ("gated", "oracle"):
                key = f"{prefix}_{method}_b{budget}"
                differences.append(abs(old[key] - new[key]))
            result[f"{prefix}_partition_b{budget}_gain_over_v34_stable"] = (
                old[f"{prefix}_stable_b{budget}"] - new[f"{prefix}_partition_b{budget}"]
            )
    result["v34_reproduction_max_abs_difference"] = max(differences)
    return result
