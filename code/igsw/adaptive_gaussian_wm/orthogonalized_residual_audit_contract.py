"""Provenance and metric rows for the v36 orthogonalized residual audit."""

from __future__ import annotations

from .object_centered_audit_runtime import append_row
from .residual_field_audit_contract import load_matching_report


V35_COMMIT = "4d5f6e1a1a509e5223d41da02c29c88e0d9c6e2c"


def load_v35_report(
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
            "decision": "reject_partition_of_unity_capacity",
            "contract": "partition_of_unity_object_field_capacity_v1",
            "git_commit": V35_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "data_manifest_sha256": data_sha256,
            "held_split": split,
            "held_items": held_items,
        },
    )


def append_orthogonalized_fit_rows(rows: dict, prefix: str, fit, budgets) -> None:
    partition = fit.partition
    reference = partition.reference
    append_row(rows, f"{prefix}_geometric_root", reference.geometric_root_error)
    append_row(rows, f"{prefix}_geometric_support_js", partition.geometric_support_js)
    for budget in budgets:
        old = partition.budgets[budget]
        solution = fit.solutions[budget]
        append_row(
            rows, f"{prefix}_gated_b{budget}", reference.geometric_errors[budget]
        )
        append_row(rows, f"{prefix}_oracle_b{budget}", reference.oracle_errors[budget])
        append_row(rows, f"{prefix}_partition_b{budget}", old.error)
        append_row(
            rows,
            f"{prefix}_partition_support_js_b{budget}",
            old.support_js,
        )
        append_row(rows, f"{prefix}_orthogonal_b{budget}", solution.error)
        append_row(
            rows,
            f"{prefix}_orthogonal_condition_b{budget}",
            solution.effective_condition,
        )
        append_row(
            rows,
            f"{prefix}_orthogonal_coefficient_rms_b{budget}",
            solution.coefficient_rms,
        )
        append_row(
            rows,
            f"{prefix}_orthogonal_energy_b{budget}",
            solution.retained_energy,
        )
        append_row(
            rows,
            f"{prefix}_orthogonal_rank_fraction_b{budget}",
            solution.retained_rank.float() / solution.full_rank,
        )


def v35_comparison(v35: dict, statistics: dict, budgets) -> dict:
    old = v35["means"]
    new = statistics["means"]
    differences = []
    result = {
        "v35_git_commit": v35["git_commit"],
        "v35_decision": v35["decision"],
        "v35_interpretation": "adaptive_support_valid_independent_experts_over_smooth",
    }
    for prefix in ("current", "future"):
        for name in ("token", "geometric_root", "geometric_support_js"):
            differences.append(abs(old[f"{prefix}_{name}"] - new[f"{prefix}_{name}"]))
        for budget in budgets:
            for method in ("gated", "oracle", "partition"):
                key = f"{prefix}_{method}_b{budget}"
                differences.append(abs(old[key] - new[key]))
            support_key = f"{prefix}_partition_support_js_b{budget}"
            differences.append(abs(old[support_key] - new[support_key]))
            result[f"{prefix}_orthogonal_b{budget}_gain_over_v35_partition"] = (
                old[f"{prefix}_partition_b{budget}"]
                - new[f"{prefix}_orthogonal_b{budget}"]
            )
    result["v35_reproduction_max_abs_difference"] = max(differences)
    return result
