"""Provenance and metric rows for the v34 stable-transport audit."""

from __future__ import annotations

from .object_centered_audit_runtime import append_row
from .residual_field_audit_contract import load_matching_report


V33_COMMIT = "150855ae91dc528470bfaa403a4fbeb6ed1e1295"


def load_v33_report(
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
            "contract": "object_centered_signed_residual_field_capacity_v1",
            "git_commit": V33_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "data_manifest_sha256": data_sha256,
            "held_split": split,
            "held_items": held_items,
        },
    )


def append_stable_fit_rows(rows: dict, prefix: str, fit, budgets) -> None:
    reference = fit.reference
    append_row(rows, f"{prefix}_geometric_root", reference.geometric_root_error)
    for budget in budgets:
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
        append_row(rows, f"{prefix}_stable_b{budget}", fit.solutions[budget].error)
        append_row(
            rows,
            f"{prefix}_stable_condition_b{budget}",
            fit.solutions[budget].condition_number,
        )
        append_row(
            rows,
            f"{prefix}_stable_coefficient_rms_b{budget}",
            fit.solutions[budget].coefficient_rms,
        )
        append_row(
            rows,
            f"{prefix}_gated_coefficient_rms_b{budget}",
            reference.coefficient_rms[f"geometric_b{budget}"],
        )
        append_row(
            rows,
            f"{prefix}_oracle_coefficient_rms_b{budget}",
            reference.coefficient_rms[f"oracle_b{budget}"],
        )


def append_sensitivity_rows(rows: dict, values: dict[str, object]) -> None:
    for name, value in values.items():
        append_row(rows, f"current_{name}_sensitivity", value)


def v33_comparison(v33: dict, statistics: dict, budgets) -> dict:
    old = v33["means"]
    new = statistics["means"]
    reproduction_differences = []
    result = {
        "v33_git_commit": v33["git_commit"],
        "v33_reported_decision": v33["decision"],
        "v33_corrected_interpretation": "repair_geometric_object_gate",
    }
    for budget in budgets:
        for prefix in ("current", "future"):
            gated = old[f"{prefix}_geometric_b{budget}"]
            stable = new[f"{prefix}_stable_b{budget}"]
            reproduction_differences.append(
                abs(gated - new[f"{prefix}_gated_b{budget}"])
            )
            result[f"{prefix}_stable_b{budget}_gain_over_v33_gated"] = gated - stable
    for name in ("future_persistence", "future_dynamics"):
        reproduction_differences.append(
            abs(old[name] - new[f"future_gated_{name[7:]}"])
        )
    result["v33_reproduction_max_abs_difference"] = max(reproduction_differences)
    return result
