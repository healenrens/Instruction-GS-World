"""Decision contract for stable object-residual transport audits."""

from __future__ import annotations

import torch

from .object_centered_residual_report import cluster_gap_recovery, cluster_ratio


def _paired_parity(
    ratios: dict,
    name: str,
    maximum_ratio: float,
) -> bool:
    return (
        ratios[f"current_{name}"]["ci95_high"] <= maximum_ratio
        and ratios[f"future_{name}"]["ci95_high"] <= maximum_ratio
    )


def summarize_stable_transport_audit(
    rows: dict[str, torch.Tensor],
    budgets: tuple[int, ...],
    compact_budget: int,
    minimum_compact_recovery: float,
    maximum_token_ratio: float,
    maximum_coefficient_ratio: float,
    maximum_sensitivity_ratio: float,
    maximum_dynamics_ratio: float,
    maximum_condition_number: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict[str, bool], str]:
    maximum = max(budgets)
    current_clusters = rows["current_sequence_index"].long()
    future_clusters = rows["future_sequence_index"].long()
    generator = torch.Generator().manual_seed(seed)
    ratios = {}
    for budget in budgets:
        ratios[f"current_stable_gap_recovery_b{budget}"] = cluster_gap_recovery(
            rows["current_geometric_root"],
            rows[f"current_stable_b{budget}"],
            rows["current_token"],
            current_clusters,
            bootstrap_samples,
            generator,
        )
        ratios[f"future_stable_gap_recovery_b{budget}"] = cluster_gap_recovery(
            rows["future_geometric_root"],
            rows[f"future_stable_b{budget}"],
            rows["future_token"],
            future_clusters,
            bootstrap_samples,
            generator,
        )
    ratios["current_stable_max_to_token"] = cluster_ratio(
        rows[f"current_stable_b{maximum}"],
        rows["current_token"].clamp_min(1e-8),
        current_clusters,
        bootstrap_samples,
        generator,
    )
    ratios["future_stable_max_to_token"] = cluster_ratio(
        rows[f"future_stable_b{maximum}"],
        rows["future_token"].clamp_min(1e-8),
        future_clusters,
        bootstrap_samples,
        generator,
    )
    for prefix, clusters in (
        ("current", current_clusters),
        ("future", future_clusters),
    ):
        ratios[f"{prefix}_stable_coefficient_to_oracle"] = cluster_ratio(
            rows[f"{prefix}_stable_coefficient_rms_b{maximum}"],
            rows[f"{prefix}_oracle_coefficient_rms_b{maximum}"].clamp_min(1e-8),
            clusters,
            bootstrap_samples,
            generator,
        )
    ratios["center_sensitivity_stable_to_gated"] = cluster_ratio(
        rows["current_stable_center_sensitivity"],
        rows["current_gated_center_sensitivity"].clamp_min(1e-8),
        current_clusters,
        bootstrap_samples,
        generator,
    )
    ratios["scale_sensitivity_stable_to_gated"] = cluster_ratio(
        rows["current_stable_scale_sensitivity"],
        rows["current_gated_scale_sensitivity"].clamp_min(1e-8),
        current_clusters,
        bootstrap_samples,
        generator,
    )
    ratios["stable_dynamics_to_persistence"] = cluster_ratio(
        rows["future_stable_dynamics"],
        rows["future_stable_persistence"].clamp_min(1e-8),
        future_clusters,
        bootstrap_samples,
        generator,
    )
    ratios["gated_dynamics_to_persistence"] = cluster_ratio(
        rows["future_gated_dynamics"],
        rows["future_gated_persistence"].clamp_min(1e-8),
        future_clusters,
        bootstrap_samples,
        generator,
    )
    means = {
        name: float(value.mean())
        for name, value in rows.items()
        if not name.endswith("sequence_index")
    }
    monotonic = all(
        means[f"{prefix}_stable_b{right}"] <= means[f"{prefix}_stable_b{left}"] + 1e-5
        for prefix in ("current", "future")
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    nonfinite = sorted(
        name for name, value in rows.items() if not bool(torch.isfinite(value).all())
    )
    condition_values = torch.cat(
        [value for name, value in rows.items() if "stable_condition_b" in name]
    )
    maximum_observed_condition = float(condition_values.max())
    compact = (
        ratios[f"current_stable_gap_recovery_b{compact_budget}"]["ci95_low"]
        >= minimum_compact_recovery
        and ratios[f"future_stable_gap_recovery_b{compact_budget}"]["ci95_low"]
        >= minimum_compact_recovery
    )
    parity = _paired_parity(ratios, "stable_max_to_token", maximum_token_ratio)
    coefficients = _paired_parity(
        ratios, "stable_coefficient_to_oracle", maximum_coefficient_ratio
    )
    sensitivity = (
        ratios["center_sensitivity_stable_to_gated"]["ci95_high"]
        <= maximum_sensitivity_ratio
        and ratios["scale_sensitivity_stable_to_gated"]["ci95_high"]
        <= maximum_sensitivity_ratio
    )
    dynamics_stable = (
        ratios["stable_dynamics_to_persistence"]["ci95_high"] <= maximum_dynamics_ratio
    )
    checks = {
        "nonempty_held_set": current_clusters.numel() > 0,
        "multiple_episode_clusters": torch.unique(current_clusters).numel() >= 2,
        "all_metrics_finite": not nonfinite,
        "ridge_condition_is_bounded": maximum_observed_condition
        <= maximum_condition_number,
        "stable_budget_curve_is_monotonic": monotonic,
        f"stable_compact_b{compact_budget}_recovery_lower_ci": compact,
        "stable_max_budget_within_token_parity": parity,
        "stable_coefficient_amplification_is_bounded": coefficients,
        "stable_transport_sensitivity_improves_gated": sensitivity,
        "stable_dynamics_does_not_explode": dynamics_stable,
    }
    execution = (
        "nonempty_held_set",
        "multiple_episode_clusters",
        "all_metrics_finite",
        "ridge_condition_is_bounded",
        "stable_budget_curve_is_monotonic",
    )
    if not all(checks[name] for name in execution):
        decision = "inconclusive"
    elif not compact or not parity:
        decision = "reject_ungated_local_residual"
    elif not coefficients:
        decision = "residual_coefficients_remain_unstable"
    elif not sensitivity:
        decision = "repair_residual_geometry_transport"
    elif not dynamics_stable:
        decision = "promote_stable_representation_dynamics_pending"
    else:
        decision = "promote_stable_object_residual_transport"
    statistics = {
        "means": means,
        "ratios": ratios,
        "numerics": {
            "maximum_observed_condition_number": maximum_observed_condition,
            "nonfinite_metrics": nonfinite,
        },
    }
    return statistics, checks, decision
