"""Decision contract for partition-of-unity object-field audits."""

from __future__ import annotations

import torch

from .object_centered_residual_report import cluster_gap_recovery, cluster_ratio


def _paired_upper_bound(
    ratios: dict,
    metric: str,
    maximum: float,
) -> bool:
    return all(
        ratios[f"{prefix}_{metric}"]["ci95_high"] <= maximum
        for prefix in ("current", "future")
    )


def summarize_partitioned_field_audit(
    rows: dict[str, torch.Tensor],
    budgets: tuple[int, ...],
    compact_budget: int,
    minimum_compact_recovery: float,
    maximum_token_ratio: float,
    maximum_support_js_ratio: float,
    maximum_coefficient_rms: float,
    maximum_condition_number: float,
    maximum_transport_ratio: float,
    maximum_dynamics_ratio: float,
    maximum_unrepresented_object_mass: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict[str, bool], str]:
    maximum_budget = max(budgets)
    current_clusters = rows["current_sequence_index"].long()
    future_clusters = rows["future_sequence_index"].long()
    generator = torch.Generator().manual_seed(seed)
    ratios = {}
    for budget in budgets:
        ratios[f"current_partition_gap_recovery_b{budget}"] = cluster_gap_recovery(
            rows["current_geometric_root"],
            rows[f"current_partition_b{budget}"],
            rows["current_token"],
            current_clusters,
            bootstrap_samples,
            generator,
        )
        ratios[f"future_partition_gap_recovery_b{budget}"] = cluster_gap_recovery(
            rows["future_geometric_root"],
            rows[f"future_partition_b{budget}"],
            rows["future_token"],
            future_clusters,
            bootstrap_samples,
            generator,
        )
    for prefix, clusters in (
        ("current", current_clusters),
        ("future", future_clusters),
    ):
        ratios[f"{prefix}_partition_max_to_token"] = cluster_ratio(
            rows[f"{prefix}_partition_b{maximum_budget}"],
            rows[f"{prefix}_token"].clamp_min(1e-8),
            clusters,
            bootstrap_samples,
            generator,
        )
        ratios[f"{prefix}_support_js_to_geometric"] = cluster_ratio(
            rows[f"{prefix}_partition_support_js_b{maximum_budget}"],
            rows[f"{prefix}_geometric_support_js"].clamp_min(1e-8),
            clusters,
            bootstrap_samples,
            generator,
        )
    for name in (
        "feature_only",
        "center_only",
        "scale_only",
        "presence_only",
        "predicted_geometry",
        "predicted_state",
        "full",
        "target_feature_only",
        "target_geometry",
        "target_presence_only",
        "target_state",
        "target_state_predicted_feature",
        "predicted_state_target_feature",
        "target_state_target_feature",
    ):
        ratios[f"{name}_to_persistence"] = cluster_ratio(
            rows[f"future_{name}"],
            rows["future_persistence"].clamp_min(1e-8),
            future_clusters,
            bootstrap_samples,
            generator,
        )
    for name in (
        "predicted_geometry",
        "predicted_state",
        "target_geometry",
        "target_state",
    ):
        ratios[f"support_{name}_to_persistence"] = cluster_ratio(
            rows[f"future_support_{name}_js"],
            rows["future_support_persistence_js"].clamp_min(1e-8),
            future_clusters,
            bootstrap_samples,
            generator,
        )

    means = {
        name: float(value.mean())
        for name, value in rows.items()
        if not name.endswith("sequence_index")
    }
    nonfinite = sorted(
        name for name, value in rows.items() if not bool(torch.isfinite(value).all())
    )
    conditions = torch.cat(
        [value for name, value in rows.items() if "partition_condition_b" in name]
    )
    coefficients = torch.cat(
        [value for name, value in rows.items() if "partition_coefficient_rms_b" in name]
    )
    gate_sum_errors = torch.cat(
        [value for name, value in rows.items() if "partition_gate_sum_error_b" in name]
    )
    gate_minimums = torch.cat(
        [value for name, value in rows.items() if "partition_gate_minimum_b" in name]
    )
    monotonic = all(
        means[f"{prefix}_partition_b{right}"]
        <= means[f"{prefix}_partition_b{left}"] + 1e-5
        for prefix in ("current", "future")
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    compact = all(
        ratios[f"{prefix}_partition_gap_recovery_b{compact_budget}"]["ci95_low"]
        >= minimum_compact_recovery
        for prefix in ("current", "future")
    )
    parity = _paired_upper_bound(ratios, "partition_max_to_token", maximum_token_ratio)
    support = _paired_upper_bound(
        ratios, "support_js_to_geometric", maximum_support_js_ratio
    )
    numerics = (
        float(conditions.max()) <= maximum_condition_number
        and float(coefficients.max()) <= maximum_coefficient_rms
    )
    normalized = (
        float(gate_sum_errors.max()) <= 1e-6 and float(gate_minimums.min()) >= 0.0
    )
    target_support_transport = (
        ratios["support_target_state_to_persistence"]["ci95_high"]
        <= maximum_transport_ratio
    )
    oracle_transport = target_support_transport and (
        ratios["target_state_target_feature_to_persistence"]["ci95_high"]
        <= maximum_transport_ratio
    )
    predicted_support_transport = (
        ratios["support_predicted_state_to_persistence"]["ci95_high"]
        <= maximum_dynamics_ratio
    )
    predicted_state = predicted_support_transport and (
        ratios["predicted_state_target_feature_to_persistence"]["ci95_high"]
        <= maximum_dynamics_ratio
    )
    predicted_feature = (
        ratios["target_state_predicted_feature_to_persistence"]["ci95_high"]
        <= maximum_dynamics_ratio
    )
    full_dynamics = ratios["full_to_persistence"]["ci95_high"] <= maximum_dynamics_ratio
    birth_coverage = (
        means["future_unrepresented_object_mass"] <= maximum_unrepresented_object_mass
    )
    checks = {
        "nonempty_held_set": current_clusters.numel() > 0,
        "multiple_episode_clusters": torch.unique(current_clusters).numel() >= 2,
        "all_metrics_finite": not nonfinite,
        "partition_budget_curve_is_monotonic": monotonic,
        "partition_support_improves_single_gaussian": support,
        f"partition_compact_b{compact_budget}_recovery_lower_ci": compact,
        "partition_max_budget_within_token_parity": parity,
        "partition_expert_numerics_are_bounded": numerics,
        "partition_weights_are_normalized_nonnegative": normalized,
        "target_state_support_transport_improves_persistence": target_support_transport,
        "target_state_transport_improves_persistence": oracle_transport,
        "predicted_support_transport_does_not_explode": predicted_support_transport,
        "predicted_object_state_does_not_explode": predicted_state,
        "predicted_feature_does_not_explode": predicted_feature,
        "full_dynamics_does_not_explode": full_dynamics,
        "future_object_mass_is_represented_by_current_memory": birth_coverage,
    }
    execution = (
        "nonempty_held_set",
        "multiple_episode_clusters",
        "all_metrics_finite",
        "partition_budget_curve_is_monotonic",
        "partition_weights_are_normalized_nonnegative",
    )
    if not all(checks[name] for name in execution):
        decision = "inconclusive"
    elif not birth_coverage:
        decision = "reject_unrepresented_object_birth_contract"
    elif not support or not compact or not parity:
        decision = "reject_partition_of_unity_capacity"
    elif not numerics:
        decision = "repair_object_expert_numerics"
    elif not oracle_transport:
        decision = "reject_rigid_object_feature_transport"
    elif not full_dynamics:
        if not predicted_state and predicted_feature:
            decision = "repair_dynamics_geometry_or_presence"
        elif predicted_state and not predicted_feature:
            decision = "repair_dynamics_feature"
        else:
            decision = "repair_joint_dynamics_state"
    else:
        decision = "promote_partitioned_object_field"
    statistics = {
        "means": means,
        "ratios": ratios,
        "numerics": {
            "maximum_observed_condition_number": float(conditions.max()),
            "maximum_observed_coefficient_rms": float(coefficients.max()),
            "maximum_partition_sum_error": float(gate_sum_errors.max()),
            "minimum_partition_weight": float(gate_minimums.min()),
            "nonfinite_metrics": nonfinite,
        },
    }
    return statistics, checks, decision
