"""Decision contract for orthogonalized adaptive residual-field audits."""

from __future__ import annotations

import torch

from .object_centered_residual_report import cluster_gap_recovery, cluster_ratio


def summarize_orthogonalized_residual_audit(
    rows: dict[str, torch.Tensor],
    budgets: tuple[int, ...],
    compact_budget: int,
    minimum_compact_recovery: float,
    maximum_token_ratio: float,
    maximum_coefficient_rms: float,
    maximum_condition_number: float,
    minimum_retained_energy: float,
    maximum_column_amplification: float,
    maximum_transport_ratio: float,
    maximum_dynamics_ratio: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict[str, bool], str]:
    maximum_budget = max(budgets)
    current_clusters = rows["current_sequence_index"].long()
    future_clusters = rows["future_sequence_index"].long()
    generator = torch.Generator().manual_seed(seed)
    ratios = {}
    for budget in budgets:
        for prefix, clusters in (
            ("current", current_clusters),
            ("future", future_clusters),
        ):
            ratios[f"{prefix}_orthogonal_gap_recovery_b{budget}"] = (
                cluster_gap_recovery(
                    rows[f"{prefix}_geometric_root"],
                    rows[f"{prefix}_orthogonal_b{budget}"],
                    rows[f"{prefix}_token"],
                    clusters,
                    bootstrap_samples,
                    generator,
                )
            )
    for prefix, clusters in (
        ("current", current_clusters),
        ("future", future_clusters),
    ):
        ratios[f"{prefix}_orthogonal_max_to_token"] = cluster_ratio(
            rows[f"{prefix}_orthogonal_b{maximum_budget}"],
            rows[f"{prefix}_token"].clamp_min(1e-8),
            clusters,
            bootstrap_samples,
            generator,
        )
    variants = (
        "feature_only",
        "target_feature_only",
        "predicted_root",
        "predicted_root_feature",
        "predicted_rigid",
        "predicted_rigid_feature",
        "target_root",
        "target_root_predicted_feature",
        "target_root_target_feature",
        "target_rigid",
        "target_rigid_predicted_feature",
        "target_rigid_target_feature",
    )
    for name in variants:
        ratios[f"{name}_to_persistence"] = cluster_ratio(
            rows[f"future_{name}"],
            rows["future_persistence"].clamp_min(1e-8),
            future_clusters,
            bootstrap_samples,
            generator,
        )
    ratios["target_rigid_to_root_with_target_feature"] = cluster_ratio(
        rows["future_target_rigid_target_feature"],
        rows["future_target_root_target_feature"].clamp_min(1e-8),
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
        [value for name, value in rows.items() if "orthogonal_condition_b" in name]
    )
    coefficients = torch.cat(
        [
            value
            for name, value in rows.items()
            if "orthogonal_coefficient_rms_b" in name
        ]
    )
    retained_energy = torch.cat(
        [value for name, value in rows.items() if "orthogonal_energy_b" in name]
    )
    amplification = torch.cat(
        [value for name, value in rows.items() if "column_amplification_max" in name]
    )
    monotonic = all(
        means[f"{prefix}_orthogonal_b{right}"]
        <= means[f"{prefix}_orthogonal_b{left}"] + 1e-5
        for prefix in ("current", "future")
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    compact = all(
        ratios[f"{prefix}_orthogonal_gap_recovery_b{compact_budget}"]["ci95_low"]
        >= minimum_compact_recovery
        for prefix in ("current", "future")
    )
    parity = all(
        ratios[f"{prefix}_orthogonal_max_to_token"]["ci95_high"] <= maximum_token_ratio
        for prefix in ("current", "future")
    )
    numerics = (
        float(conditions.max()) <= maximum_condition_number
        and float(coefficients.max()) <= maximum_coefficient_rms
        and float(retained_energy.min()) >= minimum_retained_energy
    )
    transport_columns = float(amplification.max()) <= maximum_column_amplification
    target_feature = (
        ratios["target_feature_only_to_persistence"]["ci95_high"]
        <= maximum_transport_ratio
    )
    target_root = (
        ratios["target_root_target_feature_to_persistence"]["ci95_high"]
        <= maximum_transport_ratio
    )
    target_rigid = (
        ratios["target_rigid_target_feature_to_persistence"]["ci95_high"]
        <= maximum_transport_ratio
    )
    predicted_root = (
        ratios["predicted_root_feature_to_persistence"]["ci95_high"]
        <= maximum_dynamics_ratio
    )
    predicted_rigid = (
        ratios["predicted_rigid_feature_to_persistence"]["ci95_high"]
        <= maximum_dynamics_ratio
    )
    checks = {
        "nonempty_held_set": current_clusters.numel() > 0,
        "multiple_episode_clusters": torch.unique(current_clusters).numel() >= 2,
        "all_metrics_finite": not nonfinite,
        "orthogonal_budget_curve_is_monotonic": monotonic,
        f"orthogonal_compact_b{compact_budget}_recovery_lower_ci": compact,
        "orthogonal_max_budget_within_token_parity": parity,
        "orthogonal_numerics_are_bounded": numerics,
        "transported_columns_are_bounded": transport_columns,
        "target_object_feature_improves_persistence": target_feature,
        "target_root_transport_improves_persistence": target_root,
        "target_rigid_transport_improves_persistence": target_rigid,
        "predicted_root_transport_does_not_explode": predicted_root,
        "predicted_rigid_transport_does_not_explode": predicted_rigid,
    }
    execution = (
        "nonempty_held_set",
        "multiple_episode_clusters",
        "all_metrics_finite",
        "orthogonal_budget_curve_is_monotonic",
    )
    if not all(checks[name] for name in execution):
        decision = "inconclusive"
    elif not compact or not parity:
        decision = "reject_adaptive_gated_residual_capacity"
    elif not numerics or not transport_columns:
        decision = "repair_orthogonalized_residual_numerics"
    elif not target_feature:
        decision = "promote_representation_require_spatial_feature_target"
    elif target_root and not target_rigid:
        decision = "promote_representation_disable_rigid_residual_transport"
    elif not target_root:
        decision = "promote_representation_require_spatial_residual_predictor"
    elif not predicted_root or not predicted_rigid:
        decision = "promote_representation_dynamics_pending"
    else:
        decision = "promote_orthogonalized_object_residual"
    statistics = {
        "means": means,
        "ratios": ratios,
        "numerics": {
            "maximum_observed_condition_number": float(conditions.max()),
            "maximum_observed_coefficient_rms": float(coefficients.max()),
            "minimum_retained_energy": float(retained_energy.min()),
            "maximum_transport_column_amplification": float(amplification.max()),
            "nonfinite_metrics": nonfinite,
        },
    }
    return statistics, checks, decision
