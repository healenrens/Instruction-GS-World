"""Cluster-bootstrap decisions for the object-centered carrier audit."""

from __future__ import annotations

import torch


def _cluster_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    clusters: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> dict[str, float]:
    unique = torch.unique(clusters, sorted=True)
    numerator_sum = torch.stack(
        [numerator[clusters == value].sum() for value in unique]
    )
    denominator_sum = torch.stack(
        [denominator[clusters == value].sum() for value in unique]
    )
    estimate = numerator_sum.sum() / denominator_sum.sum().clamp_min(1e-8)
    draws = torch.randint(len(unique), (samples, len(unique)), generator=generator)
    values = numerator_sum[draws].sum(dim=1) / denominator_sum[draws].sum(
        dim=1
    ).clamp_min(1e-8)
    bounds = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return {
        "estimate": float(estimate),
        "ci95_low": float(bounds[0]),
        "ci95_high": float(bounds[1]),
    }


def _cluster_gap_recovery(
    gap: torch.Tensor,
    unrecovered: torch.Tensor,
    clusters: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> dict[str, float]:
    unique = torch.unique(clusters, sorted=True)
    gap_sum = torch.stack([gap[clusters == value].sum() for value in unique])
    unrecovered_sum = torch.stack(
        [unrecovered[clusters == value].sum() for value in unique]
    )

    def recovery(total_gap: torch.Tensor, total_unrecovered: torch.Tensor):
        recovered = 1.0 - total_unrecovered / total_gap.clamp_min(1e-8)
        no_gap = (total_unrecovered <= 1e-8).to(total_gap.dtype)
        return torch.where(total_gap > 1e-8, recovered.clamp(0.0, 1.0), no_gap)

    estimate = recovery(gap_sum.sum(), unrecovered_sum.sum())
    draws = torch.randint(len(unique), (samples, len(unique)), generator=generator)
    values = recovery(gap_sum[draws].sum(dim=1), unrecovered_sum[draws].sum(dim=1))
    bounds = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return {
        "estimate": float(estimate),
        "ci95_low": float(bounds[0]),
        "ci95_high": float(bounds[1]),
    }


def _nonfinite_rows(rows: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        name for name, value in rows.items() if not bool(torch.isfinite(value).all())
    )


def summarize_carrier_audit(
    rows: dict[str, torch.Tensor],
    budgets: tuple[int, ...],
    minimum_gap_recovery: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict[str, bool], str]:
    maximum = max(budgets)
    current_clusters = rows["current_sequence_index"].long()
    future_clusters = rows["future_sequence_index"].long()
    generator = torch.Generator().manual_seed(seed)
    current_gap = (rows["current_root"] - rows["current_token"]).clamp_min(0.0)
    future_gap = (rows["future_root"] - rows["future_token"]).clamp_min(0.0)
    current_unrecovered = (
        rows[f"current_full_b{maximum}"] - rows["current_token"]
    ).clamp_min(0.0)
    future_unrecovered = (rows["future_oracle"] - rows["future_token"]).clamp_min(0.0)
    headroom = (rows["future_persistence"] - rows["future_oracle"]).clamp_min(1e-8)
    current_recovery = _cluster_gap_recovery(
        current_gap,
        current_unrecovered,
        current_clusters,
        bootstrap_samples,
        generator,
    )
    future_recovery = _cluster_gap_recovery(
        future_gap,
        future_unrecovered,
        future_clusters,
        bootstrap_samples,
        generator,
    )
    oracle_to_token = _cluster_ratio(
        rows["future_oracle"],
        rows["future_token"].clamp_min(1e-8),
        future_clusters,
        bootstrap_samples,
        generator,
    )
    scene_contribution = _cluster_ratio(
        rows[f"current_object_b{maximum}"] - rows[f"current_full_b{maximum}"],
        current_gap,
        current_clusters,
        bootstrap_samples,
        generator,
    )
    dynamics_headroom = _cluster_ratio(
        rows["future_persistence"] - rows["future_dynamics"],
        headroom,
        future_clusters,
        bootstrap_samples,
        generator,
    )
    means = {
        name: float(value.mean())
        for name, value in rows.items()
        if not name.endswith("sequence_index")
    }
    current_budget_monotonic = all(
        means[f"current_full_b{right}"] <= means[f"current_full_b{left}"] + 1e-6
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    future_budget_monotonic = all(
        means[f"future_oracle_b{right}"] <= means[f"future_oracle_b{left}"] + 1e-6
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    ratios = {
        "current_full_gap_recovery": current_recovery,
        "future_oracle_gap_recovery": future_recovery,
        "future_oracle_to_target_token": oracle_to_token,
        "scene_carrier_gap_recovery": scene_contribution,
        "dynamics_recovered_oracle_headroom": dynamics_headroom,
    }
    checks = {
        "nonempty_held_set": current_clusters.numel() > 0,
        "multiple_episode_clusters": torch.unique(current_clusters).numel() >= 2,
        "all_metrics_finite": not _nonfinite_rows(rows),
        "current_budget_curve_is_monotonic": current_budget_monotonic,
        "future_budget_curve_is_monotonic": future_budget_monotonic,
        "current_gap_recovery_lower_ci": (
            current_recovery["ci95_low"] >= minimum_gap_recovery
        ),
        "future_oracle_gap_recovery_lower_ci": (
            future_recovery["ci95_low"] >= minimum_gap_recovery
        ),
        "future_oracle_within_target_token_parity": (
            oracle_to_token["ci95_high"] <= 1.10
        ),
        "adaptive_allocator_selected_local_carriers": (
            means["current_full_selected"] > 0.0
        ),
    }
    quality = (
        "current_gap_recovery_lower_ci",
        "future_oracle_gap_recovery_lower_ci",
        "future_oracle_within_target_token_parity",
    )
    execution = tuple(name for name in checks if name not in quality)
    if not all(checks[name] for name in execution):
        decision = "inconclusive"
    elif all(checks[name] for name in quality):
        decision = "promote_object_centered_carriers"
    else:
        decision = "reject_current_carrier_parameterization"
    return {"means": means, "ratios": ratios}, checks, decision
