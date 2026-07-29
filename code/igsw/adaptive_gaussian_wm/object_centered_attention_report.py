"""Promotion decisions for anisotropic attention-carrier capacity audits."""

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
    root: torch.Tensor,
    prediction: torch.Tensor,
    token: torch.Tensor,
    clusters: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> dict[str, float]:
    gap = (root - token).clamp_min(0.0)
    unrecovered = (prediction - token).clamp_min(0.0)
    unique = torch.unique(clusters, sorted=True)
    gap_sum = torch.stack([gap[clusters == value].sum() for value in unique])
    unrecovered_sum = torch.stack(
        [unrecovered[clusters == value].sum() for value in unique]
    )

    def recovery(total_gap: torch.Tensor, total_unrecovered: torch.Tensor):
        value = 1.0 - total_unrecovered / total_gap.clamp_min(1e-8)
        no_gap = (total_unrecovered <= 1e-8).to(total_gap.dtype)
        return torch.where(total_gap > 1e-8, value.clamp(0.0, 1.0), no_gap)

    estimate = recovery(gap_sum.sum(), unrecovered_sum.sum())
    draws = torch.randint(len(unique), (samples, len(unique)), generator=generator)
    values = recovery(gap_sum[draws].sum(dim=1), unrecovered_sum[draws].sum(dim=1))
    bounds = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return {
        "estimate": float(estimate),
        "ci95_low": float(bounds[0]),
        "ci95_high": float(bounds[1]),
    }


def summarize_attention_audit(
    rows: dict[str, torch.Tensor],
    budgets: tuple[int, ...],
    minimum_compact_recovery: float,
    maximum_token_ratio: float,
    maximum_scene_fraction: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict[str, bool], str]:
    minimum = min(budgets)
    maximum = max(budgets)
    current_clusters = rows["current_sequence_index"].long()
    future_clusters = rows["future_sequence_index"].long()
    generator = torch.Generator().manual_seed(seed)
    ratios = {}
    for budget in budgets:
        ratios[f"current_gap_recovery_b{budget}"] = _cluster_gap_recovery(
            rows["current_root"],
            rows[f"current_attention_b{budget}"],
            rows["current_token"],
            current_clusters,
            bootstrap_samples,
            generator,
        )
        ratios[f"future_gap_recovery_b{budget}"] = _cluster_gap_recovery(
            rows["future_root"],
            rows[f"future_oracle_b{budget}"],
            rows["future_token"],
            future_clusters,
            bootstrap_samples,
            generator,
        )
    ratios["current_max_to_token"] = _cluster_ratio(
        rows[f"current_attention_b{maximum}"],
        rows["current_token"].clamp_min(1e-8),
        current_clusters,
        bootstrap_samples,
        generator,
    )
    ratios["future_max_to_token"] = _cluster_ratio(
        rows[f"future_oracle_b{maximum}"],
        rows["future_token"].clamp_min(1e-8),
        future_clusters,
        bootstrap_samples,
        generator,
    )
    headroom = (
        rows["future_persistence"] - rows[f"future_oracle_b{maximum}"]
    ).clamp_min(1e-8)
    ratios["dynamics_recovered_oracle_headroom"] = _cluster_ratio(
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
    current_monotonic = all(
        means[f"current_attention_b{right}"]
        <= means[f"current_attention_b{left}"] + 1e-6
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    future_monotonic = all(
        means[f"future_oracle_b{right}"] <= means[f"future_oracle_b{left}"] + 1e-6
        for left, right in zip(budgets[:-1], budgets[1:], strict=True)
    )
    nonfinite = [
        name for name, value in rows.items() if not bool(torch.isfinite(value).all())
    ]
    scene_fraction = means["current_scene_selected"] / max(
        means["current_selected"], 1.0
    )
    compact = (
        ratios[f"current_gap_recovery_b{minimum}"]["ci95_low"]
        >= minimum_compact_recovery
        and ratios[f"future_gap_recovery_b{minimum}"]["ci95_low"]
        >= minimum_compact_recovery
    )
    maximum_parity = (
        ratios["current_max_to_token"]["ci95_high"] <= maximum_token_ratio
        and ratios["future_max_to_token"]["ci95_high"] <= maximum_token_ratio
    )
    checks = {
        "nonempty_held_set": current_clusters.numel() > 0,
        "multiple_episode_clusters": torch.unique(current_clusters).numel() >= 2,
        "all_metrics_finite": not nonfinite,
        "current_budget_curve_is_monotonic": current_monotonic,
        "future_budget_curve_is_monotonic": future_monotonic,
        "adaptive_allocator_selected_local_carriers": means["current_selected"] > 0.0,
        "scene_fraction_respects_contract": scene_fraction
        <= maximum_scene_fraction + 0.02,
        f"compact_b{minimum}_recovery_lower_ci": compact,
        "max_budget_within_token_parity": maximum_parity,
    }
    execution_names = (
        "nonempty_held_set",
        "multiple_episode_clusters",
        "all_metrics_finite",
        "adaptive_allocator_selected_local_carriers",
        "scene_fraction_respects_contract",
    )
    if not all(checks[name] for name in execution_names):
        decision = "inconclusive"
    elif compact and maximum_parity and current_monotonic and future_monotonic:
        decision = "promote_compact_attention_carriers"
    elif maximum_parity and current_monotonic and future_monotonic:
        decision = "capacity_only_requires_higher_density"
    else:
        decision = "reject_attention_carrier_parameterization"
    statistics = {
        "means": means,
        "ratios": ratios,
        "allocation": {
            "current_scene_fraction": scene_fraction,
            "current_object_count_std": means["current_object_allocation_std"],
        },
    }
    return statistics, checks, decision
