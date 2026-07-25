"""Aggregate multi-seed architecture evidence into explicit mechanism gates."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from statistics import mean, stdev

VARIANTS = (
    "full",
    "no_object",
    "independent_slots",
    "fixed_density",
    "independent_prior",
    "all_degraded",
)

def metric(report: dict, path: str) -> float:
    value = report["metrics"]
    for name in path.split("."):
        value = value[name]
    return float(value)

def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]

def summary(values: list[float]) -> dict[str, float | int]:
    values = finite(values)
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }

def load_reports(input_dir: str) -> dict[str, list[dict]]:
    grouped = {variant: [] for variant in VARIANTS}
    for path in sorted(glob.glob(os.path.join(input_dir, "*_seed*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        variant = report.get("variant")
        if variant in grouped and report.get("status") == "ok":
            grouped[variant].append(report)
    for reports in grouped.values():
        reports.sort(key=lambda item: int(item["seed"]))
    return grouped

def aggregate(reports: list[dict]) -> dict:
    paths = (
        "feature_mse",
        "latent_mse",
        "center_mse",
        "current_copy_center_mse",
        "object_scores.f1",
        "temporal_identity_shuffle_latent_mse",
        "effective_token_mean",
        "effective_token_std",
        "effective_token_complexity_pearson",
        "low_complexity_feature_mse",
        "high_complexity_feature_mse",
        "density_reconstruction_mse",
        "swapped_density_reconstruction_mse",
        "mode.target_mode_separation",
        "mode.posterior_action_mode_separation",
        "mode.posterior_oracle_mse",
        "mode.posterior_oracle_mode_recall",
        "mode.posterior_oracle_center_mse",
        "mode.posterior_oracle_center_mode_recall",
        "mode.deterministic_center_extrapolation_mse",
        "mode.deterministic_center_copy_mse",
        "mode.deterministic_center_extrapolation_improvement",
        "mode.mode_recall_at_n",
        "mode.sample_precision_at_n",
        "mode.best_of_1_mse",
        "mode.best_of_n_mse",
        "mode.deterministic_prediction_mse",
        "mode.ambiguous_prior_diversity",
        "mode.deterministic_prior_diversity",
        "mode.prior_context_mode_max_difference",
        "mode.action_target_mode_separation",
        "mode.action_mode_recall_at_n",
        "mode.action_sample_precision_at_n",
        "mode.action_best_of_n_mse",
        "mode.action_deterministic_prediction_mse",
        "mode.center_mode_recall_at_n",
        "mode.center_sample_precision_at_n",
        "mode.center_best_of_n_mse",
        "mode.center_deterministic_prediction_mse",
    )
    return {
        "seeds": [int(report["seed"]) for report in reports],
        "trainable_parameters": sorted(
            {int(report["trainable_parameters"]) for report in reports}
        ),
        "metrics": {
            path: summary([metric(report, path) for report in reports])
            for path in paths
        },
    }
def paired_wins(
    left: list[dict],
    right: list[dict],
    path: str,
    lower_is_better: bool,
) -> dict[str, int]:
    right_by_seed = {int(report["seed"]): report for report in right}
    wins = 0
    comparisons = 0
    for report in left:
        seed = int(report["seed"])
        if seed not in right_by_seed:
            continue
        left_value = metric(report, path)
        right_value = metric(right_by_seed[seed], path)
        if not (math.isfinite(left_value) and math.isfinite(right_value)):
            continue
        comparisons += 1
        wins += int(
            left_value < right_value
            if lower_is_better
            else left_value > right_value
        )
    return {"wins": wins, "comparisons": comparisons}


def mean_metric(aggregates: dict, variant: str, path: str) -> float:
    return float(aggregates[variant]["metrics"][path]["mean"])


def relative_degradation(
    reports: list[dict],
    base_path: str,
    perturbed_path: str,
    minimum: float,
) -> dict:
    ratios = [
        (
            metric(report, perturbed_path) - metric(report, base_path)
        )
        / max(metric(report, base_path), 1e-8)
        for report in reports
    ]
    return {
        "minimum": minimum,
        "wins": sum(value >= minimum for value in ratios),
        "comparisons": len(ratios),
        "relative_change": summary(ratios),
    }
def build_gates(
    reports: dict[str, list[dict]],
    aggregates: dict[str, dict],
) -> dict:
    object_f1 = paired_wins(
        reports["full"],
        reports["no_object"],
        "object_scores.f1",
        False,
    )
    object_feature = paired_wins(
        reports["full"],
        reports["no_object"],
        "feature_mse",
        True,
    )
    object_latent = paired_wins(
        reports["full"],
        reports["no_object"],
        "latent_mse",
        True,
    )
    object_center = paired_wins(
        reports["full"],
        reports["no_object"],
        "center_mse",
        True,
    )
    identity_shuffle = relative_degradation(
        reports["full"],
        "latent_mse",
        "temporal_identity_shuffle_latent_mse",
        0.05,
    )
    full_count = mean_metric(aggregates, "full", "effective_token_mean")
    fixed_count = mean_metric(
        aggregates,
        "fixed_density",
        "effective_token_mean",
    )
    budget_difference = abs(full_count - fixed_count) / max(fixed_count, 1e-8)
    density_correlations = [
        metric(report, "effective_token_complexity_pearson")
        for report in reports["full"]
    ]
    density_positive = sum(value > 0.3 for value in density_correlations)
    density_high = paired_wins(
        reports["full"],
        reports["fixed_density"],
        "high_complexity_feature_mse",
        True,
    )
    full_low = mean_metric(
        aggregates,
        "full",
        "low_complexity_feature_mse",
    )
    fixed_low = mean_metric(
        aggregates,
        "fixed_density",
        "low_complexity_feature_mse",
    )
    low_cost = (full_low - fixed_low) / max(fixed_low, 1e-8)
    density_swapped = mean_metric(
        aggregates,
        "full",
        "swapped_density_reconstruction_mse",
    )
    density_base = mean_metric(
        aggregates,
        "full",
        "density_reconstruction_mse",
    )
    density_swap = relative_degradation(
        reports["full"],
        "density_reconstruction_mse",
        "swapped_density_reconstruction_mse",
        0.02,
    )
    context_difference = mean_metric(
        aggregates,
        "full",
        "mode.prior_context_mode_max_difference",
    )
    oracle_recall = mean_metric(aggregates, "full", "mode.posterior_oracle_mode_recall")
    oracle_center_recall = mean_metric(
        aggregates, "full", "mode.posterior_oracle_center_mode_recall"
    )
    observability = mean_metric(
        aggregates, "full", "mode.deterministic_center_extrapolation_improvement"
    )
    action_recall = mean_metric(
        aggregates,
        "full",
        "mode.action_mode_recall_at_n",
    )
    action_precision = mean_metric(
        aggregates,
        "full",
        "mode.action_sample_precision_at_n",
    )
    action_best = mean_metric(
        aggregates,
        "full",
        "mode.action_best_of_n_mse",
    )
    action_deterministic = mean_metric(
        aggregates,
        "full",
        "mode.action_deterministic_prediction_mse",
    )
    action_improvement = (
        action_deterministic - action_best
    ) / max(action_deterministic, 1e-8)
    center_recall = mean_metric(
        aggregates,
        "full",
        "mode.center_mode_recall_at_n",
    )
    center_precision = mean_metric(
        aggregates,
        "full",
        "mode.center_sample_precision_at_n",
    )
    center_best = mean_metric(
        aggregates,
        "full",
        "mode.center_best_of_n_mse",
    )
    center_deterministic = mean_metric(
        aggregates,
        "full",
        "mode.center_deterministic_prediction_mse",
    )
    center_improvement = (
        center_deterministic - center_best
    ) / max(center_deterministic, 1e-8)
    recall = mean_metric(aggregates, "full", "mode.mode_recall_at_n")
    precision = mean_metric(
        aggregates,
        "full",
        "mode.sample_precision_at_n",
    )
    best_n = mean_metric(aggregates, "full", "mode.best_of_n_mse")
    deterministic = mean_metric(
        aggregates,
        "full",
        "mode.deterministic_prediction_mse",
    )
    best_improvement = (deterministic - best_n) / max(deterministic, 1e-8)
    ambiguous_diversity = mean_metric(
        aggregates,
        "full",
        "mode.ambiguous_prior_diversity",
    )
    deterministic_diversity = mean_metric(
        aggregates,
        "full",
        "mode.deterministic_prior_diversity",
    )
    object_gate = (
        object_f1["comparisons"] >= 5
        and object_f1["wins"] >= 4
        and object_feature["wins"] >= 3
        and object_latent["wins"] >= 3
        and object_center["wins"] >= 3
        and identity_shuffle["wins"] >= 4
    )
    density_gate = (
        len(density_correlations) >= 5
        and budget_difference <= 0.05
        and density_positive >= 4
        and density_high["wins"] >= 3
        and low_cost <= 0.02
        and density_swap["wins"] >= 4
    )
    prior_gate = (
        len(reports["full"]) >= 5
        and context_difference <= 1e-6
        and observability > 0.0
        and oracle_recall >= 0.8
        and oracle_center_recall >= 0.8
        and action_recall >= 0.8
        and action_precision >= 0.7
        and action_improvement >= 0.15
        and center_recall >= 0.8
        and center_precision >= 0.7
        and center_improvement >= 0.15
        and recall >= 0.8
        and precision >= 0.7
        and best_improvement >= 0.15
        and ambiguous_diversity > deterministic_diversity
    )
    independent_slot_feature = paired_wins(
        reports["full"],
        reports["independent_slots"],
        "feature_mse",
        True,
    )
    independent_slot_latent = paired_wins(
        reports["full"],
        reports["independent_slots"],
        "latent_mse",
        True,
    )
    independent_prior_best = paired_wins(
        reports["full"],
        reports["independent_prior"],
        "mode.action_best_of_n_mse",
        True,
    )
    independent_prior_recall = paired_wins(
        reports["full"],
        reports["independent_prior"],
        "mode.action_mode_recall_at_n",
        False,
    )
    degraded_feature = paired_wins(
        reports["full"],
        reports["all_degraded"],
        "feature_mse",
        True,
    )
    degraded_latent = paired_wins(
        reports["full"],
        reports["all_degraded"],
        "latent_mse",
        True,
    )
    parameter_sets = {
        name: {
            int(report["trainable_parameters"])
            for report in variant_reports
        }
        for name, variant_reports in reports.items()
    }
    parameter_matched = (
        bool(parameter_sets["full"])
        and all(values == parameter_sets["full"] for values in parameter_sets.values())
    )
    coverage_gate = (
        all(len(reports[name]) >= 5 for name in VARIANTS)
        and parameter_matched
    )
    integrated_gate = (
        independent_slot_feature["wins"] >= 3
        and independent_slot_latent["wins"] >= 3
        and independent_prior_best["wins"] >= 3
        and independent_prior_recall["wins"] >= 3
        and degraded_feature["wins"] >= 3
        and degraded_latent["wins"] >= 3
    )
    overall = (
        object_gate
        and density_gate
        and prior_gate
        and coverage_gate
        and integrated_gate
    )
    return {
        "objectification": {
            "passed": object_gate,
            "f1_paired": object_f1,
            "feature_paired": object_feature,
            "latent_paired": object_latent,
            "center_paired": object_center,
            "identity_shuffle": identity_shuffle,
        },
        "adaptive_density": {
            "passed": density_gate,
            "budget_relative_difference": budget_difference,
            "correlations_above_0_3": density_positive,
            "high_complexity_paired": density_high,
            "low_complexity_relative_cost": low_cost,
            "swapped_reconstruction_ratio": density_swapped
            / max(density_base, 1e-8),
            "rank_reverse_activation": density_swap,
        },
        "multimodal_prior": {
            "passed": prior_gate,
            "prior_context_max_difference": context_difference,
            "deterministic_center_observability_improvement": observability,
            "posterior_oracle_mode_recall": oracle_recall,
            "posterior_oracle_center_mode_recall": oracle_center_recall,
            "mode_recall_at_16": recall,
            "sample_precision_at_16": precision,
            "best_of_16_improvement": best_improvement,
            "action_mode_recall_at_16": action_recall,
            "action_sample_precision_at_16": action_precision,
            "action_best_of_16_improvement": action_improvement,
            "center_mode_recall_at_16": center_recall,
            "center_sample_precision_at_16": center_precision,
            "center_best_of_16_improvement": center_improvement,
            "ambiguous_to_deterministic_diversity_ratio": ambiguous_diversity
            / max(deterministic_diversity, 1e-8),
        },
        "matrix_coverage": {
            "passed": coverage_gate,
            "parameter_matched": parameter_matched,
            "parameter_sets": {
                name: sorted(values)
                for name, values in parameter_sets.items()
            },
            "seed_counts": {
                name: len(reports[name])
                for name in VARIANTS
            },
        },
        "integrated_ablation": {
            "passed": integrated_gate,
            "independent_slot_feature": independent_slot_feature,
            "independent_slot_latent": independent_slot_latent,
            "independent_prior_best_of_n": independent_prior_best,
            "independent_prior_recall": independent_prior_recall,
            "all_degraded_feature": degraded_feature,
            "all_degraded_latent": degraded_latent,
        },
        "overall_architecture": {
            "passed": overall,
        },
        "large_scale_ready": overall,
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = load_reports(args.input_dir)
    aggregates = {
        variant: aggregate(variant_reports)
        for variant, variant_reports in reports.items()
    }
    gates = build_gates(reports, aggregates)
    result = {
        "status": "ok",
        "input_dir": os.path.abspath(args.input_dir),
        "aggregates": aggregates,
        "gates": gates,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
