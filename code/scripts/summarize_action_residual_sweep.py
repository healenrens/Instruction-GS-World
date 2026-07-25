"""Enforce held-split posterior-vs-zero gates for residual bottlenecks."""
from __future__ import annotations

import argparse
import json
import os


RESIDUAL_DIMS = (0, 8, 16)
SPLITS = ("train", "heldseed", "heldtask")
METRICS = ("feature_mse", "latent_mse", "rgb_distance")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _run_path(template: str, residual_dim: int) -> str:
    return os.path.abspath(template.format(residual_dim=residual_dim))


def _split_summary(report: dict) -> dict:
    comparisons = {
        metric: report["metrics"]["posterior_comparison"][metric]["zero_action"]
        for metric in METRICS
    }
    return {
        "samples": report["metrics"]["samples"],
        "posterior": {
            metric: report["metrics"]["mean"][metric]["posterior"]
            for metric in METRICS
        },
        "zero_action": {
            metric: report["metrics"]["mean"][metric]["zero_action"]
            for metric in METRICS
        },
        "posterior_vs_zero": comparisons,
        "all_mean_improvements_positive": all(
            item["absolute_improvement"] > 0.0
            for item in comparisons.values()
        ),
        "all_margins_exceed_2se": all(
            item["positive_2se_margin"]
            for item in comparisons.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_template",
        required=True,
        help="absolute run path containing {residual_dim}",
    )
    parser.add_argument("--checkpoint_step", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "{residual_dim}" not in args.run_template:
        raise ValueError("run_template must contain {residual_dim}")

    runs = {}
    for residual_dim in RESIDUAL_DIMS:
        root = _run_path(args.run_template, residual_dim)
        split_reports = {}
        for split in SPLITS:
            path = os.path.join(
                root,
                f"posterior_gate_step{args.checkpoint_step}_{split}.json",
            )
            report = _load(path)
            if report.get("status") != "ok":
                raise ValueError(f"invalid report status: {path}")
            if report.get("action_residual_dim") != residual_dim:
                raise ValueError(f"residual dimension mismatch: {path}")
            if report.get("action_dim") != 6 + residual_dim:
                raise ValueError(f"action dimension mismatch: {path}")
            split_reports[split] = _split_summary(report)
        component_reports = {}
        for split in ("heldseed", "heldtask"):
            path = os.path.join(
                root,
                f"action_component_ablation_{split}.json",
            )
            report = _load(path)
            if report.get("status") != "ok":
                raise ValueError(f"invalid component report status: {path}")
            if report.get("action_residual_dim") != residual_dim:
                raise ValueError(f"component residual mismatch: {path}")
            means = report["metrics"]["mean"]
            canonical_primary = {
                metric: (
                    means[metric]["posterior_canonical_only"]
                    < means[metric]["zero_action"]
                    and means[metric]["posterior_canonical_only"]
                    <= means[metric]["posterior_residual_only"]
                )
                for metric in METRICS
            }
            component_reports[split] = {
                "samples": report["metrics"]["samples"],
                "mean": {
                    metric: {
                        variant: means[metric][variant]
                        for variant in (
                            "posterior",
                            "posterior_canonical_only",
                            "posterior_residual_only",
                            "zero_action",
                        )
                    }
                    for metric in METRICS
                },
                "canonical_primary": canonical_primary,
                "canonical_primary_all_metrics": all(
                    canonical_primary.values()
                ),
            }
        held = [split_reports[name] for name in ("heldseed", "heldtask")]
        canonical_held = [
            component_reports[name] for name in ("heldseed", "heldtask")
        ]
        posterior_gate = all(
            item["all_mean_improvements_positive"] for item in held
        )
        canonical_gate = all(
            item["canonical_primary_all_metrics"] for item in canonical_held
        )
        runs[str(residual_dim)] = {
            "root": root,
            "splits": split_reports,
            "components": component_reports,
            "posterior_held_gate": posterior_gate,
            "canonical_primary_held_gate": canonical_gate,
            "eligible_for_prior": posterior_gate and canonical_gate,
            "strict_held_gate": all(
                item["all_margins_exceed_2se"] for item in held
            ),
            "mean_held_relative_improvement": sum(
                item["posterior_vs_zero"][metric]["relative_improvement"]
                for item in held
                for metric in METRICS
            )
            / (2 * len(METRICS)),
        }
    eligible = [
        residual_dim
        for residual_dim in RESIDUAL_DIMS
        if runs[str(residual_dim)]["eligible_for_prior"]
    ]
    eligible.sort(
        key=lambda value: runs[str(value)]["mean_held_relative_improvement"],
        reverse=True,
    )
    summary = {
        "status": "ok",
        "gate_definition": (
            "posterior mean error must beat zero-action for feature, latent, "
            "and RGB on both heldseed and heldtask; canonical-only must also "
            "beat zero-action and be no worse than residual-only"
        ),
        "runs": runs,
        "eligible_for_prior": eligible,
        "selected_for_prior": eligible[0] if eligible else None,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
