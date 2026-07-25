"""Summarize the fixed 256-sample RGB/Object-JEPA representation gate."""
from __future__ import annotations

import argparse
import json
import os


SPLITS = ("train", "heldseed", "heldtask")
MIN_SLOT_RELATIVE_IMPROVEMENT = 0.01


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "ok":
        raise ValueError(f"evaluation is not complete: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_root", required=True)
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rgb_root = os.path.abspath(args.rgb_root)
    baseline_root = os.path.abspath(args.baseline_root)
    records = {}
    failures = []
    for split in SPLITS:
        rgb_path = os.path.join(
            rgb_root,
            f"representation_eval_final_{split}.json",
        )
        baseline_path = os.path.join(
            baseline_root,
            f"representation_eval_0000500_{split}_diagnostic.json",
        )
        rgb = load_report(rgb_path)
        baseline = load_report(baseline_path)
        if rgb["path_digest"] != baseline["path_digest"]:
            raise ValueError(f"{split} RGB and baseline pair subsets differ")
        rgb_metrics = rgb["metrics"]
        baseline_metrics = baseline["metrics"]
        feature_degradation = (
            rgb_metrics["feature_mse"] - baseline_metrics["feature_mse"]
        ) / max(baseline_metrics["feature_mse"], 1e-8)
        rgb_paired = rgb_metrics["rgb_vs_global_color_paired"]
        feature_slot_paired = rgb_metrics["feature_slot_conditioning_paired"]
        rgb_slot_paired = rgb_metrics["rgb_slot_conditioning_paired"]
        covariance_floor = rgb["config"]["covariance_floor"]
        checks = {
            "rgb_positive_2se": bool(rgb_paired["positive_2se_margin"]),
            "feature_depends_on_slots": bool(
                feature_slot_paired["positive_2se_margin"]
                and feature_slot_paired["relative"]
                >= MIN_SLOT_RELATIVE_IMPROVEMENT
            ),
            "rgb_depends_on_slots": bool(
                rgb_slot_paired["positive_2se_margin"]
                and rgb_slot_paired["relative"]
                >= MIN_SLOT_RELATIVE_IMPROVEMENT
            ),
            "feature_degradation_within_5pct": feature_degradation <= 0.05,
            "feature_coverage_complete": (
                rgb_metrics["feature_coverage_fraction"] >= 0.999
            ),
            "covariance_floor_respected": (
                rgb_metrics["readout_covariance_min_eigenvalue"]
                >= covariance_floor * 0.999
            ),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{split}:{name}")
        records[split] = {
            "samples": rgb["samples"],
            "path_digest": rgb["path_digest"],
            "rgb_distance": rgb_metrics["rgb_distance"],
            "global_color_rgb_distance": rgb_metrics[
                "global_color_rgb_distance"
            ],
            "rgb_relative_improvement": rgb_paired["relative"],
            "rgb_paired_standard_error": rgb_paired["standard_error"],
            "rgb_win_fraction": rgb_paired["win_fraction"],
            "feature_slot_relative_improvement": feature_slot_paired["relative"],
            "rgb_slot_relative_improvement": rgb_slot_paired["relative"],
            "feature_mse": rgb_metrics["feature_mse"],
            "baseline_feature_mse": baseline_metrics["feature_mse"],
            "feature_degradation": feature_degradation,
            "feature_coverage_fraction": rgb_metrics[
                "feature_coverage_fraction"
            ],
            "readout_covariance_min_eigenvalue": rgb_metrics[
                "readout_covariance_min_eigenvalue"
            ],
            "checks": checks,
        }
    report = {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "rgb_root": rgb_root,
        "baseline_root": baseline_root,
        "min_slot_relative_improvement": MIN_SLOT_RELATIVE_IMPROVEMENT,
        "failures": failures,
        "splits": records,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
