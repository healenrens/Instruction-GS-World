"""Fail-closed gate for oracle matched-effect cross-episode transfer."""
from __future__ import annotations

import argparse
import json
import os


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def paired_signal(comparison: dict, minimum_relative: float) -> bool:
    return (
        bool(comparison.get("positive_ci95_lower", False))
        and float(comparison["relative_improvement"]) >= minimum_relative
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldseed", required=True)
    parser.add_argument("--heldtask", required=True)
    parser.add_argument("--minimum_distance_reduction", type=float, default=0.25)
    parser.add_argument("--minimum_transfer_improvement", type=float, default=0.005)
    parser.add_argument("--minimum_heldseed_samples", type=int, default=1024)
    parser.add_argument("--minimum_heldtask_samples", type=int, default=450)
    parser.add_argument("--minimum_clusters", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_distance_reduction < 1.0:
        raise ValueError("distance reduction must be in [0,1)")

    reports = {
        "heldseed": load_report(os.path.abspath(args.heldseed)),
        "heldtask": load_report(os.path.abspath(args.heldtask)),
    }
    checks: dict[str, bool] = {}
    summaries = {}
    minimum_samples = {
        "heldseed": args.minimum_heldseed_samples,
        "heldtask": args.minimum_heldtask_samples,
    }
    for split, report in reports.items():
        evaluation = report["evaluation"]
        diagnostics = evaluation["transfer_diagnostics"]
        comparison = evaluation["comparison"]
        changed = evaluation["change_weighted_comparison"]
        checks[f"{split}_report_identity"] = (
            report["status"] == "ok"
            and report["split"] == split
            and int(report["action_dim"]) == 14
            and int(report["canonical_action_dim"]) == 6
            and int(report["action_residual_dim"]) == 8
        )
        expected_samples = min(
            int(report["requested_max_items"]),
            int(report["available_samples"]),
        )
        checks[f"{split}_sample_coverage"] = (
            int(evaluation["samples"]) == expected_samples
            and int(evaluation["samples"]) >= minimum_samples[split]
            and int(evaluation["clusters"]) >= args.minimum_clusters
        )
        checks[f"{split}_cross_episode_donors"] = (
            int(diagnostics["matched_same_episode_collisions"]) == 0
            and int(diagnostics["shuffled_same_episode_collisions"]) == 0
        )
        checks[f"{split}_effect_matching_effective"] = (
            float(diagnostics["matched_distance_ratio"])
            <= 1.0 - args.minimum_distance_reduction
        )
        checks[f"{split}_matched_action_transfers"] = all(
            paired_signal(
                comparison[metric]["matched_effect_over_shuffled"],
                args.minimum_transfer_improvement,
            )
            for metric in ("feature_mse", "latent_mse", "rgb_distance")
        ) and all(
            paired_signal(
                changed[metric]["matched_effect_over_shuffled"],
                args.minimum_transfer_improvement,
            )
            for metric in (
                "change_weighted_feature_mse",
                "change_weighted_rgb_charbonnier",
            )
        )
        checks[f"{split}_matched_residual_transfers"] = all(
            paired_signal(
                comparison[metric]["matched_residual_over_random"],
                args.minimum_transfer_improvement,
            )
            for metric in ("feature_mse", "latent_mse", "rgb_distance")
        ) and all(
            paired_signal(
                changed[metric]["matched_residual_over_random"],
                args.minimum_transfer_improvement,
            )
            for metric in (
                "change_weighted_feature_mse",
                "change_weighted_rgb_charbonnier",
            )
        )
        summaries[split] = {
            "checkpoint": os.path.abspath(report["checkpoint"]),
            "checkpoint_global_step": int(report["checkpoint_global_step"]),
            "checkpoint_phase": report["checkpoint_phase"],
            "data": os.path.abspath(report["data"]),
            "data_sha256": str(report["data_sha256"]),
            "samples": int(evaluation["samples"]),
            "clusters": int(evaluation["clusters"]),
            "diagnostics": diagnostics,
        }

    checkpoint_paths = {summary["checkpoint"] for summary in summaries.values()}
    checkpoint_steps = {
        summary["checkpoint_global_step"] for summary in summaries.values()
    }
    checks["single_checkpoint"] = (
        len(checkpoint_paths) == 1
        and len(checkpoint_steps) == 1
        and {summary["checkpoint_phase"] for summary in summaries.values()}
        == {"joint"}
        and all(os.path.isfile(path) for path in checkpoint_paths)
        and os.path.basename(next(iter(checkpoint_paths)))
        == f"joint_{next(iter(checkpoint_steps)):07d}.pt"
    )
    checks["single_data_root"] = (
        len({summary["data"] for summary in summaries.values()}) == 1
        and len({summary["data_sha256"] for summary in summaries.values()}) == 1
        and next(iter(summaries.values()))["data_sha256"] != ""
    )
    passed = all(checks.values())
    report = {
        "status": "pass" if passed else "fail",
        "scope": "oracle_matched_effect_cross_episode_transfer",
        "deployable_prediction_proven": False,
        "thresholds": {
            "minimum_distance_reduction": args.minimum_distance_reduction,
            "minimum_transfer_improvement": args.minimum_transfer_improvement,
            "minimum_heldseed_samples": args.minimum_heldseed_samples,
            "minimum_heldtask_samples": args.minimum_heldtask_samples,
            "minimum_clusters": args.minimum_clusters,
        },
        "checks": checks,
        "summary": summaries,
        "limitations": [
            "Future-conditioned canonical effects select the nearest donor.",
            "This tests representation transfer, not action inference at deployment.",
        ],
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise AssertionError(f"action transfer gate failed: {failed}")


if __name__ == "__main__":
    main()
