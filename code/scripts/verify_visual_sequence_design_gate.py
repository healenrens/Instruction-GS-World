"""Fail-closed evidence gate for the posterior-conditioned world-model core."""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
from igsw.adaptive_gaussian_wm.sequence_evidence import (  # noqa: E402
    sequence_manifest_summary,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    MINIMUM_TASK_MEDIAN_RELATIVE, TASK_CONTRACT, task_evidence_passes,
    task_source_index_sha256)
def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
def paired_signal(
    comparison: dict,
    minimum_relative: float,
) -> bool:
    relative_key = (
        "relative_improvement"
        if "relative_improvement" in comparison
        else "relative"
    )
    significant = comparison.get(
        "positive_ci95_lower",
        comparison.get("positive_2se_margin", False),
    )
    return bool(significant) and float(comparison[relative_key]) >= minimum_relative
def representation_summary(report: dict) -> dict:
    metrics = report["metrics"]
    return {
        "status": report["status"],
        "checkpoint": os.path.abspath(report["checkpoint"]),
        "checkpoint_global_step": int(report["checkpoint_global_step"]),
        "checkpoint_phase": report["checkpoint_phase"],
        "data": os.path.abspath(report["data"]),
        "data_sha256": str(report["data_sha256"]),
        "split": report["split"],
        "requested_max_items": int(report["requested_max_items"]),
        "available_samples": int(report["available_samples"]),
        "samples": int(report["samples"]),
        "clusters": int(report["source_clusters"]),
        "history_frames": int(report["history_frames"]),
        "future_frames": int(report["future_frames"]),
        "anchors": [int(anchor) for anchor in report["anchors"]],
        "feature_improvement_vs_global": float(
            metrics["feature_improvement_vs_global"]
        ),
        "feature_vs_global": metrics["feature_vs_global_paired"],
        "feature_slot_conditioning": metrics[
            "feature_slot_conditioning_paired"
        ],
        "rgb_improvement_vs_global": float(
            metrics["rgb_improvement_vs_global_color"]
        ),
        "rgb_vs_global": metrics["rgb_vs_global_color_paired"],
        "rgb_slot_conditioning": metrics["rgb_slot_conditioning_paired"],
    }
def component_summary(report: dict) -> dict:
    evaluation = report["evaluation"]
    return {
        "status": report["status"],
        "checkpoint": os.path.abspath(report["checkpoint"]),
        "checkpoint_global_step": int(report["checkpoint_global_step"]),
        "checkpoint_phase": report["checkpoint_phase"],
        "data": os.path.abspath(report["data"]),
        "data_sha256": str(report["data_sha256"]),
        "split": report["split"],
        "requested_max_items": int(report["requested_max_items"]),
        "available_samples": int(report["available_samples"]),
        "anchors": [int(anchor) for anchor in report["anchors"]],
        "history_frames": int(report["history_frames"]),
        "future_frames": int(report["future_frames"]),
        "samples": int(evaluation["samples"]),
        "clusters": int(evaluation["clusters"]),
        "action_dim": int(report["action_dim"]),
        "canonical_action_dim": int(report["canonical_action_dim"]),
        "action_residual_dim": int(report["action_residual_dim"]),
        "comparison": evaluation["comparison"],
        "change_weighted_comparison": evaluation[
            "change_weighted_comparison"
        ],
        "task_group_evidence": evaluation["task_group_evidence"],
        "by_future_query": evaluation["by_future_query"],
        "component_slot_rms": evaluation[
            "posterior_component_slot_rms"
        ],
    }

def component_sample_coverage(
    summary: dict,
    minimum_samples: int,
) -> dict[str, int | bool]:
    anchors = summary["anchors"]
    expected_evaluated = min(
        summary["requested_max_items"],
        summary["available_samples"],
    )
    complete = (
        len(anchors) > 0
        and len(set(anchors)) == len(anchors)
        and summary["requested_max_items"] >= minimum_samples
        and summary["samples"] == expected_evaluated
        and summary["samples"] >= minimum_samples
    )
    return {
        "anchor_count": len(anchors),
        "requested_max_items": summary["requested_max_items"],
        "expected_evaluated_samples": expected_evaluated,
        "observed_available_samples": summary["available_samples"],
        "observed_evaluated_samples": summary["samples"],
        "minimum_samples": minimum_samples,
        "complete": complete,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_heldseed", required=True)
    parser.add_argument("--representation_heldtask", required=True)
    parser.add_argument("--component_heldseed", required=True)
    parser.add_argument("--component_heldtask", required=True)
    parser.add_argument("--collapse_gate", required=True)
    parser.add_argument("--longitudinal_gate", required=True)
    parser.add_argument("--transfer_gate", required=True)
    parser.add_argument("--causal_contract", required=True)
    parser.add_argument("--sequence_data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--minimum_representation_samples",
        type=int,
        default=144,
    )
    parser.add_argument(
        "--minimum_heldseed_dynamics_samples",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--minimum_heldtask_dynamics_samples",
        type=int,
        default=450,
    )
    parser.add_argument(
        "--minimum_representation_clusters",
        type=int,
        default=20,
    )
    parser.add_argument("--minimum_dynamics_clusters", type=int, default=100)
    parser.add_argument("--required_checkpoint_step", type=int, default=12000)
    parser.add_argument(
        "--minimum_representation_improvement",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--minimum_slot_conditioning",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--minimum_rgb_improvement",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--minimum_rgb_slot_conditioning",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--minimum_component_improvement",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--minimum_component_slot_rms",
        type=float,
        default=1e-4,
    )
    args = parser.parse_args()

    representation = {
        "heldseed": representation_summary(
            load_report(os.path.abspath(args.representation_heldseed))
        ),
        "heldtask": representation_summary(
            load_report(os.path.abspath(args.representation_heldtask))
        ),
    }
    components = {
        "heldseed": component_summary(
            load_report(os.path.abspath(args.component_heldseed))
        ),
        "heldtask": component_summary(
            load_report(os.path.abspath(args.component_heldtask))
        ),
    }
    collapse = load_report(os.path.abspath(args.collapse_gate))
    longitudinal = load_report(os.path.abspath(args.longitudinal_gate))
    transfer = load_report(os.path.abspath(args.transfer_gate))
    causal = load_report(os.path.abspath(args.causal_contract))
    sequence_root = os.path.abspath(args.sequence_data_root)
    sequence_manifest = sequence_manifest_summary(sequence_root)
    source_index_sha256 = task_source_index_sha256(sequence_root)
    causal_warm_start = causal["model"]["warm_start"]

    checks: dict[str, bool] = {
        "collapse_repair_gate": collapse["status"] == "pass",
        "longitudinal_stability_gate": longitudinal["status"] == "pass",
        "action_transfer_gate": transfer["status"] == "pass",
        "causal_contract": (
            causal["status"] == "passed"
            and bool(causal["model"]["passed"])
        ),
        "causal_checkpoint_identity": (
            causal_warm_start["source_checkpoint_version"] == 27
            and int(causal_warm_start["loaded"]) > 0
            and not causal_warm_start["missing"]
            and not causal_warm_start["unexpected"]
            and not causal_warm_start["shape_mismatch"]
            and not causal_warm_start["transformed"]
            and not causal_warm_start["dropped"]
        ),
        "sequence_manifest_verified": (
            sequence_manifest["complete"]
            and sequence_manifest["episodes"] == 7007
            and sequence_manifest["file_inventory"]["valid"]
        ),
        "sequence_split_isolation": sequence_manifest["split_isolation"][
            "valid"
        ],
    }
    checkpoint_paths = {
        causal["checkpoint"],
        *(entry["checkpoint"] for entry in representation.values()),
        *(entry["checkpoint"] for entry in components.values()),
        *(
            entry["checkpoint"]
            for entry in collapse["evaluation"].values()
        ),
    }
    checks["longitudinal_checkpoint_provenance"] = os.path.abspath(longitudinal["final_checkpoint"]) in checkpoint_paths
    checks["transfer_checkpoint_provenance"] = {os.path.abspath(entry["checkpoint"]) for entry in transfer["summary"].values()} == checkpoint_paths
    checks["transfer_data_provenance"] = {os.path.abspath(entry["data"]) for entry in transfer["summary"].values()} == {sequence_root}
    checks["single_checkpoint_provenance"] = (
        len(checkpoint_paths) == 1 and "" not in checkpoint_paths
        and all(os.path.isfile(path) for path in checkpoint_paths)
    )
    checks["sequence_data_provenance"] = (
        {entry["data"] for entry in components.values()} == {sequence_root}
        and {
            os.path.abspath(entry["data"])
            for entry in collapse["evaluation"].values()
        }
        == {sequence_root}
        and os.path.abspath(causal["data_root"]) == sequence_root
    )
    checks["representation_data_provenance"] = (
        {entry["data"] for entry in representation.values()} == {sequence_root}
    )
    reported_data_hashes = {
        *(entry["data_sha256"] for entry in representation.values()),
        *(entry["data_sha256"] for entry in components.values()),
        *(
            entry["data_sha256"]
            for entry in collapse["evaluation"].values()
        ),
        *(
            entry["data_sha256"]
            for entry in transfer["summary"].values()
        ),
    }
    checks["sequence_data_manifest_identity"] = reported_data_hashes == {
        sequence_manifest["manifest_sha256"]
    }

    for split, summary in representation.items():
        checks[f"{split}_representation_report"] = (
            summary["status"] == "ok"
            and summary["split"] == split
            and summary["checkpoint_global_step"] == args.required_checkpoint_step
            and summary["checkpoint_phase"] == "joint"
            and summary["history_frames"] == 4
            and summary["future_frames"] == 4
            and summary["anchors"] == [3, 5, 8]
            and summary["samples"] == min(
                summary["requested_max_items"],
                summary["available_samples"],
            )
            and summary["samples"] >= args.minimum_representation_samples
            and summary["clusters"] >= args.minimum_representation_clusters
        )
        checks[f"{split}_feature_representation"] = (
            summary["feature_improvement_vs_global"]
            >= args.minimum_representation_improvement
            and paired_signal(summary["feature_vs_global"], 0.0)
            and paired_signal(
                summary["feature_slot_conditioning"],
                args.minimum_slot_conditioning,
            )
        )
        checks[f"{split}_rgb_representation"] = (
            summary["rgb_improvement_vs_global"]
            >= args.minimum_rgb_improvement
            and paired_signal(summary["rgb_vs_global"], 0.0)
            and paired_signal(
                summary["rgb_slot_conditioning"],
                args.minimum_rgb_slot_conditioning,
            )
        )

    primary_metrics = ("feature_mse", "latent_mse")
    observed_metrics = (*primary_metrics, "rgb_distance")
    minimum_dynamics_samples = {
        "heldseed": args.minimum_heldseed_dynamics_samples,
        "heldtask": args.minimum_heldtask_dynamics_samples,
    }
    for split, summary in components.items():
        comparisons = summary["comparison"]
        coverage = component_sample_coverage(
            summary,
            minimum_dynamics_samples[split],
        )
        summary["sample_coverage"] = coverage
        checks[f"{split}_component_report"] = (
            summary["status"] == "ok"
            and summary["split"] == split
            and coverage["complete"]
            and summary["clusters"] >= args.minimum_dynamics_clusters
            and summary["checkpoint_global_step"] == args.required_checkpoint_step
            and summary["checkpoint_phase"] == "joint"
            and summary["history_frames"] == 4
            and summary["future_frames"] == 4
            and summary["anchors"] == [3, 5, 8]
            and summary["action_dim"] == 14
            and summary["canonical_action_dim"] == 6
            and summary["action_residual_dim"] == 8
        )
        checks[f"{split}_full_action_beats_baselines"] = all(
            paired_signal(comparisons[metric][reference], 0.0)
            for metric in observed_metrics
            for reference in (
                "posterior_over_zero",
                "posterior_over_shuffled",
                "posterior_over_copy",
            )
        )
        changed = summary["change_weighted_comparison"]
        checks[f"{split}_full_action_changes_observed_regions"] = all(
            paired_signal(changed[metric][reference], 0.0)
            for metric in (
                "change_weighted_feature_mse",
                "change_weighted_rgb_charbonnier",
            )
            for reference in (
                "posterior_over_zero",
                "posterior_over_shuffled",
                "posterior_over_copy",
            )
        )
        checks[f"{split}_task_group_consistency"] = task_evidence_passes(
            summary["task_group_evidence"], split,
            summary["samples"], summary["clusters"],
            source_index_sha256,
            ("change_weighted_feature_mse", "change_weighted_rgb_charbonnier"),
            ("posterior_over_zero", "posterior_over_shuffled", "posterior_over_copy"),
            MINIMUM_TASK_MEDIAN_RELATIVE,
        )
        longest_query = summary["by_future_query"].get("3")
        checks[f"{split}_longest_horizon_uses_action"] = (
            set(summary["by_future_query"]) == {"0", "1", "2", "3"}
            and longest_query is not None
            and all(
                paired_signal(
                    longest_query["comparison"][metric][reference],
                    0.0,
                )
                for metric in (
                    "change_weighted_feature_mse",
                    "change_weighted_rgb_charbonnier",
                )
                for reference in (
                    "posterior_over_zero",
                    "posterior_over_shuffled",
                    "posterior_over_copy",
                )
            )
        )
        checks[f"{split}_canonical_anchor_standalone"] = all(
            paired_signal(
                comparisons[metric]["canonical_over_zero"],
                0.0,
            )
            for metric in primary_metrics
        ) and all(
            paired_signal(
                changed[metric]["canonical_over_zero"],
                0.0,
            )
            for metric in (
                "change_weighted_feature_mse",
                "change_weighted_rgb_charbonnier",
            )
        )
        checks[f"{split}_learned_residual_contributes"] = all(
            paired_signal(
                comparisons[metric]["posterior_over_canonical"],
                args.minimum_component_improvement,
            )
            for metric in primary_metrics
        ) and all(
            paired_signal(changed[metric]["posterior_over_canonical"], args.minimum_component_improvement)
            for metric in ("change_weighted_feature_mse", "change_weighted_rgb_charbonnier")
        ) and (
            float(summary["component_slot_rms"]["canonical_only"])
            >= args.minimum_component_slot_rms
        )
        checks[f"{split}_canonical_anchor_contributes"] = all(
            paired_signal(
                comparisons[metric]["posterior_over_residual"],
                args.minimum_component_improvement,
            )
            for metric in primary_metrics
        ) and all(
            paired_signal(changed[metric]["posterior_over_residual"], args.minimum_component_improvement)
            for metric in ("change_weighted_feature_mse", "change_weighted_rgb_charbonnier")
        ) and (
            float(summary["component_slot_rms"]["residual_only"])
            >= args.minimum_component_slot_rms
        )

    passed = all(checks.values())
    report = {
        "status": "pass" if passed else "fail",
        "scope": "future_conditioned_posterior_core_only",
        "deployable_world_model_proven": False,
        "checkpoint": sorted(checkpoint_paths),
        "thresholds": {
            "minimum_representation_samples": (
                args.minimum_representation_samples
            ),
            "minimum_heldseed_dynamics_samples": (
                args.minimum_heldseed_dynamics_samples
            ),
            "minimum_heldtask_dynamics_samples": (
                args.minimum_heldtask_dynamics_samples
            ),
            "minimum_representation_clusters": (
                args.minimum_representation_clusters
            ),
            "minimum_dynamics_clusters": args.minimum_dynamics_clusters,
            "required_checkpoint_step": args.required_checkpoint_step,
            "minimum_representation_improvement": (
                args.minimum_representation_improvement
            ),
            "minimum_slot_conditioning": args.minimum_slot_conditioning,
            "minimum_rgb_improvement": args.minimum_rgb_improvement,
            "minimum_rgb_slot_conditioning": (
                args.minimum_rgb_slot_conditioning
            ),
            "minimum_component_improvement": (
                args.minimum_component_improvement
            ),
            "minimum_component_slot_rms": args.minimum_component_slot_rms,
            "task_group_contract": TASK_CONTRACT,
            "minimum_task_median_relative": MINIMUM_TASK_MEDIAN_RELATIVE,
        },
        "checks": checks,
        "representation": representation,
        "components": components,
        "collapse_gate": collapse,
        "longitudinal_gate": longitudinal,
        "action_transfer_gate": transfer,
        "causal_contract": causal,
        "data_contracts": {
            "sequence": sequence_manifest,
        },
        "limitations": [
            "Future-conditioned Posterior is an oracle available only in training.",
            "The history-only Prior is frozen and is not validated by this gate.",
            "Observational video does not prove paired counterfactual causality.",
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
        raise AssertionError(f"posterior-core design gate failed: {failed}")

if __name__ == "__main__":
    main()
