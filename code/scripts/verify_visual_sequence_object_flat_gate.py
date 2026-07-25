"""Gate the object representation against matched dense sequence baselines."""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    object_flat_task_gate_entries,
)


def add_check(
    checks: list[dict],
    name: str,
    passed: bool,
    observed,
    required,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def comparison(report: dict, metric: str, name: str) -> dict:
    return report["evaluation"]["comparison"][metric][name]


def check_split(
    report: dict,
    expected_split: str,
    minimum_samples: int,
    minimum_clusters: int,
    superiority: float,
    checks: list[dict],
) -> None:
    prefix = expected_split
    evaluation = report.get("evaluation", {})
    add_check(
        checks,
        f"{prefix}.report_status",
        report.get("status") == "ok" and report.get("split") == expected_split,
        {"status": report.get("status"), "split": report.get("split")},
        {"status": "ok", "split": expected_split},
    )
    samples = int(evaluation.get("samples", 0))
    clusters = int(evaluation.get("clusters", 0))
    requested = int(report.get("requested_max_items", 0))
    available = int(report.get("available_samples", 0))
    add_check(
        checks,
        f"{prefix}.sample_coverage",
        requested >= minimum_samples
        and samples == min(requested, available)
        and samples >= minimum_samples,
        {"requested": requested, "available": available, "samples": samples},
        {"complete": True, "minimum_samples": minimum_samples},
    )
    add_check(
        checks,
        f"{prefix}.episode_clusters",
        clusters >= minimum_clusters,
        clusters,
        f">={minimum_clusters}",
    )
    action = evaluation.get("action_statistics", {}).get("flat_posterior", {})
    action_std = float(action.get("sample_std_mean", 0.0))
    action_rms = float(action.get("rms", 0.0))
    add_check(
        checks,
        f"{prefix}.flat_action_noncollapsed",
        action_std > 1e-4 and action_rms > 1e-4,
        {"sample_std_mean": action_std, "rms": action_rms},
        {"sample_std_mean": ">1e-4", "rms": ">1e-4"},
    )
    causal = evaluation.get("causal_probe", {})
    add_check(
        checks,
        f"{prefix}.flat_causal_bottleneck",
        causal.get("posterior_responds_to_future") is True
        and causal.get("history_has_no_future_input") is True
        and causal.get("dynamics_has_no_direct_future_input") is True,
        causal,
        {
            "posterior_responds_to_future": True,
            "history_has_no_future_input": True,
            "dynamics_has_no_direct_future_input": True,
        },
    )
    whole_metrics = {
        "feature": "feature_mse",
        "rgb": "rgb_distance",
    }
    change_metrics = {
        "feature": "change_weighted_feature_mse",
        "rgb": "change_weighted_rgb_distance",
    }
    for label, metric in whole_metrics.items():
        learned = comparison(report, metric, "flat_history_over_copy")
        add_check(
            checks,
            f"{prefix}.history_flat_beats_copy_{label}",
            learned["relative_improvement"] >= 0.02
            and learned["positive_ci95_lower"] is True,
            learned,
            {"relative_improvement": ">=0.02", "positive_ci95_lower": True},
        )
        noninferiority = evaluation.get(
            "object_vs_flat_noninferiority_5pct",
            {},
        ).get(metric, {})
        add_check(
            checks,
            f"{prefix}.object_whole_{label}_noninferior",
            noninferiority.get("noninferior_ci95") is True,
            noninferiority,
            {"relative_margin": 0.05, "noninferior_ci95": True},
        )
    for label, metric in change_metrics.items():
        posterior = comparison(report, metric, "flat_posterior_over_history")
        add_check(
            checks,
            f"{prefix}.flat_posterior_uses_future_{label}",
            posterior["relative_improvement"] >= superiority
            and posterior["positive_ci95_lower"] is True,
            posterior,
            {
                "relative_improvement": f">={superiority}",
                "positive_ci95_lower": True,
            },
        )
        object_gain = comparison(report, metric, "object_over_flat_posterior")
        add_check(
            checks,
            f"{prefix}.object_beats_flat_on_change_{label}",
            object_gain["relative_improvement"] >= superiority
            and object_gain["positive_ci95_lower"] is True,
            object_gain,
            {
                "relative_improvement": f">={superiority}",
                "positive_ci95_lower": True,
            },
        )

    rgb_regions = evaluation.get("rgb_regions", {})
    region_reports = rgb_regions.get("regions", {})
    change_region = region_reports.get("change", {})
    static_region = region_reports.get("static", {})
    add_check(
        checks,
        f"{prefix}.rgb_region_contract",
        rgb_regions.get("mask_source")
        == "ground_truth_current_and_future_rgb_for_evaluation_only"
        and int(change_region.get("frames", 0)) >= minimum_samples
        and int(static_region.get("frames", 0)) >= minimum_samples
        and int(change_region.get("clusters", 0)) >= minimum_clusters
        and int(static_region.get("clusters", 0)) >= minimum_clusters,
        {
            "mask_source": rgb_regions.get("mask_source"),
            "change_frames": change_region.get("frames"),
            "change_clusters": change_region.get("clusters"),
            "static_frames": static_region.get("frames"),
            "static_clusters": static_region.get("clusters"),
        },
        {
            "mask_source": "ground_truth_current_and_future_rgb_for_evaluation_only",
            "minimum_frames": minimum_samples,
            "minimum_clusters": minimum_clusters,
        },
    )
    for name, label in (
        ("flat_posterior_over_history", "flat_posterior_uses_future"),
        ("object_over_flat_posterior", "object_beats_flat"),
    ):
        region_gain = change_region.get("comparison", {}).get(name, {})
        add_check(
            checks,
            f"{prefix}.{label}_on_observed_change_rgb",
            region_gain.get("relative_improvement", float("-inf"))
            >= superiority
            and region_gain.get("positive_ci95_lower") is True,
            region_gain,
            {
                "relative_improvement": f">={superiority}",
                "positive_ci95_lower": True,
            },
        )
    static_noninferiority = rgb_regions.get(
        "object_vs_flat_static_noninferiority_5pct",
        {},
    )
    add_check(
        checks,
        f"{prefix}.object_static_rgb_noninferior",
        static_noninferiority.get("noninferior_ci95") is True,
        static_noninferiority,
        {"relative_margin": 0.05, "noninferior_ci95": True},
    )
    for name, passed, observed, required in object_flat_task_gate_entries(
        evaluation.get("task_group_evidence", {}), expected_split,
        samples, clusters, report.get("task_source_index_sha256", ""),
        superiority
    ):
        add_check(checks, f"{prefix}.{name}", passed, observed, required)
    longest = str(int(report.get("future_frames", 0)) - 1)
    add_check(
        checks,
        f"{prefix}.future_query_contract",
        int(report.get("history_frames", 0)) == 4
        and int(report.get("future_frames", 0)) == 4
        and report.get("anchors") == [3, 5, 8]
        and set(evaluation.get("by_future_query", {})) == {"0", "1", "2", "3"},
        {
            "history_frames": report.get("history_frames"),
            "future_frames": report.get("future_frames"),
            "anchors": report.get("anchors"),
            "future_queries": sorted(evaluation.get("by_future_query", {})),
        },
        {"history_frames": 4, "future_frames": 4, "anchors": [3, 5, 8]},
    )
    for label, metric in change_metrics.items():
        longest_gain = (
            evaluation.get("by_future_query", {})
            .get(longest, {})
            .get("comparison", {})
            .get(metric, {})
            .get("object_over_flat_posterior", {})
        )
        add_check(
            checks,
            f"{prefix}.object_beats_flat_longest_horizon_{label}",
            longest_gain.get("relative_improvement", float("-inf")) >= superiority
            and longest_gain.get("positive_ci95_lower") is True,
            longest_gain,
            {
                "query_index": longest,
                "relative_improvement": f">={superiority}",
                "positive_ci95_lower": True,
            },
        )


def verify(
    heldseed: dict,
    heldtask: dict,
    required_step: int,
    heldseed_samples: int,
    heldtask_samples: int,
    minimum_clusters: int,
    superiority: float,
) -> dict:
    checks: list[dict] = []
    identity_fields = (
        "object_checkpoint",
        "object_checkpoint_sha256",
        "flat_checkpoint",
        "flat_checkpoint_sha256",
        "data",
        "data_sha256",
        "task_source_index_sha256",
        "history_frames",
        "future_frames",
        "anchors",
        "action_contract",
        "parameter_count",
        "training_contract",
    )
    mismatches = {
        name: {"heldseed": heldseed.get(name), "heldtask": heldtask.get(name)}
        for name in identity_fields
        if heldseed.get(name) != heldtask.get(name)
    }
    add_check(
        checks,
        "cross_split_identity",
        not mismatches,
        mismatches,
        {},
    )
    steps = {
        "heldseed_object": heldseed.get("object_checkpoint_global_step"),
        "heldseed_flat": heldseed.get("flat_checkpoint_global_step"),
        "heldtask_object": heldtask.get("object_checkpoint_global_step"),
        "heldtask_flat": heldtask.get("flat_checkpoint_global_step"),
    }
    add_check(
        checks,
        "checkpoint_step_identity",
        all(value == required_step for value in steps.values()),
        steps,
        required_step,
    )
    checkpoint_hashes = {
        name: heldseed.get(name)
        for name in ("object_checkpoint_sha256", "flat_checkpoint_sha256")
    }
    add_check(
        checks,
        "checkpoint_hash_contract",
        all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in checkpoint_hashes.values()
        ),
        checkpoint_hashes,
        "64-character lowercase SHA-256 values",
    )
    action = heldseed.get("action_contract", {})
    add_check(
        checks,
        "continuous_matched_action_contract",
        action.get("type") == "continuous"
        and int(action.get("tokens", 0)) == 16
        and int(action.get("dimensions", 0)) == 14
        and int(action.get("state_tokens", 0)) == 16
        and action.get("layout")
        == "dino_effect_3_plus_rgb_logit_effect_3_plus_residual_8"
        and action.get("object_source")
        == "future_conditioned_object_posterior_oracle"
        and action.get("flat_source")
        == "future_conditioned_unstructured_latent_posterior_oracle"
        and action.get("dynamics_future_access") == "latent_action_only"
        and action.get("flat_posterior_future_modalities") == ["dino", "rgb"],
        action,
        {
            "type": "continuous",
            "tokens": 16,
            "dimensions": 14,
            "state_tokens": 16,
            "layout": "dino_effect_3_plus_rgb_logit_effect_3_plus_residual_8",
            "object_source": "future_conditioned_object_posterior_oracle",
            "flat_source": "future_conditioned_unstructured_latent_posterior_oracle",
            "dynamics_future_access": "latent_action_only",
            "flat_posterior_future_modalities": ["dino", "rgb"],
        },
    )
    parameter_count = heldseed.get("parameter_count", {})
    capacity_ratio = float(parameter_count.get("flat_to_object_ratio", 0.0))
    add_check(
        checks,
        "capacity_matched_unstructured_baseline",
        0.75 <= capacity_ratio <= 1.25,
        parameter_count,
        {"flat_to_object_ratio": "[0.75,1.25]"},
    )
    training = heldseed.get("training_contract", {})
    add_check(
        checks,
        "shared_dynamics_and_conservative_flat_budget",
        training.get("object_checkpoint_phase_steps") == required_step
        and training.get("optimizer_contract")
        == "posterior_core_matched_flat_optimization_v1"
        and training.get("flat_additional_steps") == required_step
        and training.get("object_effective_global_batch") == 256
        and training.get("flat_effective_global_batch") == 256
        and training.get("shared_dynamics_initialization")
        == "object_dynamics_blocks_only"
        and training.get("optimization_budget_bias")
        == "flat_receives_additional_updates_after_object_source"
        and training.get("modality_matching") == "dino_rgb"
        and training.get("flat_rgb_supervision") is True
        and training.get("flat_posterior_observes_future_rgb") is True
        and training.get("history_encoder_input") == "dino_only",
        training,
        {
            "object_checkpoint_phase_steps": required_step,
            "optimizer_contract": "posterior_core_matched_flat_optimization_v1",
            "flat_additional_steps": required_step,
            "effective_global_batch": 256,
            "shared_dynamics_initialization": "object_dynamics_blocks_only",
            "optimization_budget_bias": "flat_receives_additional_updates_after_object_source",
            "modality_matching": "dino_rgb",
            "flat_rgb_supervision": True,
            "flat_posterior_observes_future_rgb": True,
            "history_encoder_input": "dino_only",
        },
    )
    check_split(
        heldseed,
        "heldseed",
        heldseed_samples,
        minimum_clusters,
        superiority,
        checks,
    )
    check_split(
        heldtask,
        "heldtask",
        heldtask_samples,
        minimum_clusters,
        superiority,
        checks,
    )
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "status": "pass" if not failed else "fail",
        "gate": "visual_sequence_object_vs_matched_flat_v2",
        "required_step": required_step,
        "evidence_identity": {
            "object_checkpoint": heldseed.get("object_checkpoint"),
            "object_checkpoint_sha256": heldseed.get("object_checkpoint_sha256"),
            "flat_checkpoint": heldseed.get("flat_checkpoint"),
            "flat_checkpoint_sha256": heldseed.get("flat_checkpoint_sha256"),
            "data": heldseed.get("data"),
            "data_sha256": heldseed.get("data_sha256"),
            "task_source_index_sha256": heldseed.get(
                "task_source_index_sha256"
            ),
            "action_contract": heldseed.get("action_contract"),
            "parameter_count": heldseed.get("parameter_count"),
            "training_contract": heldseed.get("training_contract"),
        },
        "thresholds": {
            "heldseed_samples": heldseed_samples,
            "heldtask_samples": heldtask_samples,
            "minimum_episode_clusters": minimum_clusters,
            "minimum_relative_superiority": superiority,
            "history_flat_over_copy": 0.02,
            "whole_feature_and_rgb_noninferiority_margin": 0.05,
            "static_rgb_noninferiority_margin": 0.05,
            "flat_to_object_parameter_ratio": [0.75, 1.25],
        },
        "checks": checks,
        "failed_checks": failed,
        "scope": {
            "claim": "internal_representation_sanity_not_final_paper_superiority",
            "posterior_paths": "future_conditioned_non_deployable_oracles",
            "remaining_unmatched": [
                "cumulative_pre_reference_optimization",
                "object_center_rgb_anchor_vs_dense_dino_rgb_effect_anchor",
                "object_gaussian_readout_vs_dense_grid_readout",
                "object_specific_latent_and_assignment_auxiliaries",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldseed", required=True)
    parser.add_argument("--heldtask", required=True)
    parser.add_argument("--required_step", type=int, default=12000)
    parser.add_argument("--heldseed_samples", type=int, default=1024)
    parser.add_argument("--heldtask_samples", type=int, default=450)
    parser.add_argument("--minimum_clusters", type=int, default=100)
    parser.add_argument("--superiority", type=float, default=0.03)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.heldseed, args.heldtask, args.output):
        if not os.path.isabs(path):
            raise ValueError("object-flat gate paths must be absolute")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite gate report: {args.output}")
    if min(
        args.required_step,
        args.heldseed_samples,
        args.heldtask_samples,
        args.minimum_clusters,
    ) <= 0:
        raise ValueError("object-flat gate sizes must be positive")
    if args.superiority < 0.0:
        raise ValueError("object-flat superiority threshold cannot be negative")
    report = verify(
        load_report(args.heldseed),
        load_report(args.heldtask),
        args.required_step,
        args.heldseed_samples,
        args.heldtask_samples,
        args.minimum_clusters,
        args.superiority,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
