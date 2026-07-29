#!/usr/bin/env python3
"""Held-set audit for anisotropic object-centered attention carriers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.object_centered_attention_probe import (  # noqa: E402
    fit_attention_carriers,
    render_attention_carriers,
)
from igsw.adaptive_gaussian_wm.object_centered_attention_report import (  # noqa: E402
    summarize_attention_audit,
)
from igsw.adaptive_gaussian_wm.object_centered_audit_runtime import (  # noqa: E402
    action_free_prediction,
    append_row,
    autocast_context,
    file_sha256,
    future_content_swap_differences,
    load_object_memory_model,
    load_v31_baseline_report,
    require,
)
from igsw.adaptive_gaussian_wm.object_centered_carrier_probe import (  # noqa: E402
    feature_error,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402


CONTRACT = "object_centered_attention_carrier_capacity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--max_items", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extra_budgets", default="64,128,256")
    parser.add_argument("--minimum_utility_fraction", type=float, default=0.001)
    parser.add_argument("--maximum_scene_fraction", type=float, default=0.25)
    parser.add_argument("--minimum_compact_recovery", type=float, default=0.70)
    parser.add_argument("--maximum_token_ratio", type=float, default=1.10)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3201)
    return parser.parse_args()


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    require(bool(budgets), "--extra_budgets is empty")
    require(
        tuple(sorted(set(budgets))) == budgets and min(budgets) > 0,
        "--extra_budgets must be unique, increasing, positive integers",
    )
    return budgets


def _baseline_comparison(baseline: dict, statistics: dict) -> dict:
    old = baseline["means"]
    new = statistics["means"]
    comparisons = {
        "current_b64_absolute_gain": (
            old["current_full_b64"] - new["current_attention_b64"]
        ),
        "future_b64_absolute_gain": (
            old["future_oracle_b64"] - new["future_oracle_b64"]
        ),
        "current_b64_error_ratio_to_v31": (
            new["current_attention_b64"] / old["current_full_b64"]
        ),
        "future_b64_error_ratio_to_v31": (
            new["future_oracle_b64"] / old["future_oracle_b64"]
        ),
    }
    return {
        "v31_decision": baseline["decision"],
        "v31_git_commit": baseline["git_commit"],
        "v31_current_b64": old["current_full_b64"],
        "v31_future_b64": old["future_oracle_b64"],
        **comparisons,
    }


def main() -> None:
    args = parse_args()
    for name in ("data", "checkpoint", "baseline_report", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(args.max_items > 0 and args.workers >= 0, "invalid dataset sizes")
    require(args.bootstrap_samples > 0, "bootstrap sample count must be positive")
    require(
        0.0 < args.minimum_compact_recovery <= 1.0,
        "minimum compact recovery must be in (0,1]",
    )
    require(
        args.maximum_token_ratio >= 1.0,
        "maximum token ratio must be at least one",
    )
    require(
        0.0 <= args.maximum_scene_fraction <= 1.0,
        "maximum scene fraction must be in [0,1]",
    )
    require(torch.cuda.is_available(), "attention carrier audit requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(not status.strip(), "attention audit requires tracked files to be clean")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    budgets = parse_budgets(args.extra_budgets)
    require(min(budgets) == 64, "compact attention gate requires a 64 budget")
    require(max(budgets) >= 256, "capacity attention gate requires a 256 budget")
    dataset = CausalVisualSequenceDataset(
        args.data, args.split, max_items=args.max_items
    )
    checkpoint_sha256 = file_sha256(args.checkpoint)
    baseline = load_v31_baseline_report(
        args.baseline_report,
        checkpoint_sha256,
        dataset.data_sha256,
        args.split,
        len(dataset),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda:0")
    checkpoint_metadata, model = load_object_memory_model(args.checkpoint, device)
    rows: dict[str, list[torch.Tensor]] = {}
    causal_differences: dict[str, float] | None = None
    maximum = max(budgets)

    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            with autocast_context():
                history = model.encode_history(batch)
                _, target_future = model.encode_targets(batch)
                dynamics = action_free_prediction(model, batch, history)
                predicted_features = model.object_aggregator.decode_feature(
                    dynamics.future_slots
                )
            if causal_differences is None:
                causal_differences = future_content_swap_differences(
                    model, batch, history, dynamics
                )

            current_features = batch["history_features"][0, -1].float()
            current_coordinates = batch["history_coordinates"][0, -1].float()
            current_valid = batch["history_valid"][0, -1]
            current_tokens = history["token_states"][-1]
            current_slots = history["slot_states"][-1]
            current = fit_attention_carriers(
                current_tokens,
                current_slots,
                current_features,
                current_coordinates,
                current_valid,
                extra_budgets=budgets,
                minimum_utility_fraction=args.minimum_utility_fraction,
                maximum_scene_fraction=args.maximum_scene_fraction,
            )
            append_row(rows, "current_sequence_index", batch["sequence_index"])
            append_row(
                rows,
                "current_token",
                feature_error(
                    current_tokens.reconstructed_features[0],
                    current_features,
                    current_valid,
                ),
            )
            append_row(rows, "current_root", current.root_error)
            for budget in budgets:
                append_row(
                    rows,
                    f"current_attention_b{budget}",
                    current.budget_errors[budget],
                )
            append_row(rows, "current_selected", current.selected_local_carriers)
            append_row(rows, "current_scene_selected", current.selected_scene_carriers)
            append_row(rows, "current_halted", float(current.stopped_by_utility))
            active_counts = current.local_carriers_per_object[
                current.carriers.object_active
            ].float()
            allocation_std = (
                active_counts.std(unbiased=False)
                if active_counts.numel() > 1
                else active_counts.new_zeros(())
            )
            append_row(rows, "current_object_allocation_std", allocation_std)

            current_scale = history["relative_scale"][0, -1].float()
            current_object_feature = current_slots.decoded_feature[0].float()
            for horizon, (target_tokens, target_slots) in enumerate(
                zip(
                    target_future["token_states"],
                    target_future["slot_states"],
                    strict=True,
                )
            ):
                future_features = batch["future_features"][0, horizon].float()
                future_coordinates = batch["future_coordinates"][0, horizon].float()
                future_valid = batch["future_valid"][0, horizon]
                oracle = fit_attention_carriers(
                    target_tokens,
                    target_slots,
                    future_features,
                    future_coordinates,
                    future_valid,
                    extra_budgets=budgets,
                    minimum_utility_fraction=args.minimum_utility_fraction,
                    maximum_scene_fraction=args.maximum_scene_fraction,
                )
                persistence = render_attention_carriers(
                    current.carriers,
                    future_coordinates,
                    local_budget=maximum,
                )
                predicted_scale = dynamics.future_relative_scale[0, horizon].float()
                scale_ratio = predicted_scale / current_scale.clamp_min(1e-6)
                feature_delta = (
                    predicted_features[0, horizon].float() - current_object_feature
                )
                dynamic_prediction = render_attention_carriers(
                    current.carriers,
                    future_coordinates,
                    local_budget=maximum,
                    target_object_centers=dynamics.future_centers[0, horizon].float(),
                    object_scale_ratio=scale_ratio,
                    object_feature_delta=feature_delta,
                )
                append_row(rows, "future_sequence_index", batch["sequence_index"])
                append_row(
                    rows,
                    "future_token",
                    feature_error(
                        target_tokens.reconstructed_features[0],
                        future_features,
                        future_valid,
                    ),
                )
                append_row(rows, "future_root", oracle.root_error)
                for budget in budgets:
                    append_row(
                        rows,
                        f"future_oracle_b{budget}",
                        oracle.budget_errors[budget],
                    )
                append_row(rows, "future_selected", oracle.selected_local_carriers)
                append_row(
                    rows, "future_scene_selected", oracle.selected_scene_carriers
                )
                append_row(rows, "future_halted", float(oracle.stopped_by_utility))
                append_row(
                    rows,
                    "future_persistence",
                    feature_error(persistence, future_features, future_valid),
                )
                append_row(
                    rows,
                    "future_dynamics",
                    feature_error(dynamic_prediction, future_features, future_valid),
                )

    stacked = {name: torch.cat(parts) for name, parts in rows.items()}
    statistics, checks, decision = summarize_attention_audit(
        stacked,
        budgets,
        args.minimum_compact_recovery,
        args.maximum_token_ratio,
        args.maximum_scene_fraction,
        args.bootstrap_samples,
        args.seed,
    )
    require(causal_differences is not None, "causal audit did not process a sample")
    causal_pass = (
        causal_differences["future_content_swap_input_max_difference"] > 1e-6
        and causal_differences["history_future_content_swap_max_difference"] < 1e-6
        and causal_differences["dynamics_future_content_swap_max_difference"] < 1e-6
    )
    checks["future_content_isolated_from_history_and_dynamics"] = causal_pass
    if not causal_pass:
        decision = "inconclusive"
    report = {
        "status": "completed",
        "decision": decision,
        "contract": CONTRACT,
        "git_commit": commit,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": checkpoint_metadata,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": dataset.data_sha256,
        "held_split": args.split,
        "held_items": len(dataset),
        "episode_clusters": int(
            torch.unique(stacked["current_sequence_index"]).numel()
        ),
        "extra_carrier_budgets": list(budgets),
        "minimum_utility_fraction": args.minimum_utility_fraction,
        "maximum_scene_fraction": args.maximum_scene_fraction,
        "minimum_compact_recovery": args.minimum_compact_recovery,
        "maximum_token_ratio": args.maximum_token_ratio,
        "bootstrap_samples": args.bootstrap_samples,
        "baseline_report": os.path.abspath(args.baseline_report),
        "baseline_report_sha256": file_sha256(args.baseline_report),
        "baseline_comparison": _baseline_comparison(baseline, statistics),
        "causal_contract": {
            "carrier_source": "last_history_frame_only",
            "dynamics_action": "zero_action_history_only_base",
            "future_features_used_by": "oracle_capacity_ceiling_only",
            "future_grid_role": "read_only_DINO_measurement_coordinates",
            "dense_object_assignment_used_by_renderer": False,
            **causal_differences,
        },
        "checks": checks,
        **statistics,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
