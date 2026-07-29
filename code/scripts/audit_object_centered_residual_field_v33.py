#!/usr/bin/env python3
"""Held-set attribution audit for signed object-centered residual fields."""

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
from igsw.adaptive_gaussian_wm.object_centered_residual_field import (  # noqa: E402
    fit_residual_fields,
    render_geometric_residual_field,
)
from igsw.adaptive_gaussian_wm.object_centered_residual_report import (  # noqa: E402
    summarize_residual_field_audit,
)
from igsw.adaptive_gaussian_wm.residual_field_audit_contract import (  # noqa: E402
    allocation_standard_deviation,
    append_fit_rows,
    baseline_comparison,
    load_v32_report,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402


CONTRACT = "object_centered_signed_residual_field_capacity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v31_report", required=True)
    parser.add_argument("--v32_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--max_items", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--budgets", default="64,128,256")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--minimum_utility_fraction", type=float, default=0.001)
    parser.add_argument("--maximum_scene_fraction", type=float, default=0.25)
    parser.add_argument("--minimum_compact_recovery", type=float, default=0.70)
    parser.add_argument("--maximum_token_ratio", type=float, default=1.10)
    parser.add_argument("--maximum_condition_number", type=float, default=5e6)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3301)
    return parser.parse_args()


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    require(bool(budgets), "--budgets is empty")
    require(
        tuple(sorted(set(budgets))) == budgets and min(budgets) > 0,
        "--budgets must be unique, increasing, positive integers",
    )
    return budgets


def main() -> None:
    args = parse_args()
    for name in ("data", "checkpoint", "v31_report", "v32_report", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(args.max_items > 0 and args.workers >= 0, "invalid dataset sizes")
    require(args.ridge > 0.0, "ridge regularization must be positive")
    require(args.bootstrap_samples > 0, "bootstrap sample count must be positive")
    require(
        0.0 < args.minimum_compact_recovery <= 1.0,
        "minimum compact recovery must be in (0,1]",
    )
    require(args.maximum_token_ratio >= 1.0, "token ratio must be at least one")
    require(args.maximum_condition_number > 1.0, "condition bound must exceed one")
    require(torch.cuda.is_available(), "residual-field audit requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(
        not status.strip(), "residual-field audit requires tracked files to be clean"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    budgets = parse_budgets(args.budgets)
    require(min(budgets) == 64, "compact residual gate requires a 64 budget")
    require(max(budgets) >= 256, "capacity residual gate requires a 256 budget")
    dataset = CausalVisualSequenceDataset(
        args.data, args.split, max_items=args.max_items
    )
    checkpoint_sha256 = file_sha256(args.checkpoint)
    v31 = load_v31_baseline_report(
        args.v31_report,
        checkpoint_sha256,
        dataset.data_sha256,
        args.split,
        len(dataset),
    )
    v32 = load_v32_report(
        args.v32_report,
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
        for item_index, cpu_batch in enumerate(loader, start=1):
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
            current = fit_residual_fields(
                current_tokens,
                current_slots,
                current_features,
                current_coordinates,
                current_valid,
                budgets=budgets,
                ridge=args.ridge,
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
            append_fit_rows(rows, "current", current, budgets)
            append_row(
                rows,
                "current_object_allocation_std",
                allocation_standard_deviation(current),
            )

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
                oracle = fit_residual_fields(
                    target_tokens,
                    target_slots,
                    future_features,
                    future_coordinates,
                    future_valid,
                    budgets=budgets,
                    ridge=args.ridge,
                    minimum_utility_fraction=args.minimum_utility_fraction,
                    maximum_scene_fraction=args.maximum_scene_fraction,
                )
                persistence = render_geometric_residual_field(
                    current,
                    future_coordinates,
                    local_budget=maximum,
                )
                predicted_scale = dynamics.future_relative_scale[0, horizon].float()
                scale_ratio = predicted_scale / current_scale.clamp_min(1e-6)
                feature_delta = (
                    predicted_features[0, horizon].float() - current_object_feature
                )
                dynamic_prediction = render_geometric_residual_field(
                    current,
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
                append_fit_rows(rows, "future", oracle, budgets)
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
            if item_index % 8 == 0 or item_index == len(loader):
                print(
                    f"[object-centered-residual-v33] processed={item_index}/{len(loader)}",
                    flush=True,
                )

    stacked = {name: torch.cat(parts) for name, parts in rows.items()}
    statistics, checks, decision = summarize_residual_field_audit(
        stacked,
        budgets,
        args.minimum_compact_recovery,
        args.maximum_token_ratio,
        args.maximum_scene_fraction,
        args.maximum_condition_number,
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
        "budgets": list(budgets),
        "ridge": args.ridge,
        "minimum_utility_fraction": args.minimum_utility_fraction,
        "maximum_scene_fraction": args.maximum_scene_fraction,
        "minimum_compact_recovery": args.minimum_compact_recovery,
        "maximum_token_ratio": args.maximum_token_ratio,
        "maximum_condition_number": args.maximum_condition_number,
        "bootstrap_samples": args.bootstrap_samples,
        "v31_report": os.path.abspath(args.v31_report),
        "v31_report_sha256": file_sha256(args.v31_report),
        "v32_report": os.path.abspath(args.v32_report),
        "v32_report_sha256": file_sha256(args.v32_report),
        "baseline_comparison": baseline_comparison(v31, v32, statistics),
        "causal_contract": {
            "carrier_source": "last_history_frame_only",
            "dynamics_action": "zero_action_history_only_base",
            "future_features_used_by": "oracle_capacity_ceiling_only",
            "future_grid_role": "read_only_DINO_measurement_coordinates",
            "oracle_object_gate_uses_dense_assignment": True,
            "geometric_object_renderer_uses_dense_assignment": False,
            "future_geometric_capacity_uses_target_slot_geometry": True,
            "deployable_dynamics_uses_current_field_and_predicted_geometry": True,
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
