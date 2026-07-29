#!/usr/bin/env python3
"""Held-set capacity and transport audit for stable object residual fields."""

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
    require,
)
from igsw.adaptive_gaussian_wm.object_centered_carrier_probe import (  # noqa: E402
    feature_error,
)
from igsw.adaptive_gaussian_wm.object_centered_residual_field import (  # noqa: E402
    render_geometric_residual_field,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.stable_object_residual_transport import (  # noqa: E402
    fit_stable_residual_field,
    render_stable_residual_field,
    transport_sensitivity,
)
from igsw.adaptive_gaussian_wm.stable_residual_transport_report import (  # noqa: E402
    summarize_stable_transport_audit,
)
from igsw.adaptive_gaussian_wm.stable_transport_audit_contract import (  # noqa: E402
    append_sensitivity_rows,
    append_stable_fit_rows,
    load_v33_report,
    v33_comparison,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402


CONTRACT = "stable_object_residual_transport_capacity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v33_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--max_items", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--budgets", default="64,128,256")
    parser.add_argument("--compact_budget", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--minimum_utility_fraction", type=float, default=0.001)
    parser.add_argument("--maximum_scene_fraction", type=float, default=0.25)
    parser.add_argument("--center_perturbation_fraction", type=float, default=0.01)
    parser.add_argument("--scale_log_perturbation", type=float, default=0.01)
    parser.add_argument("--minimum_compact_recovery", type=float, default=0.70)
    parser.add_argument("--maximum_token_ratio", type=float, default=1.10)
    parser.add_argument("--maximum_coefficient_ratio", type=float, default=10.0)
    parser.add_argument("--maximum_sensitivity_ratio", type=float, default=1.0)
    parser.add_argument("--maximum_dynamics_ratio", type=float, default=2.0)
    parser.add_argument("--maximum_condition_number", type=float, default=5e6)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3401)
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
    for name in ("data", "checkpoint", "v33_report", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(args.max_items > 0 and args.workers >= 0, "invalid dataset sizes")
    require(args.ridge > 0.0, "ridge regularization must be positive")
    require(args.bootstrap_samples > 0, "bootstrap sample count must be positive")
    require(
        0.0 <= args.minimum_utility_fraction < 1.0,
        "minimum utility fraction must be in [0,1)",
    )
    require(
        0.0 <= args.maximum_scene_fraction <= 1.0,
        "maximum scene fraction must be in [0,1]",
    )
    require(
        0.0 < args.minimum_compact_recovery <= 1.0,
        "minimum compact recovery must be in (0,1]",
    )
    require(args.maximum_token_ratio >= 1.0, "token ratio must be at least one")
    require(torch.cuda.is_available(), "stable residual audit requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(
        not status.strip(), "stable residual audit requires tracked files to be clean"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    budgets = parse_budgets(args.budgets)
    require(args.compact_budget in budgets, "compact budget must be audited")
    require(max(budgets) >= 256, "capacity gate requires a 256 budget")
    require(
        all(
            value > 0.0
            for value in (
                args.center_perturbation_fraction,
                args.scale_log_perturbation,
                args.minimum_compact_recovery,
                args.maximum_token_ratio,
                args.maximum_coefficient_ratio,
                args.maximum_sensitivity_ratio,
                args.maximum_dynamics_ratio,
                args.maximum_condition_number,
            )
        ),
        "audit thresholds and perturbations must be positive",
    )
    dataset = CausalVisualSequenceDataset(
        args.data, args.split, max_items=args.max_items
    )
    checkpoint_sha256 = file_sha256(args.checkpoint)
    v33 = load_v33_report(
        args.v33_report,
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
            current = fit_stable_residual_field(
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
            append_stable_fit_rows(rows, "current", current, budgets)
            append_sensitivity_rows(
                rows,
                transport_sensitivity(
                    current,
                    current_coordinates,
                    current_valid,
                    local_budget=maximum,
                    center_fraction=args.center_perturbation_fraction,
                    scale_log_delta=args.scale_log_perturbation,
                ),
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
                future = fit_stable_residual_field(
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
                gated_persistence = render_geometric_residual_field(
                    current.reference, future_coordinates, local_budget=maximum
                )
                stable_persistence = render_stable_residual_field(
                    current, future_coordinates, local_budget=maximum
                )
                predicted_scale = dynamics.future_relative_scale[0, horizon].float()
                scale_ratio = predicted_scale / current_scale.clamp_min(1e-6)
                feature_delta = (
                    predicted_features[0, horizon].float() - current_object_feature
                )
                render_arguments = {
                    "local_budget": maximum,
                    "target_object_centers": dynamics.future_centers[
                        0, horizon
                    ].float(),
                    "object_scale_ratio": scale_ratio,
                    "object_feature_delta": feature_delta,
                }
                gated_dynamics = render_geometric_residual_field(
                    current.reference, future_coordinates, **render_arguments
                )
                stable_dynamics = render_stable_residual_field(
                    current, future_coordinates, **render_arguments
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
                append_stable_fit_rows(rows, "future", future, budgets)
                for name, prediction in (
                    ("gated_persistence", gated_persistence),
                    ("stable_persistence", stable_persistence),
                    ("gated_dynamics", gated_dynamics),
                    ("stable_dynamics", stable_dynamics),
                ):
                    append_row(
                        rows,
                        f"future_{name}",
                        feature_error(prediction, future_features, future_valid),
                    )
            if item_index % 8 == 0 or item_index == len(loader):
                print(
                    f"[stable-object-residual-v34] processed={item_index}/{len(loader)}",
                    flush=True,
                )

    stacked = {name: torch.cat(parts) for name, parts in rows.items()}
    statistics, checks, decision = summarize_stable_transport_audit(
        stacked,
        budgets,
        args.compact_budget,
        args.minimum_compact_recovery,
        args.maximum_token_ratio,
        args.maximum_coefficient_ratio,
        args.maximum_sensitivity_ratio,
        args.maximum_dynamics_ratio,
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
    comparison = v33_comparison(v33, statistics, budgets)
    reproduction_pass = comparison["v33_reproduction_max_abs_difference"] < 1e-5
    checks["v33_gated_metrics_reproduced"] = reproduction_pass
    if not reproduction_pass:
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
        "compact_budget": args.compact_budget,
        "ridge": args.ridge,
        "center_perturbation_fraction": args.center_perturbation_fraction,
        "scale_log_perturbation": args.scale_log_perturbation,
        "minimum_compact_recovery": args.minimum_compact_recovery,
        "maximum_token_ratio": args.maximum_token_ratio,
        "maximum_coefficient_ratio": args.maximum_coefficient_ratio,
        "maximum_sensitivity_ratio": args.maximum_sensitivity_ratio,
        "maximum_dynamics_ratio": args.maximum_dynamics_ratio,
        "maximum_condition_number": args.maximum_condition_number,
        "bootstrap_samples": args.bootstrap_samples,
        "v33_report": os.path.abspath(args.v33_report),
        "v33_report_sha256": file_sha256(args.v33_report),
        "v33_comparison": comparison,
        "causal_contract": {
            "field_source": "last_history_frame_only",
            "dynamics_action": "zero_action_history_only_base",
            "future_features_used_by": "oracle_capacity_ceiling_only",
            "future_grid_role": "read_only_DINO_measurement_coordinates",
            "root_gate_source": "geometric_object_state",
            "local_residual_gate": "object_relative_rbf_only",
            "stable_renderer_uses_dense_assignment": False,
            "future_capacity_uses_target_slot_geometry": True,
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
