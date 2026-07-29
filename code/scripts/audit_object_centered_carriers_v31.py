#!/usr/bin/env python3
"""Held-set capacity audit for adaptive object-centered feature carriers."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import AdaptiveGaussianObjectWorldModel  # noqa: E402
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
)
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.dynamics_runtime import (  # noqa: E402
    run_object_dynamics,
)
from igsw.adaptive_gaussian_wm.object_centered_carrier_probe import (  # noqa: E402
    feature_error,
    fit_object_centered_carriers,
    render_carriers,
)
from igsw.adaptive_gaussian_wm.object_centered_carrier_report import (  # noqa: E402
    summarize_carrier_audit,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v28_training import ARCHITECTURE  # noqa: E402


CONTRACT = "object_centered_adaptive_carrier_capacity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--max_items", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extra_budgets", default="8,16,32,64")
    parser.add_argument("--minimum_marginal_gain", type=float, default=1e-5)
    parser.add_argument("--minimum_gap_recovery", type=float, default=0.70)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3101)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    require(bool(budgets), "--extra_budgets is empty")
    require(
        tuple(sorted(set(budgets))) == budgets and min(budgets) > 0,
        "--extra_budgets must be unique, increasing, positive integers",
    )
    return budgets


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    version = int(checkpoint.get("checkpoint_version", -1))
    require(
        28 <= version <= CHECKPOINT_VERSION,
        f"checkpoint version {version} is outside the supported [28,{CHECKPOINT_VERSION}] range",
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    require(config.architecture == ARCHITECTURE, "checkpoint is not object_memory_v1")
    require(config.persistent_object_memory, "checkpoint has no object memory")
    require(config.factorized_dynamics, "checkpoint has no factorized Dynamics")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "phase": checkpoint.get("phase"),
        "phase_step": checkpoint.get("phase_step"),
        "git_commit": checkpoint.get("git_commit"),
        "training_stage": checkpoint.get("args", {}).get("training_stage"),
    }
    del checkpoint
    return metadata, model.eval()


def action_free_prediction(model, batch: dict, history: dict):
    history_scale = signed_gap_scale(batch["history_times"], model.config.gap_reference)
    future_scale = signed_gap_scale(batch["future_times"], model.config.gap_reference)
    actions = torch.zeros(
        batch["history_features"].shape[0],
        model.config.action_tokens,
        model.config.action_dim,
        device=batch["history_features"].device,
        dtype=history["slots"].dtype,
    )
    history_mask = torch.zeros(
        *history["slots"].shape[:3],
        dtype=torch.bool,
        device=history["slots"].device,
    )
    return run_object_dynamics(
        model,
        history["slots"],
        history["activity"],
        history_scale,
        future_scale,
        actions,
        history_mask,
        history["center"],
        None,
        history_relative_scale=history.get("relative_scale"),
        history_relative_disparity=history.get("relative_disparity"),
        history_relations=history.get("relations"),
        history_existence=history.get("existence"),
    )


def _scene_prediction(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    mean = (features.float() * weight[:, None]).sum(dim=0)
    mean = mean / weight.sum().clamp_min(1.0)
    return mean[None].expand_as(features)


def _append(rows: dict[str, list[torch.Tensor]], name: str, value) -> None:
    tensor = torch.as_tensor(value).detach().float().reshape(-1).cpu()
    rows.setdefault(name, []).append(tensor)


def autocast_context():
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )


def main() -> None:
    args = parse_args()
    for name in ("data", "checkpoint", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(args.max_items > 0 and args.workers >= 0, "invalid dataset sizes")
    require(args.bootstrap_samples > 0, "bootstrap sample count must be positive")
    require(
        0.0 < args.minimum_gap_recovery <= 1.0,
        "minimum gap recovery must be in (0,1]",
    )
    require(torch.cuda.is_available(), "carrier audit requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(not status.strip(), "carrier audit requires tracked files to be clean")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    budgets = parse_budgets(args.extra_budgets)
    dataset = CausalVisualSequenceDataset(
        args.data, args.split, max_items=args.max_items
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda:0")
    checkpoint_metadata, model = load_model(args.checkpoint, device)
    rows: dict[str, list[torch.Tensor]] = {}
    causal_differences: dict[str, float] | None = None
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
                swapped = dict(batch)
                swapped["future_features"] = batch["future_features"].flip(1)
                with autocast_context():
                    swapped_history = model.encode_history(swapped)
                    swapped_dynamics = action_free_prediction(
                        model, swapped, swapped_history
                    )
                causal_differences = {
                    "history_future_content_swap_max_difference": float(
                        torch.stack(
                            (
                                (
                                    history["slots"].float()
                                    - swapped_history["slots"].float()
                                )
                                .abs()
                                .max(),
                                (
                                    history["center"].float()
                                    - swapped_history["center"].float()
                                )
                                .abs()
                                .max(),
                                (
                                    history["relative_scale"].float()
                                    - swapped_history["relative_scale"].float()
                                )
                                .abs()
                                .max(),
                            )
                        ).max()
                    ),
                    "dynamics_future_content_swap_max_difference": float(
                        torch.stack(
                            (
                                (
                                    dynamics.future_slots.float()
                                    - swapped_dynamics.future_slots.float()
                                )
                                .abs()
                                .max(),
                                (
                                    dynamics.future_centers.float()
                                    - swapped_dynamics.future_centers.float()
                                )
                                .abs()
                                .max(),
                                (
                                    dynamics.future_relative_scale.float()
                                    - swapped_dynamics.future_relative_scale.float()
                                )
                                .abs()
                                .max(),
                            )
                        ).max()
                    ),
                }

            current_features = batch["history_features"][0, -1].float()
            current_coordinates = batch["history_coordinates"][0, -1].float()
            current_valid = batch["history_valid"][0, -1]
            current_tokens = history["token_states"][-1]
            current_slots = history["slot_states"][-1]
            current_object = fit_object_centered_carriers(
                current_tokens,
                current_slots,
                current_features,
                current_coordinates,
                current_valid,
                extra_budgets=budgets,
                allow_scene_carriers=False,
                minimum_marginal_gain=args.minimum_marginal_gain,
            )
            current_full = fit_object_centered_carriers(
                current_tokens,
                current_slots,
                current_features,
                current_coordinates,
                current_valid,
                extra_budgets=budgets,
                allow_scene_carriers=True,
                minimum_marginal_gain=args.minimum_marginal_gain,
            )
            _append(rows, "current_sequence_index", batch["sequence_index"])
            _append(
                rows,
                "current_scene",
                feature_error(
                    _scene_prediction(current_features, current_valid),
                    current_features,
                    current_valid,
                ),
            )
            _append(
                rows,
                "current_token",
                feature_error(
                    current_tokens.reconstructed_features[0],
                    current_features,
                    current_valid,
                ),
            )
            _append(rows, "current_root", current_full.root_error)
            for budget in budgets:
                _append(
                    rows,
                    f"current_object_b{budget}",
                    current_object.budget_errors[budget],
                )
                _append(
                    rows,
                    f"current_full_b{budget}",
                    current_full.budget_errors[budget],
                )
            _append(
                rows, "current_object_selected", current_object.selected_local_carriers
            )
            _append(rows, "current_full_selected", current_full.selected_local_carriers)
            _append(
                rows, "current_scene_selected", current_full.selected_scene_carriers
            )
            _append(
                rows,
                "current_full_halted",
                float(current_full.stopped_by_marginal_gain),
            )
            active_counts = current_full.local_carriers_per_object[
                current_full.carriers.object_active
            ].float()
            allocation_std = (
                active_counts.std(unbiased=False)
                if active_counts.numel() > 1
                else active_counts.new_zeros(())
            )
            _append(rows, "current_object_allocation_std", allocation_std)

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
                oracle = fit_object_centered_carriers(
                    target_tokens,
                    target_slots,
                    future_features,
                    future_coordinates,
                    future_valid,
                    extra_budgets=budgets,
                    allow_scene_carriers=True,
                    minimum_marginal_gain=args.minimum_marginal_gain,
                )
                persistence = render_carriers(current_full.carriers, future_coordinates)
                predicted_scale = dynamics.future_relative_scale[0, horizon].float()
                scale_ratio = predicted_scale / current_scale.clamp_min(1e-6)
                feature_delta = (
                    predicted_features[0, horizon].float() - current_object_feature
                )
                dynamic_prediction = render_carriers(
                    current_full.carriers,
                    future_coordinates,
                    target_object_centers=dynamics.future_centers[0, horizon].float(),
                    object_scale_ratio=scale_ratio,
                    object_feature_delta=feature_delta,
                )
                _append(rows, "future_sequence_index", batch["sequence_index"])
                _append(
                    rows,
                    "future_token",
                    feature_error(
                        target_tokens.reconstructed_features[0],
                        future_features,
                        future_valid,
                    ),
                )
                _append(rows, "future_root", oracle.root_error)
                for budget in budgets:
                    _append(
                        rows,
                        f"future_oracle_b{budget}",
                        oracle.budget_errors[budget],
                    )
                _append(rows, "future_oracle", oracle.budget_errors[max(budgets)])
                _append(rows, "future_oracle_selected", oracle.selected_local_carriers)
                _append(
                    rows, "future_oracle_scene_selected", oracle.selected_scene_carriers
                )
                _append(
                    rows,
                    "future_oracle_halted",
                    float(oracle.stopped_by_marginal_gain),
                )
                _append(
                    rows,
                    "future_persistence",
                    feature_error(persistence, future_features, future_valid),
                )
                _append(
                    rows,
                    "future_dynamics",
                    feature_error(dynamic_prediction, future_features, future_valid),
                )

    stacked = {name: torch.cat(parts) for name, parts in rows.items()}
    statistics, checks, decision = summarize_carrier_audit(
        stacked,
        budgets,
        args.minimum_gap_recovery,
        args.bootstrap_samples,
        args.seed,
    )
    require(causal_differences is not None, "causal audit did not process a sample")
    causal_pass = all(value < 1e-6 for value in causal_differences.values())
    checks["future_content_isolated_from_history_and_dynamics"] = causal_pass
    if not causal_pass:
        decision = "inconclusive"
    report = {
        "status": "completed",
        "decision": decision,
        "contract": CONTRACT,
        "git_commit": commit,
        "checkpoint_version": checkpoint_metadata["checkpoint_version"],
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": dataset.data_sha256,
        "held_split": args.split,
        "held_items": len(dataset),
        "episode_clusters": int(
            torch.unique(stacked["current_sequence_index"]).numel()
        ),
        "extra_carrier_budgets": list(budgets),
        "minimum_marginal_gain": args.minimum_marginal_gain,
        "minimum_gap_recovery": args.minimum_gap_recovery,
        "bootstrap_samples": args.bootstrap_samples,
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
