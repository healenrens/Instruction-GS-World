"""Validate a frozen current-only rate-distortion GPSToken budget policy."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.metrics import pearson_correlation  # noqa: E402


def budget_activation(
    logits: torch.Tensor,
    target_count: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise ValueError("activation logits must have shape [B,M,1]")
    if target_count.shape != (logits.shape[0],):
        raise ValueError("target_count must have shape [B]")
    lower = logits.amin(dim=1, keepdim=True) - 20.0
    upper = logits.amax(dim=1, keepdim=True) + 20.0
    target = target_count[:, None, None]
    for _ in range(32):
        threshold = 0.5 * (lower + upper)
        count = torch.sigmoid(logits - threshold).sum(dim=1, keepdim=True)
        lower = torch.where(count > target, threshold, lower)
        upper = torch.where(count > target, upper, threshold)
    return torch.sigmoid(logits - 0.5 * (lower + upper))


def reconstruction_error(
    state,
    target: torch.Tensor,
    valid: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    active_assignment = state.assignment * activation
    coverage = active_assignment.sum(dim=1).clamp(0.0, 1.0)
    reconstruction = torch.einsum(
        "bmn,bmc->bnc",
        active_assignment,
        state.decoded_features,
    )
    valid_weight = valid.to(target.dtype)[..., None]
    background = (target * valid_weight).sum(dim=1, keepdim=True)
    background = background / valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    reconstruction = reconstruction + (1.0 - coverage)[..., None] * background
    return (reconstruction - target).square().mean(dim=(1, 2))


def current_frame_candidates(
    model: AdaptiveGaussianObjectWorldModel,
    batch: dict[str, torch.Tensor],
    budgets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = batch["history_features"][:, -1]
    coordinates = batch["history_coordinates"][:, -1]
    valid = batch["history_valid"][:, -1]
    token_state = model.allocator(features, coordinates, valid)
    errors = []
    for budget in budgets:
        target = features.new_full((features.shape[0],), float(budget))
        activation = budget_activation(token_state.activation_logits, target)
        errors.append(
            reconstruction_error(token_state, features, valid, activation)
        )
    candidate_errors = torch.stack(errors, dim=1)
    fixed_index = (budgets - budgets.new_tensor(
        model.config.max_micro_tokens * model.config.fixed_token_fraction
    )).abs().argmin()
    implementation_error = (
        token_state.reconstructed_features - features
    ).square().mean(dim=(1, 2))
    mismatch = (
        candidate_errors[:, fixed_index] - implementation_error
    ).abs().max()
    if float(mismatch) > 1e-6:
        raise ValueError("candidate reconstruction does not match allocator")
    return candidate_errors, features, batch["complexity"]


def choose_with_penalty(
    errors: torch.Tensor,
    budgets: torch.Tensor,
    penalty: torch.Tensor,
) -> torch.Tensor:
    return (errors + penalty * budgets[None]).argmin(dim=1)


def relative_routing_errors(
    errors: torch.Tensor,
    budgets: torch.Tensor,
    target_mean: float,
) -> torch.Tensor:
    fixed_index = (budgets - target_mean).abs().argmin()
    scale = errors[:, fixed_index, None].clamp_min(1e-8)
    return errors / scale


def calibrate_penalty(
    errors: torch.Tensor,
    budgets: torch.Tensor,
    target_mean: float,
) -> torch.Tensor:
    lower = errors.new_tensor(-1.0)
    upper = errors.new_tensor(1.0)
    lower_mean = budgets[
        choose_with_penalty(errors, budgets, lower)
    ].mean()
    upper_mean = budgets[
        choose_with_penalty(errors, budgets, upper)
    ].mean()
    if float(lower_mean) < target_mean or float(upper_mean) > target_mean:
        raise ValueError("target budget is outside the calibrated policy range")
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        mean_count = budgets[
            choose_with_penalty(errors, budgets, midpoint)
        ].mean()
        if float(mean_count) > target_mean:
            lower = midpoint
        else:
            upper = midpoint
    candidates = torch.stack((lower, 0.5 * (lower + upper), upper))
    deviations = torch.stack(
        [
            (
                budgets[choose_with_penalty(errors, budgets, value)].mean()
                - target_mean
            ).abs()
            for value in candidates
        ]
    )
    return candidates[deviations.argmin()]


def policy_metrics(
    error: torch.Tensor,
    count: torch.Tensor,
    complexity: torch.Tensor,
) -> dict[str, float]:
    low = complexity <= torch.quantile(complexity, 1.0 / 3.0)
    high = complexity >= torch.quantile(complexity, 2.0 / 3.0)
    return {
        "mse": float(error.mean()),
        "low_complexity_mse": float(error[low].mean()),
        "high_complexity_mse": float(error[high].mean()),
        "token_mean": float(count.mean()),
        "token_std": float(count.std()),
        "token_complexity_pearson": pearson_correlation(count, complexity),
    }


def evaluate_policy(
    errors: torch.Tensor,
    routing_errors: torch.Tensor,
    budgets: torch.Tensor,
    complexity: torch.Tensor,
    target_mean: float,
    penalty: torch.Tensor,
) -> dict[str, dict[str, float]]:
    row = torch.arange(errors.shape[0], device=errors.device)
    fixed_index = (budgets - target_mean).abs().argmin()
    fixed_count = budgets[fixed_index].expand(errors.shape[0])
    choice = choose_with_penalty(routing_errors, budgets, penalty)
    adaptive = policy_metrics(
        errors[row, choice],
        budgets[choice],
        complexity,
    )
    fixed = policy_metrics(
        errors[:, fixed_index],
        fixed_count,
        complexity,
    )
    return {
        "fixed": fixed,
        "calibrated_rate_distortion": adaptive,
    }


def make_current_batch(
    config: AdaptiveGaussianWMConfig,
    seed: int,
    batch_size: int,
    grid_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return make_synthetic_batch(
        config.feature_dim,
        batch_size,
        history_frames=3,
        future_steps=2,
        grid_size=grid_size,
        device=device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
        max_objects=config.object_slots,
        ambiguous_fraction=0.5,
        balanced_ambiguity=True,
        semantic_branch_strength=2.0,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_seed", type=int, default=9107)
    parser.add_argument(
        "--evaluation_seeds",
        type=int,
        nargs="+",
        default=(9201, 9203, 9205, 9207, 9209),
    )
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--target_mean", type=float, default=12.0)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=(6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0),
    )
    args = parser.parse_args()
    if args.batch_size % 3:
        raise ValueError("batch_size must be divisible by three")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.density_mode != "fixed":
        raise ValueError("rate-distortion calibration requires a fixed tokenizer")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    budgets = torch.tensor(args.budgets, device=device)

    calibration_batch = make_current_batch(
        config,
        args.calibration_seed,
        args.batch_size,
        args.grid_size,
        device,
    )
    calibration_errors, _, calibration_complexity = current_frame_candidates(
        model,
        calibration_batch,
        budgets,
    )
    calibration_routing_errors = relative_routing_errors(
        calibration_errors,
        budgets,
        args.target_mean,
    )
    penalty = calibrate_penalty(
        calibration_routing_errors,
        budgets,
        args.target_mean,
    )
    calibration_metrics = evaluate_policy(
        calibration_errors,
        calibration_routing_errors,
        budgets,
        calibration_complexity,
        args.target_mean,
        penalty,
    )

    evaluations = {}
    relative_gains = []
    high_complexity_gains = []
    budget_errors = []
    budget_means = []
    correlations = []
    for seed in args.evaluation_seeds:
        batch = make_current_batch(
            config,
            seed,
            args.batch_size,
            args.grid_size,
            device,
        )
        errors, _, complexity = current_frame_candidates(model, batch, budgets)
        routing_errors = relative_routing_errors(
            errors,
            budgets,
            args.target_mean,
        )
        policies = evaluate_policy(
            errors,
            routing_errors,
            budgets,
            complexity,
            args.target_mean,
            penalty,
        )
        fixed = policies["fixed"]
        adaptive = policies["calibrated_rate_distortion"]
        relative_gain = 1.0 - adaptive["mse"] / fixed["mse"]
        high_gain = (
            1.0
            - adaptive["high_complexity_mse"]
            / fixed["high_complexity_mse"]
        )
        budget_error = abs(adaptive["token_mean"] - args.target_mean)
        budget_error = budget_error / args.target_mean
        evaluations[str(seed)] = {
            "policies": policies,
            "relative_mse_gain": relative_gain,
            "high_complexity_relative_mse_gain": high_gain,
            "relative_budget_error": budget_error,
        }
        relative_gains.append(relative_gain)
        high_complexity_gains.append(high_gain)
        budget_errors.append(budget_error)
        budget_means.append(adaptive["token_mean"])
        correlations.append(adaptive["token_complexity_pearson"])

    gates = {
        "all_seed_mse_gain_positive": min(relative_gains) > 0.0,
        "all_seed_high_complexity_gain_positive": (
            min(high_complexity_gains) > 0.0
        ),
        "all_seed_mean_budget_at_or_below_target": (
            max(budget_means) <= args.target_mean
        ),
        "all_seed_budget_utilization_at_least_90_percent": (
            min(budget_means) / args.target_mean >= 0.9
        ),
        "all_seed_token_complexity_pearson_above_0_3": (
            min(correlations) >= 0.3
        ),
    }
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "input_contract": {
            "uses_current_features": True,
            "uses_current_coordinates": True,
            "uses_current_valid_mask": True,
            "uses_future_frames": False,
            "uses_future_actions": False,
            "evaluation_complexity_used_by_policy": False,
        },
        "calibration_seed": args.calibration_seed,
        "evaluation_seeds": args.evaluation_seeds,
        "batch_size": args.batch_size,
        "budgets": [float(value) for value in budgets],
        "target_mean": args.target_mean,
        "distortion_normalization": "per-sample fixed-budget reconstruction MSE",
        "calibrated_penalty": float(penalty),
        "calibration_metrics": calibration_metrics,
        "evaluations": evaluations,
        "aggregate": {
            "minimum_relative_mse_gain": min(relative_gains),
            "mean_relative_mse_gain": sum(relative_gains) / len(relative_gains),
            "minimum_high_complexity_relative_mse_gain": (
                min(high_complexity_gains)
            ),
            "maximum_relative_budget_error": max(budget_errors),
            "maximum_token_mean": max(budget_means),
            "minimum_budget_utilization": (
                min(budget_means) / args.target_mean
            ),
            "minimum_token_complexity_pearson": min(correlations),
        },
        "gates": gates,
        "gate_passed": all(gates.values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
