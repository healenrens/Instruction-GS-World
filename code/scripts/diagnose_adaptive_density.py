"""Server-side rate-distortion reachability diagnostic for adaptive GPSTokens."""
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
    active = torch.einsum(
        "bmn,bmc->bnc",
        active_assignment,
        state.decoded_features,
    )
    valid_weight = valid.to(target.dtype)[..., None]
    background = (target * valid_weight).sum(dim=1, keepdim=True)
    background = background / valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    reconstruction = active + (1.0 - coverage)[..., None] * background
    return (reconstruction - target).square().mean(dim=(1, 2))


def globally_budgeted_choice(
    errors: torch.Tensor,
    budgets: torch.Tensor,
    target_mean: float,
) -> torch.Tensor:
    lower = errors.new_tensor(-1.0)
    upper = errors.new_tensor(1.0)
    for _ in range(64):
        penalty = 0.5 * (lower + upper)
        choice = (errors + penalty * budgets[None]).argmin(dim=1)
        mean_count = budgets[choice].mean()
        if float(mean_count) > target_mean:
            lower = penalty
        else:
            upper = penalty
    penalty = 0.5 * (lower + upper)
    return (errors + penalty * budgets[None]).argmin(dim=1)


def complexity_budget(
    features: torch.Tensor,
    budgets: torch.Tensor,
    target_mean: float,
) -> torch.Tensor:
    feature_mean = features.mean(dim=1, keepdim=True)
    complexity = (
        (features - feature_mean).square().mean(dim=-1).mean(dim=1).sqrt()
    )
    relative = complexity / complexity.mean().clamp_min(1e-6)
    count = (target_mean * relative.sqrt()).clamp(
        float(budgets.min()),
        float(budgets.max()),
    )
    count = count * target_mean / count.mean().clamp_min(1e-6)
    return (count[:, None] - budgets[None]).abs().argmin(dim=1)


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--batch_size", type=int, default=192)
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
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state_dict["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()

    torch.manual_seed(args.seed)
    batch = make_synthetic_batch(
        config.feature_dim,
        args.batch_size,
        history_frames=3,
        future_steps=2,
        grid_size=args.grid_size,
        device=device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
        max_objects=config.object_slots,
        ambiguous_fraction=0.5,
        balanced_ambiguity=True,
        semantic_branch_strength=2.0,
    )
    features = batch["history_features"][:, -1]
    coordinates = batch["history_coordinates"][:, -1]
    valid = batch["history_valid"][:, -1]
    token_state = model.allocator(features, coordinates, valid)
    learned_error = reconstruction_error(
        token_state,
        features,
        valid,
        token_state.activation,
    )
    implementation_error = (
        token_state.reconstructed_features - features
    ).square().mean(dim=(1, 2))
    if float((learned_error - implementation_error).abs().max()) > 1e-6:
        raise ValueError("diagnostic reconstruction does not match allocator")

    budgets = torch.tensor(args.budgets, device=device, dtype=features.dtype)
    candidate_errors = []
    for budget in budgets:
        count = features.new_full((features.shape[0],), float(budget))
        activation = budget_activation(token_state.activation_logits, count)
        candidate_errors.append(
            reconstruction_error(token_state, features, valid, activation)
        )
    candidate_errors = torch.stack(candidate_errors, dim=1)
    fixed_choice = (budgets - args.target_mean).abs().argmin()
    fixed_count = features.new_full((features.shape[0],), args.target_mean)
    oracle_choice = globally_budgeted_choice(
        candidate_errors,
        budgets,
        args.target_mean,
    )
    heuristic_choice = complexity_budget(
        features,
        budgets,
        args.target_mean,
    )
    row = torch.arange(features.shape[0], device=device)
    complexity = batch["complexity"]
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "budgets": [float(value) for value in budgets],
        "target_mean": args.target_mean,
        "policies": {
            "learned_soft": policy_metrics(
                learned_error,
                token_state.activation.sum(dim=1).squeeze(-1),
                complexity,
            ),
            "fixed": policy_metrics(
                candidate_errors[:, fixed_choice],
                fixed_count,
                complexity,
            ),
            "complexity_heuristic": policy_metrics(
                candidate_errors[row, heuristic_choice],
                budgets[heuristic_choice],
                complexity,
            ),
            "oracle_rate_distortion": policy_metrics(
                candidate_errors[row, oracle_choice],
                budgets[oracle_choice],
                complexity,
            ),
        },
        "mse_by_budget": {
            str(float(budget)): float(candidate_errors[:, index].mean())
            for index, budget in enumerate(budgets)
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
