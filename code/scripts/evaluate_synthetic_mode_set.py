"""Evaluate one synthetic mode-set checkpoint over multiple held seeds."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.dirname(__file__))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from validate_adaptive_gaussian_architecture import evaluate_model  # noqa: E402


def flatten_metrics(
    value,
    prefix: str = "",
) -> dict[str, float]:
    result = {}
    if isinstance(value, dict):
        for name, child in value.items():
            child_prefix = f"{prefix}/{name}" if prefix else name
            result.update(flatten_metrics(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def summarize(
    per_seed: list[dict],
) -> dict[str, dict[str, float]]:
    flattened = [flatten_metrics(item["metrics"]) for item in per_seed]
    shared = set(flattened[0])
    for item in flattened[1:]:
        shared.intersection_update(item)
    result = {}
    for name in sorted(shared):
        values = [item[name] for item in flattened]
        result[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation_seeds", type=int, nargs="+", required=True)
    parser.add_argument("--eval_groups", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--prior_samples", type=int, default=16)
    parser.add_argument("--semantic_branch_strength", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(
        args.eval_groups,
        args.grid_size,
        args.future_steps,
        args.prior_samples,
    ) <= 0:
        raise ValueError(
            "eval_groups, grid_size, future_steps, and prior_samples "
            "must be positive"
        )
    if not args.evaluation_seeds:
        raise ValueError("at least one evaluation seed is required")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        raise ValueError("evaluation seeds must be unique")

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    per_seed = []
    for seed in args.evaluation_seeds:
        metrics = evaluate_model(
            model,
            args.eval_groups,
            args.grid_size,
            args.future_steps,
            args.prior_samples,
            device,
            args.semantic_branch_strength,
            False,
            seed,
        )
        per_seed.append({"evaluation_seed": seed, "metrics": metrics})
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "evaluation_seeds": args.evaluation_seeds,
        "eval_groups": args.eval_groups,
        "prior_samples": args.prior_samples,
        "per_seed": per_seed,
        "summary": summarize(per_seed),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
