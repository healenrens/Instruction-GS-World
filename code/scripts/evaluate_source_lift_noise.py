"""Evaluate source-component sampling separately from residual source noise."""
from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise_scale", type=float, required=True)
    parser.add_argument("--eval_groups", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--prior_samples", type=int, default=16)
    parser.add_argument("--semantic_branch_strength", type=float, default=2.0)
    parser.add_argument("--evaluation_seed", type=int, default=9107)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    prior = model.latent_actions.prior
    if not hasattr(prior, "sample_noise_scale"):
        raise ValueError("checkpoint does not use a source-lifted prior")
    prior.sample_noise_scale = args.noise_scale
    metrics = evaluate_model(
        model,
        args.eval_groups,
        args.grid_size,
        args.future_steps,
        args.prior_samples,
        device,
        args.semantic_branch_strength,
        False,
        args.evaluation_seed,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "noise_scale": args.noise_scale,
        "evaluation_seed": args.evaluation_seed,
        "metrics": metrics,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
