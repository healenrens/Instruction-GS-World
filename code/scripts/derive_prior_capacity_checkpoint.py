"""Resize only the Flow Prior while preserving the frozen world model."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--flow_hidden_dim", type=int, required=True)
    parser.add_argument("--flow_layers", type=int, required=True)
    parser.add_argument("--prior_effect_weight", type=float, default=3.0)
    parser.add_argument("--initialization_seed", type=int, default=7001)
    parser.add_argument("--flow_source_scale", type=float, default=1.0)
    parser.add_argument("--multi_query_prior_context", action="store_true")
    parser.add_argument("--prior_query_residual", action="store_true")
    parser.add_argument("--mode_set_global_codebook", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if state["variant"] != "full":
        raise ValueError("source checkpoint must use the full variant")
    source_config = AdaptiveGaussianWMConfig(**state["config"])
    source_model = AdaptiveGaussianObjectWorldModel(source_config)
    source_model.load_state_dict(state["model"], strict=True)

    target_config = replace(
        source_config,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_layers=args.flow_layers,
        flow_endpoint_prediction=True,
        prior_effect_weight=args.prior_effect_weight,
        correlated_flow_source=False,
        flow_source_scale=args.flow_source_scale,
        multi_query_prior_context=args.multi_query_prior_context,
        prior_query_residual=args.prior_query_residual,
        mode_set_global_codebook=args.mode_set_global_codebook,
    )
    torch.manual_seed(args.initialization_seed)
    target_model = AdaptiveGaussianObjectWorldModel(target_config)
    source_tensors = source_model.state_dict()
    target_tensors = target_model.state_dict()
    copied = []
    reinitialized = []
    for name, target_tensor in target_tensors.items():
        source_tensor = source_tensors.get(name)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            target_tensors[name] = source_tensor.clone()
            copied.append(name)
        elif name.startswith("latent_actions.prior."):
            reinitialized.append(name)
        else:
            raise ValueError(f"non-prior tensor changed shape: {name}")
    target_model.load_state_dict(target_tensors, strict=True)
    non_prior_exact = all(
        torch.equal(target_tensors[name], source_tensors[name])
        for name in copied
        if not name.startswith("latent_actions.prior.")
    )
    if not non_prior_exact:
        raise ValueError("non-prior learned tensors changed")

    output = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    torch.save(
        {
            "variant": state["variant"],
            "seed": state["seed"],
            "config": target_config.to_dict(),
            "model": target_model.state_dict(),
            "derived_from": os.path.abspath(args.input),
        },
        output,
    )
    report = {
        "status": "ok",
        "source_checkpoint": os.path.abspath(args.input),
        "output_checkpoint": output,
        "source_flow_hidden_dim": source_config.flow_hidden_dim,
        "target_flow_hidden_dim": target_config.flow_hidden_dim,
        "source_flow_layers": source_config.flow_layers,
        "target_flow_layers": target_config.flow_layers,
        "prior_effect_weight": target_config.prior_effect_weight,
        "initialization_seed": args.initialization_seed,
        "flow_source_scale": target_config.flow_source_scale,
        "multi_query_prior_context": target_config.multi_query_prior_context,
        "prior_query_residual": target_config.prior_query_residual,
        "mode_set_global_codebook": target_config.mode_set_global_codebook,
        "total_parameter_count": sum(
            parameter.numel() for parameter in target_model.parameters()
        ),
        "prior_parameter_count": sum(
            parameter.numel()
            for parameter in target_model.latent_actions.prior.parameters()
        ),
        "copied_tensor_count": len(copied),
        "reinitialized_tensor_count": len(reinitialized),
        "reinitialized_parameter_count": sum(
            target_tensors[name].numel() for name in reinitialized
        ),
        "non_prior_exact": non_prior_exact,
        "reinitialized_tensors": reinitialized,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
