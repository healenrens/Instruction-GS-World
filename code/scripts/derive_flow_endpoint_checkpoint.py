"""Derive an endpoint-parameterized Flow Prior checkpoint."""
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
    parser.add_argument("--prior_effect_weight", type=float, default=0.0)
    parser.add_argument("--correlated_flow_source", action="store_true")
    parser.add_argument("--flow_source_scale", type=float, default=1.0)
    parser.add_argument("--multi_query_prior_context", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if state["variant"] != "full":
        raise ValueError("source checkpoint must use the full variant")

    source_config = AdaptiveGaussianWMConfig(**state["config"])
    if source_config.flow_endpoint_prediction:
        raise ValueError("source checkpoint already predicts flow endpoints")

    source_model = AdaptiveGaussianObjectWorldModel(source_config)
    source_model.load_state_dict(state["model"], strict=True)

    target_config = replace(
        source_config,
        flow_endpoint_prediction=True,
        prior_effect_weight=args.prior_effect_weight,
        correlated_flow_source=args.correlated_flow_source,
        flow_source_scale=args.flow_source_scale,
        multi_query_prior_context=args.multi_query_prior_context,
    )
    target_model = AdaptiveGaussianObjectWorldModel(target_config)
    target_model.load_state_dict(source_model.state_dict(), strict=True)

    source_tensors = source_model.state_dict()
    target_tensors = target_model.state_dict()
    exact_weights_equal = all(
        torch.equal(source_tensors[name], target_tensors[name])
        for name in source_tensors
    )
    if not exact_weights_equal:
        raise ValueError("derived checkpoint changed learned tensors")

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
        "variant": state["variant"],
        "source_flow_endpoint_prediction": (
            source_config.flow_endpoint_prediction
        ),
        "target_flow_endpoint_prediction": (
            target_config.flow_endpoint_prediction
        ),
        "joint_flow": target_config.joint_flow,
        "prior_effect_weight": target_config.prior_effect_weight,
        "correlated_flow_source": target_config.correlated_flow_source,
        "flow_source_scale": target_config.flow_source_scale,
        "multi_query_prior_context": target_config.multi_query_prior_context,
        "parameter_count": sum(p.numel() for p in target_model.parameters()),
        "state_tensor_count": len(target_tensors),
        "exact_weights_equal": exact_weights_equal,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
