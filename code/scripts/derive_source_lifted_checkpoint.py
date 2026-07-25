"""Replace only the Flow Prior with a source-lifted multimodal prior."""
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
    parser.add_argument("--flow_source_components", type=int, default=3)
    parser.add_argument("--flow_lift_scale", type=float, default=1.0)
    parser.add_argument("--flow_responsibility_floor", type=float, default=0.05)
    parser.add_argument("--flow_source_fit_weight", type=float, default=0.1)
    parser.add_argument("--flow_source_min_scale", type=float, default=0.05)
    parser.add_argument("--balanced_source_assignment", action="store_true")
    parser.add_argument("--flow_assignment_temperature", type=float, default=0.05)
    parser.add_argument("--prior_effect_weight", type=float, default=0.0)
    parser.add_argument("--initialization_seed", type=int, default=7001)
    parser.add_argument("--multi_query_prior_context", action="store_true")
    parser.add_argument("--mode_set_prior", action="store_true")
    parser.add_argument("--mode_set_geometry_weight", type=float, default=0.0)
    parser.add_argument("--mode_set_ordered_assignment", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if state["variant"] != "full":
        raise ValueError("source checkpoint must use the full variant")
    source_config = AdaptiveGaussianWMConfig(**state["config"])
    source_model = AdaptiveGaussianObjectWorldModel(source_config)
    source_model.load_state_dict(state["model"], strict=True)

    target_config = replace(
        source_config,
        flow_endpoint_prediction=True,
        prior_effect_weight=args.prior_effect_weight,
        correlated_flow_source=False,
        flow_source_scale=1.0,
        multi_query_prior_context=args.multi_query_prior_context,
        flow_source_components=args.flow_source_components,
        flow_lift_scale=args.flow_lift_scale,
        flow_responsibility_floor=args.flow_responsibility_floor,
        flow_source_fit_weight=args.flow_source_fit_weight,
        flow_source_min_scale=args.flow_source_min_scale,
        flow_balanced_source_assignment=args.balanced_source_assignment,
        flow_assignment_temperature=args.flow_assignment_temperature,
        mode_set_prior=args.mode_set_prior,
        mode_set_geometry_weight=args.mode_set_geometry_weight,
        mode_set_ordered_assignment=args.mode_set_ordered_assignment,
    )
    torch.manual_seed(args.initialization_seed)
    target_model = AdaptiveGaussianObjectWorldModel(target_config)
    source_tensors = source_model.state_dict()
    target_tensors = target_model.state_dict()
    copied = []
    reinitialized = []
    for name, target_tensor in target_tensors.items():
        if name.startswith("latent_actions.prior."):
            reinitialized.append(name)
            continue
        source_tensor = source_tensors.get(name)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            raise ValueError(f"non-prior tensor changed: {name}")
        target_tensors[name] = source_tensor.clone()
        copied.append(name)
    target_model.load_state_dict(target_tensors, strict=True)
    non_prior_exact = all(
        torch.equal(target_tensors[name], source_tensors[name])
        for name in copied
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
        "flow_source_components": target_config.flow_source_components,
        "flow_lift_scale": target_config.flow_lift_scale,
        "flow_responsibility_floor": (
            target_config.flow_responsibility_floor
        ),
        "flow_source_fit_weight": target_config.flow_source_fit_weight,
        "flow_source_min_scale": target_config.flow_source_min_scale,
        "flow_balanced_source_assignment": (
            target_config.flow_balanced_source_assignment
        ),
        "flow_assignment_temperature": (
            target_config.flow_assignment_temperature
        ),
        "flow_endpoint_prediction": target_config.flow_endpoint_prediction,
        "prior_effect_weight": target_config.prior_effect_weight,
        "initialization_seed": args.initialization_seed,
        "mode_set_prior": target_config.mode_set_prior,
        "mode_set_geometry_weight": target_config.mode_set_geometry_weight,
        "mode_set_ordered_assignment": (
            target_config.mode_set_ordered_assignment
        ),
        "total_parameter_count": sum(
            parameter.numel() for parameter in target_model.parameters()
        ),
        "prior_parameter_count": sum(
            parameter.numel()
            for parameter in target_model.latent_actions.prior.parameters()
        ),
        "copied_non_prior_tensor_count": len(copied),
        "reinitialized_prior_tensor_count": len(reinitialized),
        "reinitialized_prior_parameter_count": sum(
            target_tensors[name].numel() for name in reinitialized
        ),
        "non_prior_exact": non_prior_exact,
        "reinitialized_prior_tensors": reinitialized,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
