"""Remote CPU contract for canonical action-projection-only tuning."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.model import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.training_modes import (  # noqa: E402
    configure_posterior_dynamics_gate,
    update_target_for_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(97)
    config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=6,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        bounded_residual_action=True,
        action_query_modulation=True,
        prior_query_residual=True,
        rgb_supervision=True,
    )
    model = AdaptiveGaussianObjectWorldModel(config)
    with torch.no_grad():
        model.target_allocator.queries.add_(1.0)
    target_before = model.target_allocator.queries.detach().clone()
    update_target_for_training(model, posterior_dynamics_gate=True)
    if not torch.equal(model.target_allocator.queries, target_before):
        raise AssertionError("posterior gate updated the EMA target")
    configure_posterior_dynamics_gate(model, "action_projection")
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected = ["dynamics.action_input.weight"]
    if trainable != expected:
        raise AssertionError(f"unexpected trainable parameters: {trainable}")

    dynamics = model.dynamics.eval()
    history = torch.randn(2, 1, 4, config.object_dim)
    activity = torch.ones(2, 1, 4)
    history_scale = torch.zeros(2, 1)
    future_scale = torch.ones(2, 1)
    centers = torch.randn(2, 1, 4, 2)
    zero_action = torch.zeros(2, 1, 4, 6)
    action = torch.randn_like(zero_action)
    with torch.no_grad():
        zero_before = dynamics(
            history,
            activity,
            history_scale,
            future_scale,
            zero_action,
            history_centers=centers,
        ).future_slots
        action_before = dynamics(
            history,
            activity,
            history_scale,
            future_scale,
            action,
            history_centers=centers,
        ).future_slots

    optimizer = torch.optim.SGD(
        [model.dynamics.action_input.weight],
        lr=0.1,
    )
    prediction = dynamics(
        history,
        activity,
        history_scale,
        future_scale,
        action,
        history_centers=centers,
    ).future_slots
    prediction.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        zero_after = dynamics(
            history,
            activity,
            history_scale,
            future_scale,
            zero_action,
            history_centers=centers,
        ).future_slots
        action_after = dynamics(
            history,
            activity,
            history_scale,
            future_scale,
            action,
            history_centers=centers,
        ).future_slots
    zero_error = float((zero_after - zero_before).abs().max())
    action_change = float((action_after - action_before).abs().max())
    if zero_error != 0.0:
        raise AssertionError("action projection update changed the zero-action path")
    if action_change == 0.0:
        raise AssertionError("action projection update did not change action output")

    report = {
        "status": "ok",
        "trainable_parameters": trainable,
        "trainable_count": model.dynamics.action_input.weight.numel(),
        "target_teacher_unchanged": True,
        "zero_path_max_error": zero_error,
        "action_path_max_change": action_change,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
