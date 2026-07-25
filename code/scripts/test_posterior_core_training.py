"""Remote CPU contract for long-run posterior-core training boundaries."""
from __future__ import annotations

import argparse
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
    configure_posterior_core_training,
    staged_loss_weights,
    update_target_for_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(113)
    model = AdaptiveGaussianObjectWorldModel(
        AdaptiveGaussianWMConfig.tiny(12)
    )
    with torch.no_grad():
        model.allocator.queries.add_(1.0)
        model.target_allocator.queries.zero_()
    target_before = model.target_allocator.queries.detach().clone()
    configure_posterior_core_training(model)
    update_target_for_training(model, posterior_dynamics_gate=False)
    target_after = model.target_allocator.queries.detach().clone()
    if torch.equal(target_before, target_after):
        raise AssertionError("posterior core did not update the EMA target")

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    forbidden = {
        name
        for name, _ in model.latent_actions.prior.named_parameters(
            prefix="latent_actions.prior"
        )
    }
    condition_ids = {
        id(parameter)
        for parameter in model.latent_actions.prior_condition_parameters()
    }
    forbidden.update(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in condition_ids
    )
    if trainable.intersection(forbidden):
        raise AssertionError("posterior core left blind Prior parameters trainable")
    required_prefixes = (
        "allocator.",
        "object_aggregator.",
        "latent_actions.posterior.",
        "dynamics.",
        "gaussian_readout.",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in trainable)
    ]
    if missing:
        raise AssertionError(f"posterior core froze required modules: {missing}")

    weights = staged_loss_weights(
        posterior_dynamics_gate=False,
        posterior_core_training=True,
    )
    if (
        weights.flow != 0.0
        or weights.allocator <= 0.0
        or weights.slot <= 0.0
        or weights.action_specificity <= 0.0
    ):
        raise AssertionError("posterior core loss boundary differs")
    report = {
        "status": "ok",
        "trainable_parameter_tensors": len(trainable),
        "frozen_prior_parameter_tensors": len(forbidden),
        "ema_target_updated": True,
        "loss_weights": vars(weights),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
