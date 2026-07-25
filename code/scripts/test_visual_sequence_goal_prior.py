"""Remote synthetic contract test for the explicit image-goal action Prior."""
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
from igsw.adaptive_gaussian_wm.goal_conditioning import (  # noqa: E402
    ObjectGoalConditioner,
)
from igsw.adaptive_gaussian_wm.goal_prior_contract import (  # noqa: E402
    causal_goal_prior_contract,
)
from igsw.adaptive_gaussian_wm.goal_prior_objective import (  # noqa: E402
    GoalPriorObjective,
)
from igsw.adaptive_gaussian_wm.synthetic import (  # noqa: E402
    make_synthetic_batch,
)


def _gradient_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.float().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    ) ** 0.5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("goal Prior contract test requires remote CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=6,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        action_query_modulation=True,
        prior_query_residual=True,
        condition_dim=0,
        rgb_supervision=False,
        rgb_semantic_action=False,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device).eval()
    conditioner = ObjectGoalConditioner(config).to(device)
    batch = make_synthetic_batch(
        config.feature_dim,
        batch_size=4,
        history_frames=3,
        future_steps=2,
        grid_size=8,
        device=device,
        paired_futures=False,
        irregular_gaps=True,
        max_objects=config.object_slots,
        semantic_branch_strength=0.2,
    )
    batch.update(
        goal_features=batch["future_features"][:, -1],
        goal_coordinates=batch["future_coordinates"][:, -1],
        goal_valid=batch["future_valid"][:, -1],
        goal_time=batch["future_times"][:, -1],
        goal_frame_index=torch.full(
            (4,),
            4,
            device=device,
            dtype=torch.long,
        ),
        goal_control_index=torch.arange(
            4,
            device=device,
            dtype=torch.long,
        ),
        sequence_index=torch.arange(
            4,
            device=device,
            dtype=torch.long,
        ),
    )

    with torch.no_grad():
        contract = causal_goal_prior_contract(model, conditioner, batch)
    _require(contract["gate"]["all_passed"], "causal goal contract failed")

    model.requires_grad_(False)
    for parameter in model.latent_actions.prior.parameters():
        parameter.requires_grad_(True)
    for parameter in model.latent_actions.prior_condition_parameters():
        parameter.requires_grad_(True)
    conditioner.requires_grad_(True)
    objective = GoalPriorObjective(
        model,
        conditioner,
        effect_weight=0.1,
        goal_anchor_weight=0.5,
        goal_rank_weight=0.5,
        goal_relative_margin=0.05,
        action_activity_floor=0.25,
    )
    parts = objective(batch)
    _require(bool(torch.isfinite(parts["loss"])), "objective is not finite")
    parts["loss"].backward()
    prior_gradient = _gradient_norm(model.latent_actions.prior)
    conditioner_gradient = _gradient_norm(conditioner)
    posterior_gradient = _gradient_norm(model.latent_actions.posterior)
    dynamics_gradient = _gradient_norm(model.dynamics)
    _require(prior_gradient > 0.0, "Prior received no gradient")
    _require(conditioner_gradient > 0.0, "goal conditioner received no gradient")
    _require(posterior_gradient == 0.0, "Posterior teacher received a gradient")
    _require(dynamics_gradient == 0.0, "frozen Dynamics received a gradient")

    report = {
        "status": "ok",
        "config": {
            "object_slots": config.object_slots,
            "action_tokens": config.action_tokens,
            "action_dim": config.action_dim,
            "language_condition": "off",
        },
        "causal_contract": contract,
        "loss": {
            name: float(value.detach())
            for name, value in parts.items()
        },
        "gradients": {
            "prior": prior_gradient,
            "goal_conditioner": conditioner_gradient,
            "posterior": posterior_gradient,
            "dynamics": dynamics_gradient,
        },
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
