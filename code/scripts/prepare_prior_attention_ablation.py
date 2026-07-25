"""Create joint/factorized prior checkpoints with identical non-prior state."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)


PRIOR_PREFIXES = (
    "latent_actions.prior.",
    "latent_actions.prior_history_input.",
    "latent_actions.prior_gap_input.",
    "latent_actions.prior_slot_input.",
    "latent_actions.prior_center_input.",
    "latent_actions.prior_history_scale_input.",
    "latent_actions.prior_attention.",
    "latent_actions.prior_norm.",
    "latent_actions.prior_query",
)


def save_variant(
    base: dict,
    state: dict[str, torch.Tensor],
    config: AdaptiveGaussianWMConfig,
    variant: str,
    output: str,
    seed: int,
) -> None:
    result = dict(base)
    result.update(
        {
            "variant": variant,
            "config": config.to_dict(),
            "model": state,
            "prior_ablation": {
                "reset_seed": seed,
                "non_prior_state_identical": True,
                "only_difference": "joint_vs_length1_attention",
            },
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(result, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--joint_out", required=True)
    parser.add_argument("--factorized_out", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**base["config"])
    if not config.structured_action or not config.joint_flow:
        raise ValueError("base checkpoint must use structured joint action prior")
    torch.manual_seed(args.seed)
    fresh = AdaptiveGaussianObjectWorldModel(config)
    fresh_state = fresh.state_dict()
    state = {
        name: (
            fresh_state[name].clone()
            if name.startswith(PRIOR_PREFIXES)
            else value.clone()
        )
        for name, value in base["model"].items()
    }
    joint_config = replace(config, joint_flow=True)
    factorized_config = replace(config, joint_flow=False)
    joint = AdaptiveGaussianObjectWorldModel(joint_config)
    factorized = AdaptiveGaussianObjectWorldModel(factorized_config)
    joint.load_state_dict(state, strict=True)
    factorized.load_state_dict(state, strict=True)
    save_variant(
        base,
        state,
        joint_config,
        "full",
        args.joint_out,
        args.seed,
    )
    save_variant(
        base,
        state,
        factorized_config,
        "independent_prior",
        args.factorized_out,
        args.seed,
    )
    print(
        {
            "status": "ok",
            "joint_parameters": sum(
                parameter.numel()
                for parameter in joint.latent_actions.prior.parameters()
            ),
            "factorized_parameters": sum(
                parameter.numel()
                for parameter in factorized.latent_actions.prior.parameters()
            ),
        }
    )


if __name__ == "__main__":
    main()
