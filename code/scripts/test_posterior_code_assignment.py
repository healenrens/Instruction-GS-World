"""Server-GPU structural checks for posterior-aligned global mode codes."""
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
from igsw.adaptive_gaussian_wm.joint_mode_set import (  # noqa: E402
    matched_mode_set_dynamics_loss,
)
from igsw.adaptive_gaussian_wm.mode_set_prior import (  # noqa: E402
    ModeSetActionPrior,
)
from igsw.adaptive_gaussian_wm.set_matching import (  # noqa: E402
    exact_target_component_assignment,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _gradient_sum(
    model: AdaptiveGaussianObjectWorldModel,
    prefixes: tuple[str, ...],
) -> float:
    return float(
        sum(
            parameter.grad.abs().sum()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and any(name.startswith(prefix) for prefix in prefixes)
        )
    )


def _swap_future(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    swapped = dict(batch)
    order = torch.arange(
        batch["future_features"].shape[0] - 1,
        -1,
        -1,
        device=batch["future_features"].device,
    )
    for key in ("future_features", "future_labels"):
        swapped[key] = batch[key][order]
    return swapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=223)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    _require(config.mode_set_prior, "checkpoint does not use a mode-set prior")
    _require(
        config.mode_set_global_codebook,
        "checkpoint does not use a global mode codebook",
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    prior = model.latent_actions.prior
    _require(
        isinstance(prior, ModeSetActionPrior),
        "checkpoint prior type is not ModeSetActionPrior",
    )

    model.train()
    batch = make_synthetic_batch(
        feature_dim=config.feature_dim,
        batch_size=12,
        history_frames=3,
        future_steps=2,
        grid_size=8,
        device=device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=prior.components,
        max_objects=config.object_slots,
        ambiguous_fraction=1.0,
        balanced_ambiguity=False,
        semantic_branch_strength=2.0,
    )
    output = model(batch)
    loss, parts = matched_mode_set_dynamics_loss(
        model,
        batch,
        output,
        assignment_strategy="posterior_code",
        action_fit_weight=0.0,
        canonical_action_fit_weight=0.0,
        coverage_margin_weight=0.0,
        posterior_code_weight=1.0,
    )
    _require(bool(torch.isfinite(loss)), "joint loss is non-finite")
    model.zero_grad(set_to_none=True)
    loss.backward()

    code_assignment_gradient = _gradient_sum(
        model,
        ("latent_actions.prior.code_assignment.",),
    )
    codebook_gradient = _gradient_sum(
        model,
        (
            "latent_actions.prior.mode_handles",
            "latent_actions.prior.action_handles",
            "latent_actions.prior.blocks.",
            "latent_actions.prior.prototype.",
        ),
    )
    dynamics_gradient = _gradient_sum(model, ("dynamics.",))
    context_input_gradient = _gradient_sum(
        model,
        ("latent_actions.prior.context_input.",),
    )
    _require(code_assignment_gradient > 0.0, "assignment head has no gradient")
    _require(codebook_gradient > 0.0, "global codebook has no gradient")
    _require(dynamics_gradient > 0.0, "Dynamics has no gradient")
    _require(
        context_input_gradient == 0.0,
        "global prototypes depend on contextual prototype input",
    )

    groups = batch["group_id"].numel() // prior.components
    code_logits = prior.code_assignment_logits(
        output["posterior_actions"].detach()
    ).reshape(groups, prior.components, prior.components)
    assignment = exact_target_component_assignment(-code_logits)
    usage = torch.nn.functional.one_hot(
        assignment,
        num_classes=prior.components,
    ).float().mean(dim=(0, 1))
    usage_error = float(
        (
            usage - usage.new_full(usage.shape, 1.0 / prior.components)
        ).abs().max()
    )
    _require(usage_error < 1e-7, "code assignment is not balanced")

    model.eval()
    swapped = _swap_future(batch)
    with torch.no_grad():
        original_output = model(batch)
        swapped_output = model(swapped)
        context_difference = float(
            (
                original_output["prior_context"]
                - swapped_output["prior_context"]
            ).abs().max()
        )
        logits_a, prototypes_a = prior._distribution(
            original_output["prior_context"]
        )
        logits_b, prototypes_b = prior._distribution(
            original_output["prior_context"] + 0.5
        )
        prototype_context_difference = float(
            (prototypes_a - prototypes_b).abs().max()
        )
        logit_context_difference = float((logits_a - logits_b).abs().max())
        posterior_future_difference = float(
            (
                original_output["posterior_actions"]
                - swapped_output["posterior_actions"]
            ).abs().max()
        )
    _require(
        context_difference == 0.0,
        "future targets changed history-only prior context",
    )
    _require(
        prototype_context_difference == 0.0,
        "global code prototypes changed with history context",
    )
    _require(
        logit_context_difference > 1e-6,
        "history context does not change conditional mode logits",
    )
    _require(
        posterior_future_difference > 1e-6,
        "posterior did not respond to changed futures",
    )

    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "strict_checkpoint_load": True,
        "joint_loss": float(loss.detach()),
        "joint_parts": {
            name: float(value.detach()) for name, value in parts.items()
        },
        "code_assignment_gradient_sum": code_assignment_gradient,
        "codebook_gradient_sum": codebook_gradient,
        "dynamics_gradient_sum": dynamics_gradient,
        "context_input_gradient_sum": context_input_gradient,
        "assignment_usage": usage.tolist(),
        "assignment_usage_max_error": usage_error,
        "prior_context_future_swap_max_difference": context_difference,
        "prototype_context_shift_max_difference": (
            prototype_context_difference
        ),
        "mode_logit_context_shift_max_difference": logit_context_difference,
        "posterior_future_swap_max_difference": posterior_future_difference,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
