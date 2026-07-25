"""Jointly expose Dynamics to deploy-time mode-set latent actions."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.dirname(__file__))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.joint_mode_set import (  # noqa: E402
    matched_mode_set_dynamics_loss,
)
from igsw.adaptive_gaussian_wm.losses import (  # noqa: E402
    adaptive_world_model_loss,
)
from validate_adaptive_gaussian_architecture import evaluate_model  # noqa: E402


def make_batch(
    model,
    batch_size: int,
    history_frames: int,
    future_steps: int,
    grid_size: int,
    device: torch.device,
    semantic_branch_strength: float,
    ambiguous_only: bool,
) -> dict[str, torch.Tensor]:
    return make_synthetic_batch(
        model.config.feature_dim,
        batch_size,
        history_frames,
        future_steps,
        grid_size,
        device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
        max_objects=model.config.object_slots,
        ambiguous_fraction=1.0 if ambiguous_only else 0.5,
        balanced_ambiguity=not ambiguous_only,
        semantic_branch_strength=semantic_branch_strength,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_in", required=True)
    parser.add_argument("--checkpoint_out", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--eval_groups", type=int, default=64)
    parser.add_argument("--prior_samples", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--joint_weight", type=float, default=1.0)
    parser.add_argument(
        "--assignment_strategy",
        choices=("ordered", "exact_effect", "posterior_code"),
        default="exact_effect",
    )
    parser.add_argument("--action_fit_weight", type=float, default=0.01)
    parser.add_argument(
        "--canonical_action_fit_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--coverage_margin_weight", type=float, default=0.0)
    parser.add_argument("--posterior_code_weight", type=float, default=0.1)
    parser.add_argument("--freeze_posterior", action="store_true")
    parser.add_argument("--semantic_branch_strength", type=float, default=2.0)
    parser.add_argument("--single_history_probability", type=float, default=0.25)
    parser.add_argument("--ambiguous_only", action="store_true")
    parser.add_argument("--evaluation_seed", type=int, default=9107)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch_size,
        args.future_steps,
        args.eval_groups,
        args.prior_samples,
    ) <= 0:
        raise ValueError(
            "steps, batch_size, future_steps, eval_groups, and "
            "prior_samples must be positive"
        )
    if args.batch_size % 3:
        raise ValueError("batch_size must be divisible by three")
    if args.joint_weight <= 0.0:
        raise ValueError("joint_weight must be positive")
    if args.action_fit_weight < 0.0:
        raise ValueError("action_fit_weight must be non-negative")
    if args.canonical_action_fit_weight < 0.0:
        raise ValueError("canonical_action_fit_weight must be non-negative")
    if args.coverage_margin_weight < 0.0:
        raise ValueError("coverage_margin_weight must be non-negative")
    if args.posterior_code_weight < 0.0:
        raise ValueError("posterior_code_weight must be non-negative")
    if not 0.0 <= args.single_history_probability <= 1.0:
        raise ValueError("single_history_probability must be in [0, 1]")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint_in, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    if not config.mode_set_prior:
        raise ValueError("checkpoint must use a mode-set prior")
    if (
        args.assignment_strategy == "ordered"
        and not config.mode_set_ordered_assignment
    ):
        raise ValueError("ordered matching requires an ordered mode-set prior")
    if (
        args.assignment_strategy == "posterior_code"
        and not config.mode_set_global_codebook
    ):
        raise ValueError("posterior code assignment requires a global codebook")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    for module in (model.allocator, model.object_aggregator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if args.freeze_posterior:
        for parameter in model.latent_actions.posterior.parameters():
            parameter.requires_grad_(False)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    weights = AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=0.5,
        allocator=0.2,
        slot=0.2,
        action=0.8,
    )
    trace = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        history_frames = (
            1
            if float(torch.rand(())) < args.single_history_probability
            else 3
        )
        batch = make_batch(
            model,
            args.batch_size,
            history_frames,
            args.future_steps,
            args.grid_size,
            device,
            args.semantic_branch_strength,
            args.ambiguous_only,
        )
        output = model(batch)
        posterior_loss, posterior_parts = adaptive_world_model_loss(
            model,
            batch,
            output,
            weights,
        )
        joint_loss, joint_parts = matched_mode_set_dynamics_loss(
            model,
            batch,
            output,
            args.assignment_strategy,
            args.action_fit_weight,
            args.canonical_action_fit_weight,
            args.coverage_margin_weight,
            args.posterior_code_weight,
        )
        loss = posterior_loss + args.joint_weight * joint_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        model.update_target()
        if step == 1 or step % max(args.steps // 10, 1) == 0:
            record = {
                "step": step,
                "total": float(loss.detach()),
                "posterior": float(posterior_loss.detach()),
                "joint": float(joint_loss.detach()),
                "posterior_future": float(
                    posterior_parts["future"].detach()
                ),
                **{
                    f"joint_{name}": float(value.detach())
                    for name, value in joint_parts.items()
                },
                "gradient_norm": float(gradient_norm),
            }
            trace.append(record)
            print(json.dumps({"training": record}, sort_keys=True), flush=True)
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
    checkpoint_out = os.path.abspath(args.checkpoint_out)
    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
    torch.save(
        {
            "variant": state["variant"],
            "seed": state["seed"],
            "config": config.to_dict(),
            "model": model.state_dict(),
            "joint_finetuned_from": os.path.abspath(args.checkpoint_in),
        },
        checkpoint_out,
    )
    report = {
        "status": "ok",
        "checkpoint_in": os.path.abspath(args.checkpoint_in),
        "checkpoint_out": checkpoint_out,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "joint_weight": args.joint_weight,
        "assignment_strategy": args.assignment_strategy,
        "action_fit_weight": args.action_fit_weight,
        "canonical_action_fit_weight": args.canonical_action_fit_weight,
        "coverage_margin_weight": args.coverage_margin_weight,
        "posterior_code_weight": args.posterior_code_weight,
        "freeze_posterior": args.freeze_posterior,
        "ambiguous_only": args.ambiguous_only,
        "evaluation_seed": args.evaluation_seed,
        "elapsed_seconds": time.time() - started,
        "training_trace": trace,
        "metrics": metrics,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
