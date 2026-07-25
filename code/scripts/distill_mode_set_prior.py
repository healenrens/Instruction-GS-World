"""Distill an unordered history-only mode set from a frozen world model."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.dirname(__file__))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.mode_set_distillation import (  # noqa: E402
    frozen_teacher_mode_set_loss,
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
        ambiguous_fraction=0.5,
        balanced_ambiguity=True,
        semantic_branch_strength=semantic_branch_strength,
    )


def unique_parameters(
    parameters: list[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    result = []
    seen = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_in", required=True)
    parser.add_argument("--checkpoint_out", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--eval_groups", type=int, default=64)
    parser.add_argument("--prior_samples", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--latent_weight", type=float, default=1.0)
    parser.add_argument("--canonical_weight", type=float, default=0.0)
    parser.add_argument("--effect_weight", type=float, default=1.0)
    parser.add_argument("--unique_mode_cardinality", action="store_true")
    parser.add_argument("--normalize_prototypes", action="store_true")
    parser.add_argument("--transformer_prior", action="store_true")
    parser.add_argument("--semantic_branch_strength", type=float, default=2.0)
    parser.add_argument("--single_history_probability", type=float, default=0.25)
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
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if min(
        args.latent_weight,
        args.canonical_weight,
        args.effect_weight,
    ) < 0.0:
        raise ValueError("distillation weights must be non-negative")
    if (
        args.latent_weight == 0.0
        and args.canonical_weight == 0.0
        and args.effect_weight == 0.0
    ):
        raise ValueError("at least one distillation weight must be positive")
    if not 0.0 <= args.single_history_probability <= 1.0:
        raise ValueError("single_history_probability must be in [0, 1]")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint_in, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    if args.normalize_prototypes:
        config = replace(config, mode_set_normalize_prototypes=True)
    if args.transformer_prior:
        config = replace(config, mode_set_transformer=True)
    if not config.mode_set_prior:
        raise ValueError("checkpoint must use a mode-set prior")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    source_transformer = state["config"].get("mode_set_transformer", False)
    if args.transformer_prior and not source_transformer:
        incompatible = model.load_state_dict(state["model"], strict=False)
        expected_missing = {
            name
            for name in model.state_dict()
            if name.startswith("latent_actions.prior.blocks.")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise ValueError(
                "unexpected missing keys while adding mode-set transformer: "
                + json.dumps(sorted(incompatible.missing_keys))
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                "unexpected checkpoint keys while adding mode-set transformer: "
                + json.dumps(sorted(incompatible.unexpected_keys))
            )
    else:
        model.load_state_dict(state["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = unique_parameters(
        [
            *model.latent_actions.prior.parameters(),
            *model.latent_actions.prior_condition_parameters(),
        ]
    )
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    trace = []
    started = time.time()
    model.eval()
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
        )
        history_mask = torch.zeros(
            args.batch_size,
            history_frames,
            config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with torch.no_grad():
            teacher = model(batch, history_mask=history_mask)
        loss, parts = frozen_teacher_mode_set_loss(
            model,
            batch,
            teacher,
            args.latent_weight,
            args.canonical_weight,
            args.effect_weight,
            args.unique_mode_cardinality,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if step == 1 or step % max(args.steps // 10, 1) == 0:
            record = {
                "step": step,
                **{
                    name: float(value.detach())
                    for name, value in parts.items()
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
            "mode_set_distilled_from": os.path.abspath(args.checkpoint_in),
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
        "latent_weight": args.latent_weight,
        "canonical_weight": args.canonical_weight,
        "effect_weight": args.effect_weight,
        "unique_mode_cardinality": args.unique_mode_cardinality,
        "normalize_prototypes": args.normalize_prototypes,
        "transformer_prior": args.transformer_prior,
        "evaluation_seed": args.evaluation_seed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
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
