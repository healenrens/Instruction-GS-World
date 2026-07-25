"""Diagnose source-component specialization without using future at inference."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_groups", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--semantic_branch_strength", type=float, default=2.0)
    parser.add_argument("--evaluation_seed", type=int, default=9107)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.evaluation_seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    batch = make_synthetic_batch(
        feature_dim=config.feature_dim,
        batch_size=args.eval_groups * 3,
        history_frames=3,
        future_steps=args.future_steps,
        grid_size=args.grid_size,
        device=device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
        max_objects=config.object_slots,
        ambiguous_fraction=0.5,
        balanced_ambiguity=True,
        semantic_branch_strength=args.semantic_branch_strength,
    )
    with torch.no_grad():
        history = model.encode_history(batch)
        _, target = model.encode_targets(batch)
        history_scale = signed_gap_scale(
            batch["history_times"],
            config.gap_reference,
        )
        future_scale = signed_gap_scale(
            batch["future_times"],
            config.gap_reference,
        )
        posterior = model.latent_actions.posterior(
            history["slots"],
            history["activity"],
            target["slots"],
            target["activity"],
            future_scale,
            history["center"],
            target["center"],
        )
        context = model.prior_context(
            history,
            future_scale,
            history_scale,
        )
        prior = model.latent_actions.prior
        logits, mean, scale = prior._source_distribution(context)
        responsibility, source_fit = prior._responsibilities(
            posterior,
            logits,
            mean,
            scale,
            batch["group_id"],
        )

    probability = F.softmax(logits, dim=-1)
    probability_entropy = -(
        probability * probability.clamp_min(1e-8).log()
    ).sum(dim=-1)
    responsibility_entropy = -(
        responsibility * responsibility.clamp_min(1e-8).log()
    ).sum(dim=-1)
    assignment = responsibility.argmax(dim=-1)
    component_count = torch.bincount(
        assignment,
        minlength=config.flow_source_components,
    )
    group_assignment = assignment.reshape(args.eval_groups, 3)
    group_unique = torch.tensor(
        [row.unique().numel() for row in group_assignment],
        device=device,
    )
    ambiguous = batch["ambiguity"][::3].bool()
    ambiguous_unique = group_unique[ambiguous]
    normalized_mean = F.normalize(mean.flatten(2), dim=-1)
    mean_distance = (
        normalized_mean[:, :, None] - normalized_mean[:, None, :]
    ).square().mean(dim=-1)
    off_diagonal = ~torch.eye(
        config.flow_source_components,
        device=device,
        dtype=torch.bool,
    )[None]
    target_error = (
        F.normalize(mean, dim=-1)
        - F.normalize(posterior[:, None], dim=-1)
    ).square().mean(dim=(2, 3, 4))
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "evaluation_seed": args.evaluation_seed,
        "eval_groups": args.eval_groups,
        "flow_source_components": config.flow_source_components,
        "flow_lift_scale": config.flow_lift_scale,
        "flow_balanced_source_assignment": (
            config.flow_balanced_source_assignment
        ),
        "source_fit": float(source_fit),
        "probability_mean": probability.mean(dim=0).tolist(),
        "probability_entropy_mean": float(probability_entropy.mean()),
        "probability_max_mean": float(probability.max(dim=-1).values.mean()),
        "responsibility_mean": responsibility.mean(dim=0).tolist(),
        "responsibility_entropy_mean": float(responsibility_entropy.mean()),
        "component_assignment_count": component_count.tolist(),
        "ambiguous_unique_component_mean": float(ambiguous_unique.float().mean()),
        "ambiguous_all_components_fraction": float(
            (ambiguous_unique == config.flow_source_components).float().mean()
        ),
        "source_mean_pairwise_normalized_mse": float(
            mean_distance.masked_select(
                off_diagonal.expand_as(mean_distance)
            ).mean()
        ),
        "source_scale_mean": float(scale.mean()),
        "source_scale_min": float(scale.min()),
        "source_scale_max": float(scale.max()),
        "source_mean_nearest_target_mse": float(
            target_error.min(dim=-1).values.mean()
        ),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
