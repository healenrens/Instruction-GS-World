"""Evaluate slot identity and effect anchoring for continuous latent actions."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.counterfactuals import (  # noqa: E402
    predict_shuffled_action,
    render_state,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    rgb_reconstruction_loss,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


def _weighted_sample_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    expanded = weight.to(value.dtype)
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return (value * expanded).flatten(1).sum(dim=1) / expanded.expand_as(
        value
    ).flatten(1).sum(dim=1).clamp_min(1e-6)


def _feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return _weighted_sample_mean(
        (prediction - target).square().mean(dim=-1),
        valid,
    )


def _latent_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    return _weighted_sample_mean(
        (
            F.normalize(prediction, dim=-1)
            - F.normalize(target, dim=-1)
        ).square(),
        activity,
    )


def _rgb_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    return torch.stack(
        [
            rgb_reconstruction_loss(
                prediction[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                ssim_weight,
            )[0]
            for index in range(len(prediction))
        ]
    )


def _paired_improvement(
    matched: torch.Tensor,
    permuted: torch.Tensor,
) -> dict[str, float | bool]:
    improvement = permuted - matched
    standard_error = improvement.std(unbiased=False) / math.sqrt(
        max(len(improvement), 1)
    )
    mean = improvement.mean()
    return {
        "absolute_improvement": float(mean),
        "relative_improvement": float(
            mean / permuted.mean().clamp_min(1e-8)
        ),
        "matched_win_fraction": float((improvement > 0.0).float().mean()),
        "paired_standard_error": float(standard_error),
        "positive_2se_margin": bool(mean > 2.0 * standard_error),
    }


@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> dict:
    metrics = {
        name: {"matched": [], "permuted": []}
        for name in ("feature_mse", "latent_mse", "rgb_distance")
    }
    diagnostics = {
        name: []
        for name in (
            "center_anchor_max_error",
            "semantic_anchor_max_error",
            "matched_object_effect_mse",
            "permuted_object_effect_mse",
            "object_effect_cosine",
            "slot_permutation_rms",
            "feature_permutation_rms",
        )
    }
    action_codes = []
    effect_codes = []
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        batch_size = batch["history_features"].shape[0]
        history_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=history_mask)
            actions = output["posterior_actions"]
            permuted_actions = actions.roll(1, dims=-2)
            permuted_slots, permuted_centers = predict_shuffled_action(
                model,
                batch,
                output,
                permuted_actions,
            )
            permuted_features, permuted_rgb = render_state(
                model,
                batch,
                output,
                permuted_slots,
                permuted_centers,
            )
        if output["rendered_future_rgb"] is None or permuted_rgb is None:
            raise ValueError("object-slot anchor evaluation requires RGB")
        matched_predictions = (
            output["rendered_future_features"].float(),
            output["predicted_future_slots"].float(),
            output["rendered_future_rgb"].float(),
        )
        permuted_predictions = (
            permuted_features.float(),
            permuted_slots.float(),
            permuted_rgb.float(),
        )
        metric_functions = (
            lambda value: _feature_error(
                value,
                batch["future_features"].float(),
                batch["future_valid"],
            ),
            lambda value: _latent_error(
                value,
                output["target_future_slots"].float(),
                output["target_future_activity"],
            ),
            lambda value: _rgb_error(
                value,
                batch["future_rgb"],
                batch["future_rgb_valid"],
                model.config.rgb_ssim_weight,
            ),
        )
        for name, function, matched, permuted in zip(
            metrics,
            metric_functions,
            matched_predictions,
            permuted_predictions,
            strict=True,
        ):
            metrics[name]["matched"].append(function(matched).cpu())
            metrics[name]["permuted"].append(function(permuted).cpu())

        center_delta = (
            output["target_future_centers"]
            - output["online_history_centers"][:, -1, None]
        )
        expected_center = torch.tanh(
            torch.cat(
                (center_delta, center_delta.norm(dim=-1, keepdim=True)),
                dim=-1,
            )
            / 0.25
        )
        slot_delta = (
            output["target_future_slots"]
            - output["online_history_slots"][:, -1, None]
        )
        projection = model.latent_actions.posterior.semantic_projection.to(
            slot_delta.dtype
        )
        expected_semantic = torch.tanh(slot_delta @ projection / 0.25)
        diagnostics["center_anchor_max_error"].append(
            (actions[..., :3].float() - expected_center)
            .abs()
            .flatten(1)
            .max(dim=1).values.cpu()
        )
        diagnostics["semantic_anchor_max_error"].append(
            (actions[..., 3:6].float() - expected_semantic)
            .abs()
            .flatten(1)
            .max(dim=1).values.cpu()
        )
        predicted_effect = model.latent_actions.predict_object_effect(
            actions
        ).float()
        permuted_effect = model.latent_actions.predict_object_effect(
            permuted_actions
        ).float()
        activity = output["target_future_activity"].float()
        diagnostics["matched_object_effect_mse"].append(
            _weighted_sample_mean(
                (predicted_effect - slot_delta.float()).square(),
                activity,
            ).cpu()
        )
        diagnostics["permuted_object_effect_mse"].append(
            _weighted_sample_mean(
                (permuted_effect - slot_delta.float()).square(),
                activity,
            ).cpu()
        )
        diagnostics["object_effect_cosine"].append(
            _weighted_sample_mean(
                F.cosine_similarity(predicted_effect, slot_delta.float(), dim=-1),
                activity,
            ).cpu()
        )
        diagnostics["slot_permutation_rms"].append(
            (output["predicted_future_slots"].float() - permuted_slots.float())
            .square().mean(dim=(1, 2, 3)).sqrt().cpu()
        )
        diagnostics["feature_permutation_rms"].append(
            (
                output["rendered_future_features"].float()
                - permuted_features.float()
            ).square().mean(dim=(1, 2, 3)).sqrt().cpu()
        )
        action_codes.append(actions.float().flatten(0, 2).cpu())
        effect_codes.append(
            (slot_delta.float() * activity[..., None]).flatten(0, 2).cpu()
        )

    tensors = {
        metric: {
            variant: torch.cat(chunks)
            for variant, chunks in variants.items()
        }
        for metric, variants in metrics.items()
    }
    diagnostic_tensors = {
        name: torch.cat(chunks)
        for name, chunks in diagnostics.items()
    }
    comparisons = {
        name: _paired_improvement(
            variants["matched"],
            variants["permuted"],
        )
        for name, variants in tensors.items()
    }
    actions = F.normalize(torch.cat(action_codes), dim=-1)
    effects = F.normalize(torch.cat(effect_codes), dim=-1)
    relation_samples = min(256, actions.shape[0])
    relation_index = torch.linspace(
        0,
        actions.shape[0] - 1,
        relation_samples,
    ).round().long()
    actions = actions[relation_index]
    effects = effects[relation_index]
    action_relation = actions @ actions.transpose(0, 1)
    effect_relation = effects @ effects.transpose(0, 1)
    anchor_tolerance = 5e-3 if amp == "bf16" else 1e-5
    return {
        "samples": int(next(iter(tensors.values()))["matched"].shape[0]),
        "mean": {
            metric: {
                variant: float(value.mean())
                for variant, value in variants.items()
            }
            for metric, variants in tensors.items()
        },
        "matched_vs_object_permutation": comparisons,
        "diagnostics": {
            name: float(value.mean())
            for name, value in diagnostic_tensors.items()
        }
        | {
            "action_effect_relation_mse": float(
                (action_relation - effect_relation).square().mean()
            ),
            "action_effect_relation_samples": relation_samples,
            "center_anchor_global_max_error": float(
                diagnostic_tensors["center_anchor_max_error"].max()
            ),
            "semantic_anchor_global_max_error": float(
                diagnostic_tensors["semantic_anchor_max_error"].max()
            ),
            "anchor_tolerance": anchor_tolerance,
        },
        "gate": {
            "canonical_anchor_exact": bool(
                diagnostic_tensors["center_anchor_max_error"].max()
                < anchor_tolerance
                and diagnostic_tensors["semantic_anchor_max_error"].max()
                < anchor_tolerance
            ),
            "all_permutations_worse": all(
                item["absolute_improvement"] > 0.0
                for item in comparisons.values()
            ),
            "all_permutation_margins_exceed_2se": all(
                item["positive_2se_margin"]
                for item in comparisons.values()
            ),
            "effect_mapping_uses_slot_identity": bool(
                diagnostic_tensors["permuted_object_effect_mse"].mean()
                > diagnostic_tensors["matched_object_effect_mse"].mean()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=144)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items % args.batch:
        raise ValueError("max_items must be divisible by batch >= 2")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if not (
        config.object_aligned_actions
        and config.canonical_center_action
        and config.canonical_semantic_action
    ):
        raise ValueError("checkpoint does not use object-slot action anchors")
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        args.split,
        max_items=args.max_items,
        condition_cache=args.condition_cache,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, True, True)
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "action_dim": config.action_dim,
        "action_residual_dim": config.action_residual_dim,
        "action_residual_gate": config.action_residual_gate,
        "action_residual_dropout": config.action_residual_dropout,
        "learned_semantic_action_basis": config.learned_semantic_action_basis,
        "metrics": evaluate(model, loader, device, args.amp),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
