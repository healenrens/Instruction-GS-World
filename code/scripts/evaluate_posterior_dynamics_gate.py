"""Evaluate whether future-conditioned latent actions causally improve Dynamics."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.action_embedding import (  # noqa: E402
    effect_supervision_actions,
)
from igsw.adaptive_gaussian_wm.counterfactuals import (  # noqa: E402
    predict_shuffled_action,
    render_state,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.posterior_diagnostics import (  # noqa: E402
    posterior_effect_batch,
    summarize_posterior_effects,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    rgb_reconstruction_loss,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


PREDICTIONS = ("posterior", "zero_action", "shuffled_action", "copy")
REFERENCES = ("zero_action", "shuffled_action", "copy")


def masked_feature_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    weight = valid.to(error.dtype)
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def weighted_latent_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    error = (
        torch.nn.functional.normalize(prediction, dim=-1)
        - torch.nn.functional.normalize(target, dim=-1)
    ).square().mean(dim=-1)
    weight = activity.to(error.dtype)
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1e-6)


def rgb_distance(
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


def comparison(
    posterior: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    improvement = reference - posterior
    mean = float(improvement.mean())
    standard_error = float(
        improvement.std(unbiased=False) / math.sqrt(max(len(improvement), 1))
    )
    return {
        "absolute_improvement": mean,
        "relative_improvement": float(
            mean / reference.mean().clamp_min(1e-8)
        ),
        "posterior_win_fraction": float((improvement > 0.0).float().mean()),
        "paired_standard_error": standard_error,
        "positive_2se_margin": mean > 2.0 * standard_error,
    }


@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> dict:
    values = {
        metric: {name: [] for name in PREDICTIONS}
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    action_codes = []
    target_effects = []
    predicted_effects = []
    effect_weights = []
    center_effect_errors = []
    slot_action_effect = []
    feature_action_effect = []
    action_shuffle_distance = []
    slot_shuffle_effect = []
    feature_shuffle_effect = []
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    action_bank = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        object_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=object_mask)
        action_bank.append(output["posterior_actions"].float().cpu())
    action_bank = torch.cat(action_bank)
    if action_bank.shape[0] < 2:
        raise ValueError("shuffled-action evaluation requires at least two samples")
    shuffle_index = torch.arange(action_bank.shape[0]).roll(
        action_bank.shape[0] // 2
    )
    sample_offset = 0
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        batch_size = batch["history_features"].shape[0]
        shuffled_actions = action_bank[
            shuffle_index[sample_offset : sample_offset + batch_size]
        ].to(device, non_blocking=True)
        object_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=object_mask)
            zero_feature, zero_rgb = render_state(
                model,
                batch,
                output,
                output["zero_action_future_slots"],
                output["zero_action_future_centers"],
            )
            shuffled_slots, shuffled_centers = predict_shuffled_action(
                model,
                batch,
                output,
                shuffled_actions,
            )
            shuffled_feature, shuffled_rgb = render_state(
                model,
                batch,
                output,
                shuffled_slots,
                shuffled_centers,
            )
        sample_offset += batch_size
        future_count = batch["future_features"].shape[1]
        feature_predictions = {
            "posterior": output["rendered_future_features"],
            "zero_action": zero_feature,
            "shuffled_action": shuffled_feature,
            "copy": batch["history_features"][:, -1:].expand(
                -1, future_count, -1, -1
            ),
        }
        latent_predictions = {
            "posterior": output["predicted_future_slots"],
            "zero_action": output["zero_action_future_slots"],
            "shuffled_action": shuffled_slots,
            "copy": output["online_history_slots"][:, -1:].expand(
                -1, future_count, -1, -1
            ),
        }
        if (
            output["rendered_future_rgb"] is None
            or zero_rgb is None
            or shuffled_rgb is None
        ):
            raise ValueError("posterior Dynamics gate requires RGB supervision")
        rgb_predictions = {
            "posterior": output["rendered_future_rgb"],
            "zero_action": zero_rgb,
            "shuffled_action": shuffled_rgb,
            "copy": batch["history_rgb"][:, -1:].expand(
                -1, future_count, -1, -1, -1
            ).float()
            / 255.0,
        }
        for name in PREDICTIONS:
            values["feature_mse"][name].append(
                masked_feature_mse(
                    feature_predictions[name].float(),
                    batch["future_features"].float(),
                    batch["future_valid"],
                ).cpu()
            )
            values["latent_mse"][name].append(
                weighted_latent_mse(
                    latent_predictions[name].float(),
                    output["target_future_slots"].float(),
                    output["target_future_activity"],
                ).cpu()
            )
            values["rgb_distance"][name].append(
                rgb_distance(
                    rgb_predictions[name].float(),
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
        action_codes.append(output["posterior_actions"].float().cpu())
        target_effect, predicted_effect, effect_weight = posterior_effect_batch(
            model,
            output,
        )
        target_activity = output["target_future_activity"].float()
        target_effects.append(target_effect.cpu())
        predicted_effects.append(predicted_effect.cpu())
        effect_weights.append(effect_weight.cpu())
        if model.latent_actions.center_effect_head is not None:
            predicted_center_effect = model.latent_actions.predict_center_effect(
                effect_supervision_actions(
                    output["posterior_actions"],
                    model.config.canonical_action_dim,
                )
            ).float()
            target_center_effect = (
                output["target_future_centers"]
                - output["target_history_centers"][:, -1, None]
            ).float()
            center_error = (
                (predicted_center_effect - target_center_effect)
                .square()
                .mean(dim=-1)
            )
            center_effect_errors.append(
                (
                    center_error * target_activity
                ).flatten(1).sum(dim=1)
                / target_activity.flatten(1).sum(dim=1).clamp_min(1e-6)
            )
        slot_action_effect.append(
            (
                output["predicted_future_slots"]
                - output["zero_action_future_slots"]
            )
            .float()
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .cpu()
        )
        feature_action_effect.append(
            (output["rendered_future_features"] - zero_feature)
            .float()
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .cpu()
        )
        action_shuffle_distance.append(
            (output["posterior_actions"].float() - shuffled_actions.float())
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .cpu()
        )
        slot_shuffle_effect.append(
            (output["predicted_future_slots"].float() - shuffled_slots.float())
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .cpu()
        )
        feature_shuffle_effect.append(
            (output["rendered_future_features"].float() - shuffled_feature.float())
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .cpu()
        )
    if sample_offset != action_bank.shape[0]:
        raise RuntimeError("action bank and evaluation loader differ")
    tensors = {
        metric: {
            name: torch.cat(chunks)
            for name, chunks in predictions.items()
        }
        for metric, predictions in values.items()
    }
    comparisons = {
        metric: {
            reference: comparison(
                predictions["posterior"],
                predictions[reference],
            )
            for reference in REFERENCES
        }
        for metric, predictions in tensors.items()
    }
    all_comparisons = [
        item
        for metric in comparisons.values()
        for item in metric.values()
    ]
    actions = torch.cat(action_codes)
    effect_diagnostics = summarize_posterior_effects(
        target_effects,
        predicted_effects,
        effect_weights,
    )
    return {
        "samples": len(actions),
        "mean": {
            metric: {
                name: float(value.mean())
                for name, value in predictions.items()
            }
            for metric, predictions in tensors.items()
        },
        "posterior_comparison": comparisons,
        "action_diagnostics": {
            "posterior_action_std": float(
                actions.flatten(0, 2).std(dim=0, unbiased=False).mean()
            ),
            "posterior_action_norm": float(
                actions.flatten(-2).norm(dim=-1).mean()
            ),
            **effect_diagnostics,
            "center_effect_mse": (
                float(torch.cat(center_effect_errors).mean())
                if center_effect_errors
                else 0.0
            ),
            "posterior_vs_zero_slot_rms": float(
                torch.cat(slot_action_effect).mean()
            ),
            "posterior_vs_zero_feature_rms": float(
                torch.cat(feature_action_effect).mean()
            ),
            "posterior_vs_shuffled_action_rms": float(
                torch.cat(action_shuffle_distance).mean()
            ),
            "posterior_vs_shuffled_slot_rms": float(
                torch.cat(slot_shuffle_effect).mean()
            ),
            "posterior_vs_shuffled_feature_rms": float(
                torch.cat(feature_shuffle_effect).mean()
            ),
        },
        "gate": {
            "all_mean_improvements_positive": all(
                item["absolute_improvement"] > 0.0
                for item in all_comparisons
            ),
            "all_paired_margins_exceed_2se": all(
                item["positive_2se_margin"]
                for item in all_comparisons
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
    if not checkpoint.get("args", {}).get("posterior_dynamics_gate", False):
        raise ValueError("checkpoint was not trained in posterior Dynamics gate mode")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if not config.rgb_supervision or config.condition_dim <= 0:
        raise ValueError("posterior Dynamics gate requires RGB and condition features")
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
    expected_cache = checkpoint.get("args", {}).get("condition_feature_sha256", "")
    if expected_cache and expected_cache != dataset.condition_store.feature_sha256:
        raise ValueError("evaluation condition cache differs from checkpoint")
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
        "action_anchor": (
            "object_slot" if config.object_aligned_actions else "global"
        ),
        "action_dim": config.action_dim,
        "action_residual_dim": config.action_residual_dim,
        "canonical_center_gate": config.canonical_center_gate, "canonical_activity_gate": config.canonical_activity_gate, "canonical_activity_power": config.canonical_activity_power, "action_residual_gate": config.action_residual_gate, "action_residual_dropout": config.action_residual_dropout, "learned_semantic_action_basis": config.learned_semantic_action_basis, "rgb_semantic_action": config.rgb_semantic_action,
        "metrics": evaluate(model, loader, device, args.amp),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
