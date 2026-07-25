"""Diagnose whether causal residual magnitude can gate reliable predictions."""
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

SCALES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
TOP_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)


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


def top_fraction_gate(
    magnitude: torch.Tensor,
    valid: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    if magnitude.shape != valid.shape:
        raise ValueError("magnitude and valid mask shapes differ")
    gate = torch.zeros_like(valid)
    for batch_index in range(magnitude.shape[0]):
        for time_index in range(magnitude.shape[1]):
            sample_valid = valid[batch_index, time_index]
            values = magnitude[batch_index, time_index][sample_valid]
            count = max(1, math.ceil(fraction * len(values)))
            threshold = values.topk(count).values[-1]
            gate[batch_index, time_index] = (
                magnitude[batch_index, time_index] >= threshold
            ) & sample_valid
    return gate


def summarize(
    values: dict[str, torch.Tensor],
    baseline: torch.Tensor,
) -> dict[str, dict[str, float | bool]]:
    report = {}
    for name, value in values.items():
        improvement = baseline - value
        mean = float(improvement.mean())
        standard_error = float(
            improvement.std(unbiased=False) / math.sqrt(len(improvement))
        )
        report[name] = {
            "mean": float(value.mean()),
            "absolute_improvement_over_copy": mean,
            "relative_improvement_over_copy": float(
                mean / baseline.mean().clamp_min(1e-8)
            ),
            "paired_standard_error": standard_error,
            "positive_2se_margin": mean > 2.0 * standard_error,
        }
    return report


@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> dict:
    feature_scale = {str(value): [] for value in SCALES}
    rgb_scale = {str(value): [] for value in SCALES}
    feature_top = {str(value): [] for value in TOP_FRACTIONS}
    rgb_top = {str(value): [] for value in TOP_FRACTIONS}
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        object_mask = torch.zeros(
            batch["history_features"].shape[0],
            1,
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=object_mask)
        current_feature = batch["history_features"][:, -1:].float()
        future_feature = output["rendered_future_features"].float()
        feature_delta = future_feature - current_feature
        current_rgb = batch["history_rgb"][:, -1:].float() / 255.0
        future_rgb = output["rendered_future_rgb"].float()
        rgb_delta = future_rgb - current_rgb
        for scale in SCALES:
            name = str(scale)
            feature_scale[name].append(
                masked_feature_mse(
                    current_feature + scale * feature_delta,
                    batch["future_features"].float(),
                    batch["future_valid"],
                ).cpu()
            )
            rgb_scale[name].append(
                rgb_distance(
                    current_rgb + scale * rgb_delta,
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
        feature_magnitude = feature_delta.square().mean(dim=-1).sqrt()
        rgb_magnitude = rgb_delta.abs().mean(dim=2)
        for fraction in TOP_FRACTIONS:
            name = str(fraction)
            feature_gate = top_fraction_gate(
                feature_magnitude,
                batch["future_valid"],
                fraction,
            )
            rgb_gate = top_fraction_gate(
                rgb_magnitude,
                batch["future_rgb_valid"],
                fraction,
            )
            feature_top[name].append(
                masked_feature_mse(
                    current_feature + feature_delta * feature_gate[..., None],
                    batch["future_features"].float(),
                    batch["future_valid"],
                ).cpu()
            )
            rgb_top[name].append(
                rgb_distance(
                    current_rgb + rgb_delta * rgb_gate[:, :, None],
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
    feature_scale = {
        name: torch.cat(chunks) for name, chunks in feature_scale.items()
    }
    rgb_scale = {name: torch.cat(chunks) for name, chunks in rgb_scale.items()}
    feature_top = {
        name: torch.cat(chunks) for name, chunks in feature_top.items()
    }
    rgb_top = {name: torch.cat(chunks) for name, chunks in rgb_top.items()}
    return {
        "samples": len(rgb_scale["0.0"]),
        "feature_scale": summarize(feature_scale, feature_scale["0.0"]),
        "rgb_scale": summarize(rgb_scale, rgb_scale["0.0"]),
        "feature_top_fraction": summarize(
            feature_top,
            feature_scale["0.0"],
        ),
        "rgb_top_fraction": summarize(rgb_top, rgb_scale["0.0"]),
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
    parser.add_argument("--max_items", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
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
    device = torch.device("cuda")
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
