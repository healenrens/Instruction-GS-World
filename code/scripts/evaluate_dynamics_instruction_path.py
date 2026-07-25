"""Isolate the direct instruction path in Dynamics with fixed posterior action."""
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
from igsw.adaptive_gaussian_wm.counterfactuals import render_state  # noqa: E402
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    rgb_reconstruction_loss,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


VARIANTS = ("correct", "none", "wrong")


def _sample_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.to(value.dtype)
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return (value * expanded).flatten(1).sum(dim=1) / expanded.expand_as(
        value
    ).flatten(1).sum(dim=1).clamp_min(1e-6)


def _paired(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    improvement = reference - candidate
    mean = improvement.mean()
    standard_error = improvement.std(unbiased=False) / math.sqrt(len(improvement))
    return {
        "absolute_improvement": float(mean),
        "relative_improvement": float(
            mean / reference.mean().clamp_min(1e-8)
        ),
        "candidate_win_fraction": float((improvement > 0.0).float().mean()),
        "paired_standard_error": float(standard_error),
        "positive_2se_margin": bool(mean > 2.0 * standard_error),
    }


def _predict(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    condition: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    )
    prediction = model.dynamics(
        output["online_history_slots"],
        history_activity,
        signed_gap_scale(batch["history_times"], model.config.gap_reference),
        signed_gap_scale(batch["future_times"], model.config.gap_reference),
        output["posterior_actions"],
        output["history_mask"],
        output["online_history_centers"],
        condition,
    )
    centers = (
        prediction.future_centers
        if prediction.future_centers is not None
        else model.object_aggregator.decode_center(prediction.future_slots)
    )
    return prediction.future_slots, centers


@torch.no_grad()
def evaluate(model, loader, device, amp, condition_bank) -> dict:
    values = {
        metric: {variant: [] for variant in VARIANTS}
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    slot_rms = {"none": [], "wrong": []}
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        history_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=history_mask)
            correct_condition = output["language_condition"]
            if correct_condition is None:
                raise ValueError("instruction-path evaluation requires condition")
            wrong_index = (
                batch["condition_index"].detach().cpu() + 1
            ) % condition_bank.shape[0]
            wrong_condition = model.language_condition(
                condition_bank[wrong_index].to(device, non_blocking=True)
            )
            states = {
                "correct": (
                    output["predicted_future_slots"],
                    output["predicted_future_centers"],
                ),
                "none": _predict(model, batch, output, None),
                "wrong": _predict(model, batch, output, wrong_condition),
            }
            rendered = {
                "correct": (
                    output["rendered_future_features"],
                    output["rendered_future_rgb"],
                ),
                "none": render_state(model, batch, output, *states["none"]),
                "wrong": render_state(model, batch, output, *states["wrong"]),
            }
        for variant in VARIANTS:
            features, rgb = rendered[variant]
            if rgb is None:
                raise ValueError("instruction-path evaluation requires RGB")
            values["feature_mse"][variant].append(
                _sample_mean(
                    (features.float() - batch["future_features"].float())
                    .square()
                    .mean(dim=-1),
                    batch["future_valid"],
                ).cpu()
            )
            values["latent_mse"][variant].append(
                _sample_mean(
                    (
                        F.normalize(states[variant][0].float(), dim=-1)
                        - F.normalize(
                            output["target_future_slots"].float(),
                            dim=-1,
                        )
                    ).square(),
                    output["target_future_activity"],
                ).cpu()
            )
            values["rgb_distance"][variant].append(
                torch.stack(
                    [
                        rgb_reconstruction_loss(
                            rgb[index : index + 1].float(),
                            batch["future_rgb"][index : index + 1],
                            batch["future_rgb_valid"][index : index + 1],
                            model.config.rgb_ssim_weight,
                        )[0]
                        for index in range(len(rgb))
                    ]
                ).cpu()
            )
        for variant in ("none", "wrong"):
            slot_rms[variant].append(
                (states[variant][0].float() - states["correct"][0].float())
                .square()
                .mean(dim=(1, 2, 3))
                .sqrt()
                .cpu()
            )
    tensors = {
        metric: {
            variant: torch.cat(chunks)
            for variant, chunks in variants.items()
        }
        for metric, variants in values.items()
    }
    return {
        "samples": len(next(iter(tensors.values()))["correct"]),
        "mean": {
            metric: {
                variant: float(value.mean())
                for variant, value in variants.items()
            }
            for metric, variants in tensors.items()
        },
        "correct_vs_none": {
            metric: _paired(variants["correct"], variants["none"])
            for metric, variants in tensors.items()
        },
        "correct_vs_wrong": {
            metric: _paired(variants["correct"], variants["wrong"])
            for metric, variants in tensors.items()
        },
        "slot_rms_from_correct": {
            variant: float(torch.cat(chunks).mean())
            for variant, chunks in slot_rms.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--max_items", type=int, default=64)
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
    if dataset.condition_store is None:
        raise ValueError("instruction-path evaluation requires condition cache")
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
    result = evaluate(
        model,
        loader,
        device,
        args.amp,
        dataset.condition_store.features,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "action_source": "fixed_future_conditioned_posterior",
        "result": result,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
