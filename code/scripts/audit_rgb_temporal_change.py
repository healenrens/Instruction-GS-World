"""Measure how much of each causal RGB pair actually changes over time."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)


def masked_sample_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(value.dtype)
    return (value * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def summarize(value: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(value.mean()),
        "median": float(value.median()),
        "p10": float(value.quantile(0.1)),
        "p90": float(value.quantile(0.9)),
    }


def audit(loader: DataLoader) -> dict:
    change_means = []
    copy_errors = []
    changed_errors = []
    fractions = {threshold: [] for threshold in (0.01, 0.03, 0.05)}
    for batch in loader:
        current = batch["history_rgb"][:, -1:].float() / 255.0
        future = batch["future_rgb"].float() / 255.0
        valid = batch["future_rgb_valid"]
        channel_change = (future - current).abs()
        change = channel_change.mean(dim=2)
        charbonnier = torch.sqrt(channel_change.square() + 1e-6).mean(dim=2)
        change_means.append(masked_sample_mean(change, valid))
        copy_errors.append(masked_sample_mean(charbonnier, valid))
        changed = valid & (change >= 0.03)
        changed_errors.append(masked_sample_mean(charbonnier, changed))
        for threshold in fractions:
            fractions[threshold].append(
                masked_sample_mean((change >= threshold).float(), valid)
            )
    change_mean = torch.cat(change_means)
    copy_error = torch.cat(copy_errors)
    changed_error = torch.cat(changed_errors)
    return {
        "samples": len(change_mean),
        "mean_absolute_rgb_change": summarize(change_mean),
        "raw_copy_charbonnier": summarize(copy_error),
        "changed_pixel_charbonnier_at_0.03": summarize(changed_error),
        "changed_pixel_fraction": {
            str(threshold): summarize(torch.cat(chunks))
            for threshold, chunks in fractions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb_short_side", type=int, default=256)
    parser.add_argument("--rgb_pad_multiple", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        args.split,
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=args.rgb_short_side,
        rgb_pad_multiple=args.rgb_pad_multiple,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
    )
    report = {
        "status": "ok",
        "split": args.split,
        "metrics": audit(loader),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
