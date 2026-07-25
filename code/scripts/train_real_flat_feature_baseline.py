"""Train and evaluate the non-object flat feature baseline on causal pairs."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.baselines import FlatFeaturePredictor  # noqa: E402
from igsw.adaptive_gaussian_wm.metrics import masked_feature_mse  # noqa: E402
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: FlatFeaturePredictor,
    dataset: CausalPairFeatureDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    prediction_errors = []
    copy_errors = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        prediction = model(
            batch["history_features"],
            batch["future_coordinates"],
            signed_gap_scale(batch["future_times"], 1.0),
        )
        prediction_errors.append(
            (prediction - batch["future_features"]).square().mean(dim=(1, 2, 3))
        )
        copy_errors.append(
            (
                batch["history_features"][:, -1, None]
                - batch["future_features"]
            )
            .square()
            .mean(dim=(1, 2, 3))
        )
    prediction_error = torch.cat(prediction_errors).mean()
    copy_error = torch.cat(copy_errors).mean()
    return {
        "samples": len(dataset),
        "feature_mse": float(prediction_error),
        "current_copy_feature_mse": float(copy_error),
        "improvement_vs_copy": float(
            (copy_error - prediction_error) / copy_error.clamp_min(1e-8)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max_eval_items", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError("steps and batch must be positive")
    torch.manual_seed(17)
    device = torch.device(args.device)
    train_dataset = CausalPairFeatureDataset(args.data, args.dino, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    model = FlatFeaturePredictor(train_dataset.feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    step = 0
    while step < args.steps:
        for cpu_batch in train_loader:
            batch = move_to_device(cpu_batch, device)
            prediction = model(
                batch["history_features"],
                batch["future_coordinates"],
                signed_gap_scale(batch["future_times"], 1.0),
            )
            loss = masked_feature_mse(
                prediction,
                batch["future_features"],
                batch["future_valid"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= args.steps:
                break
    metrics = {}
    for split in ("heldseed", "heldtask"):
        dataset = CausalPairFeatureDataset(
            args.data,
            args.dino,
            split,
            max_items=args.max_eval_items,
        )
        metrics[split] = evaluate(model, dataset, args.batch, device)
    report = {"status": "ok", "steps": args.steps, "metrics": metrics}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
