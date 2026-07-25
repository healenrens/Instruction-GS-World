"""Evaluate one canonical activity exponent without changing model weights."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from evaluate_posterior_dynamics_gate import evaluate  # noqa: E402
from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    validate_data_model_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--max_items", type=int, default=144)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.power <= 0.0:
        raise ValueError("activity power must be positive")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if not config.canonical_activity_gate:
        raise ValueError("activity power sweep requires an activity-gated model")
    device = torch.device("cuda")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.latent_actions.posterior.canonical_activity_power = args.power
    model.eval()
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "canonical_activity_power": args.power,
        "splits": {},
    }
    for split in ("heldseed", "heldtask"):
        dataset = CausalPairFeatureDataset(
            args.data,
            args.dino,
            split,
            max_items=args.max_items,
            condition_cache=args.condition_cache,
            load_rgb=True,
            rgb_short_side=config.rgb_short_side,
            rgb_pad_multiple=config.rgb_pad_multiple,
        )
        validate_data_model_contract(config, dataset, True, True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        report["splits"][split] = evaluate(
            model,
            loader,
            device,
            args.amp,
        )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
