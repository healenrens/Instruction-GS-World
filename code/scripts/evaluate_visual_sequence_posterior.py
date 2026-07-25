"""Evaluate posterior-conditioned Dynamics on language-free visual sequences."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "scripts"))

from evaluate_posterior_dynamics_gate import evaluate  # noqa: E402
from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    validate_data_model_contract,
)


def saved_or_override(saved: dict, name: str, override):
    return override if override not in (0, "") else saved[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=144)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--history_frames", type=int, default=0)
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--sequence_anchors", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items < args.batch:
        raise ValueError("posterior evaluation requires at least two samples")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    saved = checkpoint.get("args", {})
    if saved.get("data_format") != "sequence":
        raise ValueError("checkpoint was not trained with visual sequences")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.condition_dim != 0 or not config.rgb_supervision:
        raise ValueError("visual sequence evaluation requires RGB and no language")
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=saved_or_override(
            saved,
            "history_frames",
            args.history_frames,
        ),
        future_frames=saved_or_override(
            saved,
            "future_frames",
            args.future_frames,
        ),
        anchors=saved_or_override(
            saved,
            "sequence_anchors",
            args.sequence_anchors,
        ),
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, False, True)
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
        "data": os.path.abspath(args.data),
        "split": args.split,
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "posterior_dynamics_gate": bool(
            saved.get("posterior_dynamics_gate", False)
        ),
        "evaluation": evaluate(model, loader, device, args.amp),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
