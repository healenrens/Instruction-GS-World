"""Evaluate posterior action effects in changed and static RGB regions."""
from __future__ import annotations

import argparse
import json
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
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.temporal_region_evaluation import (  # noqa: E402
    TemporalRegionConfig,
    evaluate_temporal_regions,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    validate_data_model_contract,
)


def saved_or_override(saved: dict, name: str, override):
    return override if override not in (0, "") else saved[name]


def _available_samples(dataset) -> int:
    full_length = getattr(dataset, "_full_length", None)
    return len(dataset) if full_length is None else int(full_length)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("train", "heldseed", "heldtask"), required=True)
    parser.add_argument("--max_items", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--history_frames", type=int, default=0)
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--sequence_anchors", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--blur_kernel", type=int, default=5)
    parser.add_argument("--blur_sigma", type=float, default=1.0)
    parser.add_argument("--low_floor", type=float, default=0.015)
    parser.add_argument("--high_floor", type=float, default=0.04)
    parser.add_argument("--low_mad_scale", type=float, default=2.0)
    parser.add_argument("--high_mad_scale", type=float, default=4.0)
    parser.add_argument("--hysteresis_steps", type=int, default=4)
    parser.add_argument("--change_dilation", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items < args.batch:
        raise ValueError("temporal-region evaluation requires at least two samples")
    output_path = os.path.abspath(args.output)
    if os.path.exists(output_path):
        raise FileExistsError(f"refusing to overwrite evaluation: {output_path}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_global_step = int(checkpoint.get("global_step", -1))
    checkpoint_phase = checkpoint.get("phase")
    if checkpoint_global_step < 1 or checkpoint_phase != "joint":
        raise ValueError("temporal-region evaluation requires a joint checkpoint")
    saved = checkpoint.get("args", {})
    if saved.get("data_format") != "sequence":
        raise ValueError("checkpoint was not trained with visual sequences")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.condition_dim != 0 or not config.rgb_supervision:
        raise ValueError("temporal-region evaluation requires RGB and no language")
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=saved_or_override(saved, "history_frames", args.history_frames),
        future_frames=saved_or_override(saved, "future_frames", args.future_frames),
        anchors=saved_or_override(saved, "sequence_anchors", args.sequence_anchors),
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
    region_config = TemporalRegionConfig(
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        low_floor=args.low_floor,
        high_floor=args.high_floor,
        low_mad_scale=args.low_mad_scale,
        high_mad_scale=args.high_mad_scale,
        hysteresis_steps=args.hysteresis_steps,
        change_dilation=args.change_dilation,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_global_step": checkpoint_global_step,
        "checkpoint_phase": checkpoint_phase,
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "data": os.path.abspath(args.data),
        "data_sha256": dataset.data_sha256,
        "split": args.split,
        "amp": args.amp,
        "requested_max_items": args.max_items,
        "available_samples": _available_samples(dataset),
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "evaluation": evaluate_temporal_regions(model, loader, device, args.amp, region_config),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
