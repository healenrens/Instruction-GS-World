"""Held-split evaluation for observation-complete object-state v46."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.observation_complete_world_model import (  # noqa: E402
    ObservationCompleteWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import TemporalObjectVideoDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.v46_config import (  # noqa: E402
    ARCHITECTURE, CHECKPOINT_VERSION, ObservationCompleteConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("v46 evaluator collected no values")
    return sum(values) / len(values)


def _batch_metrics(output) -> dict[str, float]:
    parts = output["parts"]
    names = (
        "loss_observation_complete", "diagnostic_scene_only_error",
        "diagnostic_object_gain", "diagnostic_motion_object_gain",
        "loss_masked_observation", "object_identity_top1",
        "object_effective_count", "scene_owner_fraction",
        "owner_assignment_entropy", "association_unmatched_probability",
        "effect_correct_distance", "effect_zero_distance",
        "effect_shuffled_distance", "effect_relative_gain_over_zero",
        "effect_relative_gain_over_shuffled", "effect_action_norm",
        "effect_action_std", "goal_correct_distance", "goal_zero_distance",
        "goal_shuffled_distance", "goal_relative_gain_over_zero",
    )
    return {name: float(parts[name]) for name in names}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v46 evaluator requires a visible CUDA device")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        os.path.abspath(args.checkpoint), map_location="cpu",
        weights_only=False, mmap=True,
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v46 evaluator requires a version-46 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v46 evaluator checkpoint architecture differs")
    config = ObservationCompleteConfig(**checkpoint["config"])
    dataset = TemporalObjectVideoDataset(
        args.data, args.split, max_items=args.max_items,
        seed=args.seed, record_manifest_hash=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    device = torch.device("cuda:0")
    model = ObservationCompleteWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder = FrozenDinoVideoRuntime(config, device, args.amp, args.dino_frame_batch)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    collected: dict[str, list[float]] = {}
    items = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = {
                name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for name, value in cpu_batch.items()
            }
            features = encoder(batch)
            with amp_context():
                output = model(
                    features.patches, features.coordinates, features.valid,
                    batch["frame_times"], batch["observation_mask"], config.total_steps,
                )
            metrics = _batch_metrics(output)
            for name, value in metrics.items():
                collected.setdefault(name, []).append(value)
            items += len(batch["video_rgb"])
    metrics = {name: _mean(values) for name, values in collected.items()}
    report = {
        "status": "completed", "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data), "split": args.split, "items": items,
        "metrics": metrics,
        "checks": {
            "objects_improve_observation_over_scene": metrics["diagnostic_object_gain"] > 0,
            "objects_help_motion_patches": metrics["diagnostic_motion_object_gain"] > 0,
            "identity_beats_chance": metrics["object_identity_top1"] > 1.0 / config.object_slots,
            "effect_beats_zero": metrics["effect_relative_gain_over_zero"] > 0,
            "effect_beats_shuffled": metrics["effect_relative_gain_over_shuffled"] > 0,
            "image_goal_beats_zero": metrics["goal_relative_gain_over_zero"] > 0,
            "effect_not_unit_norm": metrics["effect_action_norm"] < 0.95,
        },
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
