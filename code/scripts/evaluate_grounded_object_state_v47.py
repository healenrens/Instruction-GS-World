"""Held-split evaluation for grounded object-state v47."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.observation_complete_world_model import (  # noqa: E402
    ObservationCompleteWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import TemporalObjectVideoDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.v47_config import (  # noqa: E402
    ARCHITECTURE, CHECKPOINT_VERSION, ObservationCompleteConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk_lengths", default="8,16,24,32")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("v47 evaluator collected no values")
    return sum(values) / len(values)


def _batch_metrics(output) -> dict[str, float]:
    parts = output["parts"]
    names = (
        "loss_observation_complete", "diagnostic_scene_only_error",
        "diagnostic_object_gain", "diagnostic_motion_object_gain",
        "loss_masked_observation", "loss_masked_state",
        "loss_object_grounding", "loss_track_observation_retrieval",
        "track_observation_retrieval_top1",
        "loss_presence_prediction", "loss_visibility_prediction",
        "object_effective_count", "object_supported_count",
        "object_owner_fraction", "object_utility", "scene_owner_fraction",
        "presence_mean", "visibility_mean", "presence_visibility_gap",
        "owner_assignment_entropy", "association_unmatched_probability",
        "effect_correct_distance", "effect_zero_distance",
        "effect_shuffled_distance", "effect_relative_gain_over_zero",
        "effect_relative_gain_over_shuffled", "effect_action_norm",
        "effect_action_std", "goal_correct_distance", "goal_zero_distance",
        "goal_shuffled_distance", "goal_relative_gain_over_zero",
    )
    return {name: float(parts[name]) for name in names}


class _FixedChunkView(Dataset):
    def __init__(self, dataset: TemporalObjectVideoDataset, chunk_length: int):
        self.dataset = dataset
        self.chunk_length = int(chunk_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.dataset[(index, self.chunk_length)]


def _evaluate_length(
    model,
    encoder,
    dataset,
    chunk_length: int,
    args,
    device: torch.device,
    amp_context,
) -> tuple[dict[str, float], int]:
    loader = DataLoader(
        _FixedChunkView(dataset, chunk_length),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
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
                    batch["frame_times"], batch["observation_mask"], model.config.total_steps,
                )
            for name, value in _batch_metrics(output).items():
                collected.setdefault(name, []).append(value)
            items += len(batch["video_rgb"])
    return {name: _mean(values) for name, values in collected.items()}, items


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v47 evaluator requires a visible CUDA device")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        os.path.abspath(args.checkpoint), map_location="cpu",
        weights_only=False, mmap=True,
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v47 evaluator requires a version-47 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v47 evaluator checkpoint architecture differs")
    config = ObservationCompleteConfig(**checkpoint["config"])
    dataset = TemporalObjectVideoDataset(
        args.data, args.split, max_items=args.max_items,
        seed=args.seed, record_manifest_hash=False,
    )
    device = torch.device("cuda:0")
    model = ObservationCompleteWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    chunk_lengths = tuple(int(value) for value in args.chunk_lengths.split(","))
    if not chunk_lengths or any(value not in dataset.dynamic_history_lengths for value in chunk_lengths):
        raise ValueError("v47 evaluator chunk lengths differ from the dataset contract")
    metrics_by_chunk_length, items_by_chunk_length = {}, {}
    for chunk_length in chunk_lengths:
        metrics, items = _evaluate_length(
            model, encoder, dataset, chunk_length, args, device, amp_context
        )
        metrics_by_chunk_length[str(chunk_length)] = metrics
        items_by_chunk_length[str(chunk_length)] = items
    metric_names = tuple(next(iter(metrics_by_chunk_length.values())))
    metrics = {
        name: _mean([values[name] for values in metrics_by_chunk_length.values()])
        for name in metric_names
    }
    report = {
        "status": "completed", "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data), "split": args.split,
        "items_by_chunk_length": items_by_chunk_length,
        "metrics": metrics, "metrics_by_chunk_length": metrics_by_chunk_length,
        "checks": {
            "objects_improve_observation_over_scene": metrics["diagnostic_object_gain"] > 0,
            "objects_help_motion_patches": metrics["diagnostic_motion_object_gain"] > 0,
            "track_observation_retrieval_beats_chance": (
                metrics["track_observation_retrieval_top1"] > 1.0 / config.object_slots
            ),
            "at_least_one_supported_object": metrics["object_supported_count"] > 0,
            "effect_beats_zero": metrics["effect_relative_gain_over_zero"] > 0,
            "effect_beats_shuffled": metrics["effect_relative_gain_over_shuffled"] > 0,
            "image_goal_beats_zero": metrics["goal_relative_gain_over_zero"] > 0,
            "effect_not_unit_norm": metrics["effect_action_norm"] < 0.95,
            "all_lengths_improve_over_scene": all(
                values["diagnostic_object_gain"] > 0
                for values in metrics_by_chunk_length.values()
            ),
            "all_lengths_have_supported_objects": all(
                values["object_supported_count"] > 0
                for values in metrics_by_chunk_length.values()
            ),
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
