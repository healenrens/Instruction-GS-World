#!/usr/bin/env python3
"""Decode one native clip per source and audit diversity/sampling contracts."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MULTISOURCE_POINT_TRACK_CONTRACT,
    MultiSourcePointTrackObjectVideoDataset,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def probe(dataset, index: int, chunk_length: int) -> dict:
    sample = dataset[(index, chunk_length)]
    rgb = sample["video_rgb"]
    times = sample["frame_times"]
    require(rgb.shape == (chunk_length, 3, 518, 518), "source RGB shape differs")
    require(rgb.dtype == torch.uint8, "source RGB must remain uint8 before DINO")
    require(bool((times[1:] > times[:-1]).all()), "source frame times are not increasing")
    valid_fraction = sample["video_pixel_valid"].float().mean(dim=(-2, -1))
    require(bool((valid_fraction >= 0.25).all()), "source clip has too little valid image area")
    forbidden = {
        "instruction", "condition_feature", "teacher_sidecar", "segmentation",
        "action", "dino", "task", "source_name",
    }
    require(not forbidden.intersection(sample), "source clip exposed non-video supervision")
    return {
        "source_index": int(sample["source_index"]),
        "task_group_index": int(sample["task_group_index"]),
        "temporal_step_seconds": float(sample["temporal_step_seconds"]),
        "valid_pixel_fraction": float(valid_fraction.mean()),
        "rgb_min": int(rgb.min()),
        "rgb_max": int(rgb.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        chunk_lengths=str(args.chunk_length),
        max_items=0,
        seed=args.seed,
    )
    require(
        len(dataset.source_probe_indices) == len(dataset.source_names),
        "data verifier cannot probe every source",
    )
    first_probes = [
        probe(dataset, index, args.chunk_length) for index in dataset.source_probe_indices
    ]
    offset_probes = [
        probe(dataset, index, args.chunk_length)
        for index in dataset.source_offset_probe_indices
    ]
    observed_sources = {item["source_index"] for item in first_probes}
    require(
        observed_sources == set(range(len(dataset.source_names))),
        "source probes do not cover the source table",
    )
    require(min(dataset.source_task_counts) > 0, "a source contains no usable task")
    normalized_budgets = [
        target / source.weight
        for target, source in zip(dataset.source_target_samples, dataset.sources)
    ]
    require(
        max(normalized_budgets) - min(normalized_budgets) <= 1.0,
        "source-first sampling budgets are not balanced by configured weights",
    )
    report = {
        "status": "passed",
        "contract": MULTISOURCE_POINT_TRACK_CONTRACT,
        "data_index": os.path.abspath(args.data_index),
        "dataset_examples": len(dataset),
        "balanced_epoch_samples": dataset.minimum_balanced_samples,
        "source_names": list(dataset.source_names),
        "source_episode_counts": list(dataset.source_episode_counts),
        "source_task_counts": list(dataset.source_task_counts),
        "source_target_samples": list(dataset.source_target_samples),
        "first_episode_probes": first_probes,
        "shared_video_offset_probes": offset_probes,
        "model_inputs": [
            "video_rgb", "video_pixel_valid", "observation_mask", "frame_times"
        ],
        "sampling_only_fields": ["source_index", "task_group_index"],
        "instruction_used": False,
        "explicit_action_used": False,
        "historical_checkpoint_used": False,
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
