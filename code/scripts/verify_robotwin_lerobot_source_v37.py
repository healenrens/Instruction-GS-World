#!/usr/bin/env python3
"""Validate the native-30-Hz RoboTwin LeRobot visual source before caching."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    LEROBOT_DEFAULT_ROOT,
    LEROBOT_DEFAULT_VARIANTS,
    LEROBOT_EXPECTED_FPS,
    assign_episode_filenames,
    decode_lerobot_batch,
    discover_lerobot_episodes,
    parse_source_variants,
    source_index_sha256,
)
from igsw.adaptive_gaussian_wm.sequence_contract import RT2_HELDTASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default=LEROBOT_DEFAULT_ROOT)
    parser.add_argument(
        "--source_variants",
        default=",".join(LEROBOT_DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--expected_source_fps", type=float, default=LEROBOT_EXPECTED_FPS
    )
    parser.add_argument("--samples_per_variant", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for name in ("source_root", "output"):
        if not os.path.isabs(getattr(args, name)):
            raise ValueError(f"--{name} must be absolute")
    if args.samples_per_variant < 1:
        raise ValueError("samples_per_variant must be positive")
    return args


def evenly_spaced(records: list[dict], count: int) -> list[dict]:
    count = min(count, len(records))
    if count == 1:
        return [records[0]]
    return [
        records[(index * (len(records) - 1)) // (count - 1)] for index in range(count)
    ]


def frame_digest(frame) -> str:
    return hashlib.sha256(frame.contiguous().numpy().tobytes()).hexdigest()


def decode_samples(
    episodes: list[dict], variants: tuple[str, ...], count: int
) -> list[dict]:
    samples = []
    for variant in variants:
        candidates = [item for item in episodes if item["source_variant"] == variant]
        for episode in evenly_spaced(candidates, count):
            indices = sorted(
                {
                    0,
                    int(episode["frame_count"]) // 2,
                    int(episode["frame_count"]) - 1,
                }
            )
            decoded = [
                decode_lerobot_batch(episode, index, index + 1)[0] for index in indices
            ]
            shapes = {tuple(frame.shape) for frame in decoded}
            if len(shapes) != 1:
                raise RuntimeError(
                    f"episode RGB shape changes: {episode['video_path']}"
                )
            samples.append(
                {
                    "task": episode["task"],
                    "source_variant": variant,
                    "episode": int(episode["episode"]),
                    "indices": indices,
                    "shape": list(shapes.pop()),
                    "frame_sha256": [frame_digest(frame) for frame in decoded],
                    "video_path": episode["video_path"],
                }
            )
    return samples


def main() -> None:
    args = parse_args()
    variants = parse_source_variants(args.source_variants)
    episodes = assign_episode_filenames(
        discover_lerobot_episodes(
            args.source_root,
            variants,
            args.expected_source_fps,
        )
    )
    task_variants = {(item["task"], item["source_variant"]) for item in episodes}
    tasks = sorted({item["task"] for item in episodes})
    missing_pairs = [
        (task, variant)
        for task in tasks
        for variant in variants
        if (task, variant) not in task_variants
    ]
    if missing_pairs:
        raise RuntimeError(
            f"tasks are missing requested variants: {missing_pairs[:10]}"
        )
    missing_heldtasks = sorted(set(RT2_HELDTASKS) - set(tasks))
    if missing_heldtasks:
        raise RuntimeError(
            f"held tasks are absent from RoboTwin source: {missing_heldtasks}"
        )
    samples = decode_samples(episodes, variants, args.samples_per_variant)
    episode_counts = Counter(item["source_variant"] for item in episodes)
    frame_counts = Counter()
    for item in episodes:
        frame_counts[item["source_variant"]] += int(item["frame_count"])
    report = {
        "status": "passed",
        "contract": "robotwin2_lerobot_native_30hz_visual_v1",
        "source_root": os.path.abspath(args.source_root),
        "source_variants": list(variants),
        "source_fps": float(args.expected_source_fps),
        "source_frame_stride": 1,
        "tasks": len(tasks),
        "episodes": len(episodes),
        "episodes_by_variant": dict(sorted(episode_counts.items())),
        "frames_by_variant": dict(sorted(frame_counts.items())),
        "source_index_sha256": source_index_sha256(episodes),
        "decoded_samples": samples,
        "ignored_fields": ["action", "cot_mapping", "instruction", "task_index"],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
