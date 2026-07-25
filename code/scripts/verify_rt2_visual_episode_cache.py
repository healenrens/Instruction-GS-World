"""Verify a complete dense visual-episode cache and sampling contract."""
from __future__ import annotations

import argparse
import glob
import hashlib
import h5py
import json
import os
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.episode_sequence_dataset import (  # noqa: E402
    CausalVisualEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.episode_cache_contract import (  # noqa: E402
    file_sha256,
    validate_episode_cache_header,
)
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    sqrt_coverage_targets,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    CONTROL_HZ,
    EPISODE_CACHE_VERSION,
    EPISODE_MANIFEST_NAME,
    EXPECTED_FRAME_COUNT,
    GROUP_SAMPLER_VERSION,
    RT2_HELDSEED_FRACTION,
    RT2_HELDTASKS,
    control_frame_indices,
    parse_anchor_indices,
    parse_control_windows,
    temporal_layout,
    stable_rt2_episode_split,
)


SOURCE_INDEX_NAME = "episode_source_index.json"


def start_count(frame_count: int, window: int, stride: int) -> int:
    last = frame_count - 1 - window
    if last < 0:
        return 0
    return last // stride + 1 + int(last % stride != 0)


def sampling_summary(manifest: dict, anchors: tuple[int, ...]) -> dict:
    windows = parse_control_windows(
        manifest["sampling"]["window_lengths"]
    )
    stride = int(manifest["sampling"]["sample_stride"])
    split_windows = Counter()
    split_examples = Counter()
    split_frames = Counter()
    task_examples: dict[str, Counter] = {}
    task_episodes: dict[str, Counter] = {}
    for episode in manifest["episodes"]:
        split = episode["split"]
        group = str(episode["sampling_group"])
        frame_count = int(episode["frame_count"])
        split_frames[split] += frame_count
        window_counts = {
            window: start_count(frame_count, window, stride)
            for window in windows
        }
        count = sum(window_counts.values())
        split_windows[split] += count
        split_examples[split] += count * len(anchors)
        task_examples.setdefault(split, Counter())[group] += (
            count * len(anchors)
        )
        task_episodes.setdefault(split, Counter())[group] += 1
    time_rows = set()
    for window in windows:
        controls = control_frame_indices(0, window, EXPECTED_FRAME_COUNT)
        for anchor in anchors:
            _, future = temporal_layout(
                EXPECTED_FRAME_COUNT,
                anchor,
                history_frames=4,
                future_frames=4,
            )
            relative = tuple(
                round(float(value), 9)
                for value in (
                    (controls[future] - controls[anchor]).float() / CONTROL_HZ
                )
            )
            time_rows.add(relative)
    group_ranges = {}
    for split, counts in task_examples.items():
        episodes = task_episodes[split]
        maximum = max(counts.values())
        targets = sqrt_coverage_targets(tuple(counts.values()))
        group_ranges[split] = {
            "task_groups": len(counts),
            "raw_examples_min": min(counts.values()),
            "raw_examples_max": max(counts.values()),
            "coverage_target_min": min(targets),
            "coverage_target_max": max(targets),
            "coverage_target_total": sum(targets),
            "maximum_repeat_factor": max(
                target / count
                for target, count in zip(targets, counts.values())
            ),
            "episodes_min": min(episodes.values()),
            "episodes_max": max(episodes.values()),
        }
    return {
        "window_lengths": list(windows),
        "sample_stride": stride,
        "group_balance": manifest["sampling"].get("group_balance", "none"),
        "group_sampling_temperature": manifest["sampling"].get(
            "group_sampling_temperature"
        ),
        "group_sampler_version": manifest["sampling"].get(
            "group_sampler_version"
        ),
        "anchors": list(anchors),
        "unique_future_time_rows": len(time_rows),
        "frames_by_split": dict(sorted(split_frames.items())),
        "windows_by_split": dict(sorted(split_windows.items())),
        "examples_by_split": dict(sorted(split_examples.items())),
        "group_ranges_by_split": dict(sorted(group_ranges.items())),
    }


def verify_files(data: str, manifest: dict) -> dict:
    entries = manifest["episodes"]
    expected = {entry["filename"] for entry in entries}
    actual = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(data, "*.pt"))
    }
    if actual != expected:
        raise ValueError(
            "episode cache file set differs: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    projection_hash = manifest.get("projection_sha256")
    bytes_total = 0
    split_count = Counter()
    for entry in entries:
        path = os.path.join(data, entry["filename"])
        cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        CausalVisualEpisodeDataset._validate_cache(cache, path)
        validate_episode_cache_header(
            cache,
            path,
            entry,
            manifest["cache"],
            projection_hash,
        )
        if (
            cache["split"] != entry["split"]
            or cache["source_name"] != entry["filename"]
            or len(cache["dino"]) != int(entry["frame_count"])
            or cache["projection_sha256"] != projection_hash
        ):
            raise ValueError(f"episode cache header differs from manifest: {path}")
        if (
            os.path.getsize(path) != int(entry["cache_bytes"])
            or file_sha256(path) != entry["cache_sha256"]
        ):
            raise ValueError(f"episode cache payload hash differs: {path}")
        split_count[entry["split"]] += 1
        bytes_total += os.path.getsize(path)
    return {
        "files": len(entries),
        "files_by_split": dict(sorted(split_count.items())),
        "bytes": bytes_total,
        "gib": bytes_total / 2**30,
        "projection_sha256": projection_hash,
    }


def verify_source_index(data: str, manifest: dict) -> dict:
    path = os.path.join(data, SOURCE_INDEX_NAME)
    with open(path, encoding="utf-8") as handle:
        source = json.load(handle)
    indexed = {
        (
            item["filename"],
            int(item["frame_count"]),
            item["split"],
        )
        for item in source
    }
    manifested = {
        (
            item["filename"],
            int(item["frame_count"]),
            item["split"],
        )
        for item in manifest["episodes"]
    }
    if indexed != manifested:
        raise ValueError("episode source index differs from final manifest")
    source_by_filename = {item["filename"]: item for item in source}
    for item in manifest["episodes"]:
        task = str(source_by_filename[item["filename"]]["task"])
        expected_group = hashlib.sha256(task.encode()).hexdigest()[:16]
        if item.get("sampling_group") != expected_group:
            raise ValueError("episode sampling group differs from source task")
    missing = [
        item["hdf5"]
        for item in source
        if not os.path.isfile(item["hdf5"])
    ]
    if missing:
        raise ValueError(f"source index contains missing HDF5 files: {missing[:3]}")
    digest = hashlib.sha256()
    for item in source:
        digest.update(
            (
                f"{item['task']}\0{item['episode']}\0"
                f"{item['frame_count']}\0{item['split']}\0"
                f"{item['hdf5']}\0"
            ).encode()
        )
    if digest.hexdigest() != manifest["source"]["index_sha256"]:
        raise ValueError("episode source index digest differs from manifest")
    if manifest["source"]["kind"] == "robotwin_root":
        if manifest["split_contract"] != {
            "name": "rt2_task_md5_v1",
            "heldseed_fraction": RT2_HELDSEED_FRACTION,
            "heldtasks": list(RT2_HELDTASKS),
        }:
            raise ValueError("full RoboTwin source split contract differs")
        root = manifest["source"]["path"]
        discovered = set(
            glob.glob(
                os.path.join(root, "*", "demo_clean", "data", "episode*.hdf5")
            )
        )
        indexed_paths = {item["hdf5"] for item in source}
        if discovered != indexed_paths:
            raise ValueError(
                "full RoboTwin root differs from source index: "
                f"missing={len(discovered - indexed_paths)} "
                f"extra={len(indexed_paths - discovered)}"
            )
        for item in source:
            if item["split"] != stable_rt2_episode_split(
                item["task"],
                int(item["episode"]),
            ):
                raise ValueError("source index split differs from stable contract")
            with h5py.File(item["hdf5"], "r") as handle:
                frame_count = len(handle["observation/head_camera/rgb"])
            if frame_count != int(item["frame_count"]):
                raise ValueError("source index frame count differs from HDF5")
    return {
        "source_kind": manifest["source"]["kind"],
        "source_path": manifest["source"]["path"],
        "source_index_sha256": digest.hexdigest(),
        "tasks": len({item["task"] for item in source}),
    }


def verify_dataset_samples(
    data: str,
    anchors: str,
    max_items: int,
) -> dict:
    result = {}
    for split in ("train", "heldseed", "heldtask"):
        dataset = CausalVisualEpisodeDataset(
            data,
            split,
            anchors=anchors,
            max_items=max_items,
            load_rgb=True,
            explicit_goal=True,
        )
        if not dataset.balance_sampling:
            raise ValueError("episode dataset did not enable group balancing")
        schedules = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample["goal_control_index"] != sample["future_control_indices"][-1]:
                raise ValueError("goal endpoint differs from future endpoint")
            if not bool((sample["future_times"] > 0).all()):
                raise ValueError("future physical times must be positive")
            schedules.add(
                tuple(round(float(value), 9) for value in sample["future_times"])
            )
        result[split] = {
            "sampled": len(dataset),
            "sampled_unique_future_time_rows": len(schedules),
            "sampled_groups": len(dataset.sampling_group_spans),
            "feature_shape": list(dataset[0]["history_features"].shape),
            "rgb_shape": list(dataset[0]["history_rgb"].shape),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anchors", default="3,5,8")
    parser.add_argument("--sample_items", type=int, default=48)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.sample_items < 1:
        raise ValueError("sample_items must be positive")
    parse_anchor_indices(args.anchors)
    return args


def main() -> None:
    args = parse_args()
    manifest_path = os.path.join(args.data, EPISODE_MANIFEST_NAME)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("episode_cache_version") != EPISODE_CACHE_VERSION
        or manifest.get("complete") is not True
        or manifest.get("sampling", {}).get("group_sampler_version")
        != GROUP_SAMPLER_VERSION
    ):
        raise ValueError("episode manifest is incomplete or incompatible")
    anchors = parse_anchor_indices(args.anchors)
    report = {
        "status": "ok",
        "data": os.path.abspath(args.data),
        "manifest_sha256": file_sha256(manifest_path),
        "source": verify_source_index(args.data, manifest),
        "cache": verify_files(args.data, manifest),
        "sampling": sampling_summary(manifest, anchors),
        "dataset_samples": verify_dataset_samples(
            args.data,
            args.anchors,
            args.sample_items,
        ),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
