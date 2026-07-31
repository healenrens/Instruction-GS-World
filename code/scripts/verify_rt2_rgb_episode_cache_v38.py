#!/usr/bin/env python3
"""Verify the complete RGB-only RoboTwin cache used by JIT DINO."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
import subprocess
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.episode_sequence_dataset import (  # noqa: E402
    CausalVisualEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    JIT_DINO_FEATURE_CONTRACT,
    SOURCE_INDEX_NAME,
    file_sha256,
    validate_episode_payload,
    validate_final_entry,
    validate_manifest,
)
from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    assign_episode_filenames,
    discover_lerobot_episodes,
    source_index_sha256,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(os.path.isabs(args.data), "--data must be absolute")
    require(os.path.isabs(args.output), "--output must be absolute")
    return args


def main() -> None:
    args = parse_args()
    manifest_path = os.path.join(args.data, EPISODE_MANIFEST_NAME)
    checksum_path = os.path.join(args.data, EPISODE_VERIFIED_NAME)
    require(os.path.isfile(manifest_path), "RGB episode manifest is missing")
    require(os.path.isfile(checksum_path), "RGB manifest checksum is missing")
    checksum = subprocess.run(
        ["sha256sum", "-c", "--status", EPISODE_VERIFIED_NAME],
        cwd=args.data,
        check=False,
    )
    require(checksum.returncode == 0, "RGB manifest checksum failed")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    control_hz = validate_manifest(manifest, manifest_path)

    source_path = os.path.join(args.data, SOURCE_INDEX_NAME)
    with open(source_path, encoding="utf-8") as handle:
        source = json.load(handle)
    require(
        source_index_sha256(source) == manifest["source"]["index_sha256"],
        "source index hash differs",
    )
    discovered = assign_episode_filenames(
        discover_lerobot_episodes(
            manifest["source"]["path"],
            manifest["source"]["variants"],
            float(manifest["source"]["expected_source_fps"]),
        )
    )
    require(discovered == source, "authoritative RoboTwin source metadata changed")

    entries = manifest["episodes"]
    expected = {entry["filename"] for entry in entries}
    actual = {
        os.path.basename(path) for path in glob.glob(os.path.join(args.data, "*.pt"))
    }
    require(actual == expected, "RGB episode file set differs from manifest")
    source_by_name = {episode["filename"]: episode for episode in source}
    total_bytes = 0
    split_counts = Counter()
    for entry in entries:
        path = os.path.join(args.data, entry["filename"])
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        validate_episode_payload(payload, path, entry, manifest)
        validate_final_entry(path, entry)
        source_episode = source_by_name.get(entry["filename"])
        require(
            source_episode is not None, "manifest episode is absent from source index"
        )
        require(
            int(source_episode["frame_count"]) == int(entry["frame_count"]),
            "episode frame count differs",
        )
        total_bytes += os.path.getsize(path)
        split_counts[entry["split"]] += 1

    dataset = CausalVisualEpisodeDataset(
        args.data,
        "train",
        history_frames=4,
        future_frames=4,
        anchors="3,5,8",
        max_items=1,
        feature_source="jit",
    )
    sample = dataset[0]
    require("dino" not in sample, "dataset sample leaked persisted DINO")
    require("history_features" not in sample, "dataset bypassed JIT DINO")
    require(
        sample["history_jit_rgb"].shape == (4, 3, 518, 518),
        "history JIT RGB shape differs",
    )
    require(
        sample["future_jit_rgb"].shape == (4, 3, 518, 518),
        "future JIT RGB shape differs",
    )
    require(
        dataset.feature_contract == JIT_DINO_FEATURE_CONTRACT,
        "dataset feature contract differs",
    )
    report = {
        "status": "passed",
        "contract": "rgb_only_episode_cache_for_jit_dino_v1",
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": file_sha256(manifest_path),
        "source_index_sha256": source_index_sha256(source),
        "control_hz": control_hz,
        "episodes": len(entries),
        "episodes_by_split": dict(sorted(split_counts.items())),
        "cache_bytes": total_bytes,
        "cache_gib": total_bytes / 2**30,
        "persisted_dino": False,
        "feature_source": "jit",
        "feature_contract": JIT_DINO_FEATURE_CONTRACT,
        "sample_history_rgb_shape": list(sample["history_jit_rgb"].shape),
        "sample_future_rgb_shape": list(sample["future_jit_rgb"].shape),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
