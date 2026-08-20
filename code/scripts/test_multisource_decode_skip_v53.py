#!/usr/bin/env python3
"""Contract tests for deterministic v53 video replacement."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourcePointTrackObjectVideoDataset,
    _EpisodeRecord,
)
from igsw.adaptive_gaussian_wm.multisource_video_index import (  # noqa: E402
    MULTISOURCE_VIDEO_CONTRACT,
    load_multisource_index,
)
from igsw.adaptive_gaussian_wm.video_file_decoder import VideoDecodeError  # noqa: E402


def record(index: int, group: int, path: str, source: int = 0) -> _EpisodeRecord:
    return _EpisodeRecord(
        source_index=source,
        episode_index=index,
        sequence_index=index,
        group_index=group,
        adapter="video_file",
        path=path,
        fps=30.0,
        frame_count=20,
        frame_offset=0,
        start_stride=1,
        regular_starts=4,
        include_tail=False,
    )


def replacement_contract() -> None:
    dataset = MultiSourcePointTrackObjectVideoDataset.__new__(
        MultiSourcePointTrackObjectVideoDataset
    )
    dataset.seed = 17
    dataset.dynamic_history_lengths = (3,)
    dataset.temporal_step_ms = (100,)
    dataset._records = [
        record(0, 0, "/bad-a.mp4"),
        record(1, 0, "/good-a.mp4"),
        record(2, 1, "/good-b.mp4"),
    ]
    dataset._prefix = [4, 8, 12]
    dataset._full_length = dataset._length = 12
    dataset._decode_quarantine = bytearray(len(dataset._records))
    dataset._build_replacement_pools()
    attempts = []

    def decode(self, item, indices):
        attempts.append(item.path)
        if item.path == "/bad-a.mp4":
            raise VideoDecodeError("synthetic decode miss")
        frames = len(indices)
        return (
            torch.zeros(frames, 3, 4, 4, dtype=torch.uint8),
            torch.ones(frames, 4, 4, dtype=torch.bool),
        )

    dataset._decode = types.MethodType(decode, dataset)
    sample = dataset[(0, 3)]
    assert bool(sample["decode_replaced"])
    assert int(sample["requested_sequence_index"]) == 0
    assert int(sample["task_group_index"]) == 0
    assert attempts.count("/bad-a.mp4") == 1

    dataset[(0, 3)]
    assert attempts.count("/bad-a.mp4") == 1

    dataset._decode_quarantine[1] = 1
    fallback = dataset[(0, 3)]
    assert int(fallback["source_index"]) == 0
    assert int(fallback["task_group_index"]) == 1


def exhausted_scope_contract() -> None:
    dataset = MultiSourcePointTrackObjectVideoDataset.__new__(
        MultiSourcePointTrackObjectVideoDataset
    )
    dataset.seed = 17
    dataset.dynamic_history_lengths = (3,)
    dataset.temporal_step_ms = (100,)
    dataset._records = [
        *[record(index, 0, f"/bad-{index}.mp4") for index in range(40)],
        record(40, 1, "/source-fallback.mp4"),
        record(41, 2, "/global-fallback.mp4", source=1),
    ]
    dataset._prefix = [4 * (index + 1) for index in range(len(dataset._records))]
    dataset._full_length = dataset._length = dataset._prefix[-1]
    dataset._decode_quarantine = bytearray(len(dataset._records))
    dataset._build_replacement_pools()

    def decode(self, item, indices):
        if item.path.startswith("/bad-"):
            raise VideoDecodeError("synthetic decode miss")
        frames = len(indices)
        return (
            torch.zeros(frames, 3, 4, 4, dtype=torch.uint8),
            torch.ones(frames, 4, 4, dtype=torch.bool),
        )

    dataset._decode = types.MethodType(decode, dataset)
    source_fallback = dataset[(0, 3)]
    assert int(source_fallback["source_index"]) == 0
    assert int(source_fallback["task_group_index"]) == 1

    dataset._decode_quarantine[:41] = bytes([1]) * 41
    global_fallback = dataset[(0, 3)]
    assert int(global_fallback["source_index"]) == 1


def path_quarantine_contract() -> None:
    dataset = MultiSourcePointTrackObjectVideoDataset.__new__(
        MultiSourcePointTrackObjectVideoDataset
    )
    dataset.seed = 17
    dataset._records = [
        record(0, 0, "/missing.mp4"),
        record(1, 0, "/missing.mp4"),
        record(2, 0, "/good.mp4"),
    ]
    dataset._decode_quarantine = bytearray(len(dataset._records))
    dataset._build_replacement_pools()
    dataset._quarantine_decode_failure(
        0, VideoDecodeError("missing", path_unusable=True)
    )
    assert dataset._decode_quarantine == bytearray((1, 1, 0))


def missing_payload_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        present = os.path.join(directory, "present.mp4")
        missing = os.path.join(directory, "missing.mp4")
        open(present, "wb").close()
        payload = {
            "contract": MULTISOURCE_VIDEO_CONTRACT,
            "sources": [{"name": "source", "adapter": "video_file", "weight": 1}],
            "episodes": [
                {
                    "source_index": 0,
                    "episode_index": index,
                    "split": "train",
                    "group": "task",
                    "path": path,
                    "fps": 30,
                    "frame_count": 20,
                    "frame_offset": 0,
                }
                for index, path in enumerate((present, missing))
            ],
        }
        index_path = os.path.join(directory, "index.json")
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        _, episodes, loaded = load_multisource_index(
            index_path, skip_missing_payloads=True
        )
        assert len(episodes) == 1
        assert loaded["runtime_missing_video_count"] == 1
        try:
            load_multisource_index(index_path)
        except ValueError as error:
            assert "episode payload is missing" in str(error)
        else:
            raise AssertionError("strict index loading accepted a missing payload")


if __name__ == "__main__":
    replacement_contract()
    exhausted_scope_contract()
    path_quarantine_contract()
    missing_payload_contract()
    print("multisource decode skip v53: passed")
