"""CPU contract test for dense visual-episode sequence sampling."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
from torchvision.io import encode_jpeg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    CONTROL_HZ,
    EPISODE_CACHE_VERSION,
    EPISODE_MANIFEST_NAME,
    EXPECTED_FRAME_COUNT,
    GROUP_SAMPLER_VERSION,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)


def packed_rgb(frames: torch.Tensor) -> dict:
    encoded = [encode_jpeg(frame, quality=95) for frame in frames]
    lengths = torch.tensor([len(value) for value in encoded], dtype=torch.long)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0))
    )
    return {
        "jpeg_bytes": torch.cat(encoded),
        "jpeg_offsets": offsets,
        "content_height": frames.shape[-2],
        "content_width": frames.shape[-1],
        "padded_height": frames.shape[-2],
        "padded_width": 32,
        "short_side": 16,
        "pad_multiple": 16,
        "jpeg_quality": 95,
    }


def write_episode(root: str, split: str, index: int, frame_count: int) -> dict:
    filename = f"synthetic_ep{index:02d}_{split}.pt"
    generator = torch.Generator().manual_seed(100 + index)
    rgb = torch.randint(
        0,
        256,
        (frame_count, 3, 16, 24),
        dtype=torch.uint8,
        generator=generator,
    )
    features = torch.randn(
        frame_count,
        2,
        3,
        4,
        generator=generator,
    ).to(torch.bfloat16)
    cache = {
        "episode_version": EPISODE_CACHE_VERSION,
        "source_name": filename,
        "split": split,
        "control_hz": CONTROL_HZ,
        "frame_control_indices": torch.arange(frame_count),
        "dino": features,
        "model": "synthetic",
        "image_size": 24,
        "feature_dim": 4,
        "projection_seed": 17,
        "projection_sha256": "a" * 64,
        "visual_preprocess": "synthetic",
        "rgb": packed_rgb(rgb),
    }
    torch.save(cache, os.path.join(root, filename))
    return {
        "filename": filename,
        "frame_count": frame_count,
        "split": split,
        "sampling_group": f"group-{index}",
    }


def expected_examples(frame_count: int) -> int:
    return sum(frame_count - window for window in (25, 50, 75, 100)) * 3


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        entries = [
            write_episode(root, split, index, 130)
            for index, split in enumerate(
                ("train", "train", "heldseed", "heldtask")
            )
        ]
        manifest = {
            "episode_cache_version": EPISODE_CACHE_VERSION,
            "complete": True,
            "source": {
                "kind": "synthetic",
                "path": root,
                "index_sha256": "b" * 64,
            },
            "split_contract": {
                "name": "synthetic",
                "heldseed_fraction": 0.2,
                "heldtasks": [],
            },
            "control_hz": CONTROL_HZ,
            "sample_frame_count": EXPECTED_FRAME_COUNT,
            "sampling": {
                "window_lengths": [25, 50, 75, 100],
                "sample_stride": 1,
                "group_balance": "task_sqrt_coverage",
                "group_sampling_temperature": 0.5,
                "group_sampler_version": GROUP_SAMPLER_VERSION,
            },
            "cache": {
                "model": "synthetic",
                "image_size": 24,
                "feature_dim": 4,
                "projection_seed": 17,
                "rgb_short_side": 16,
                "rgb_pad_multiple": 16,
                "jpeg_quality": 95,
            },
            "episodes": entries,
            "projection_sha256": "a" * 64,
        }
        manifest_path = os.path.join(root, EPISODE_MANIFEST_NAME)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)

        dataset = CausalVisualSequenceDataset(
            root,
            "train",
            load_rgb=True,
            explicit_goal=True,
            rgb_short_side=16,
            rgb_pad_multiple=16,
        )
        if len(dataset) != 2 * expected_examples(130):
            raise AssertionError("dense episode example count differs")
        if not dataset.balance_sampling or len(dataset.sampling_group_spans) != 2:
            raise AssertionError("dense episode sampling groups differ")
        samples = [dataset[index] for index in (0, 1, 314, len(dataset) - 1)]
        for sample in samples:
            if sample["history_features"].shape != (4, 6, 4):
                raise AssertionError("history feature shape differs")
            if sample["future_features"].shape != (4, 6, 4):
                raise AssertionError("future feature shape differs")
            if sample["history_rgb"].shape != (4, 3, 16, 32):
                raise AssertionError("history RGB shape differs")
            if float(sample["history_times"][-1]) != 0.0:
                raise AssertionError("history anchor time must be zero")
            if not bool((sample["future_times"] > 0).all()):
                raise AssertionError("future times must be positive")
            if sample["goal_time"] != sample["future_times"][-1]:
                raise AssertionError("goal time differs from future endpoint")
            if sample["goal_control_index"] != sample["future_control_indices"][-1]:
                raise AssertionError("goal control index differs from endpoint")

        selected = CausalVisualSequenceDataset(
            root,
            "heldseed",
            max_items=17,
            load_rgb=False,
            rgb_short_side=16,
            rgb_pad_multiple=16,
        )
        if len(selected) != 17 or not selected.data_sha256:
            raise AssertionError("dense episode selection or manifest digest differs")
        schedules = {
            tuple(float(value) for value in selected[index]["future_times"])
            for index in range(len(selected))
        }
        if len(schedules) < 4:
            raise AssertionError("dense sampling does not expose enough time schedules")
        balanced_selected = CausalVisualSequenceDataset(
            root,
            "train",
            max_items=17,
            load_rgb=False,
            rgb_short_side=16,
            rgb_pad_multiple=16,
        )
        if (
            len(balanced_selected.sampling_group_spans) != 2
            or balanced_selected.sampling_group_spans[0][0] != 0
            or balanced_selected.sampling_group_spans[-1][1] != 17
        ):
            raise AssertionError("selected sampling groups do not cover the subset")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "full_examples": len(dataset),
                    "selected_examples": len(selected),
                    "time_schedules": len(schedules),
                    "data_sha256": selected.data_sha256,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
