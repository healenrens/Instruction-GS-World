"""Continuous raw-video chunks for the v44 Temporal Object Set model."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .group_balanced_sampler import sqrt_coverage_targets
from .rgb_episode_cache_contract import (
    file_sha256,
    validate_episode_payload,
    validate_manifest,
)
from .sequence_contract import EPISODE_MANIFEST_NAME


TEMPORAL_OBJECT_VIDEO_CONTRACT = "temporal_object_video_v1"


def _stable_integer(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def parse_int_choices(value: str, label: str) -> tuple[int, ...]:
    choices = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not choices or choices[0] < 1:
        raise ValueError(f"{label} must contain positive integers")
    return choices


@dataclass(frozen=True)
class _ChunkRecord:
    path: str
    manifest_index: int
    group: str
    frame_count: int
    regular_starts: int
    include_tail: bool

    @property
    def start_count(self) -> int:
        return self.regular_starts + int(self.include_tail)


class TemporalObjectVideoDataset(Dataset):
    """Decode deterministic 8-32 frame chunks without semantic supervision."""

    def __init__(
        self,
        cache_root: str,
        split: str,
        chunk_lengths: str = "8,16,24,32",
        temporal_strides: str = "1,2,3,4",
        observation_mask_probability: float = 0.20,
        max_items: int = 0,
        seed: int = 17,
    ):
        self.cache_root = os.path.abspath(cache_root)
        manifest_path = os.path.join(self.cache_root, EPISODE_MANIFEST_NAME)
        with open(manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.control_hz = validate_manifest(self.manifest, manifest_path)
        self.data_sha256 = file_sha256(manifest_path)
        self.contract_label = "raw-video-only continuous Temporal Object Set chunks"
        self.feature_dim = int(self.manifest["feature"]["feature_dim"])
        self.condition_dim = 0
        self.teacher_sidecar_sha256 = ""
        self.dynamic_history_lengths = parse_int_choices(chunk_lengths, "chunk lengths")
        self.temporal_strides = parse_int_choices(temporal_strides, "temporal strides")
        if self.dynamic_history_lengths[0] < 3 or self.dynamic_history_lengths[-1] > 32:
            raise ValueError("v44 chunk lengths must stay within [3,32]")
        if self.temporal_strides[-1] > 8:
            raise ValueError("v44 temporal stride is unexpectedly large")
        if not 0.0 <= observation_mask_probability < 1.0:
            raise ValueError("observation mask probability must be in [0,1)")
        self.observation_mask_probability = float(observation_mask_probability)
        self.seed = int(seed)
        self.sample_stride = int(self.manifest["sampling"]["sample_stride"])
        self.balance_sampling = True
        self._build_index(split, max_items)

    def _build_index(self, split: str, max_items: int) -> None:
        grouped: dict[str, list[tuple[int, dict]]] = {}
        for index, entry in enumerate(self.manifest["episodes"]):
            if entry["split"] == split:
                grouped.setdefault(str(entry["sampling_group"]), []).append(
                    (index, entry)
                )
        if not grouped:
            raise ValueError(f"RGB cache has no {split} episodes")
        minimum_span = self.dynamic_history_lengths[0] - 1
        records: list[_ChunkRecord] = []
        for group, entries in grouped.items():
            for manifest_index, entry in entries:
                path = os.path.join(self.cache_root, entry["filename"])
                if not os.path.isfile(path):
                    raise ValueError(f"episode cache is missing: {path}")
                last_start = int(entry["frame_count"]) - 1 - minimum_span
                if last_start < 0:
                    continue
                regular = last_start // self.sample_stride + 1
                records.append(
                    _ChunkRecord(
                        path=path,
                        manifest_index=manifest_index,
                        group=group,
                        frame_count=int(entry["frame_count"]),
                        regular_starts=regular,
                        include_tail=last_start % self.sample_stride != 0,
                    )
                )
        if not records:
            raise ValueError("RGB cache has no episode long enough for v44")
        self._records = records
        self._prefix: list[int] = []
        group_spans: list[tuple[int, int]] = []
        total = 0
        active_group = None
        group_start = 0
        for record in records:
            if active_group is None:
                active_group = record.group
            elif record.group != active_group:
                group_spans.append((group_start, total))
                group_start = total
                active_group = record.group
            total += record.start_count
            self._prefix.append(total)
        group_spans.append((group_start, total))
        self._full_length = total
        self._length = min(max_items, total) if max_items > 0 else total
        self.sampling_group_spans = self._project_spans(group_spans)
        self.sampling_group_targets = sqrt_coverage_targets(
            tuple(end - start for start, end in self.sampling_group_spans)
        )
        self.minimum_balanced_samples = sum(self.sampling_group_targets)
        self.paths = [record.path for record in records]
        first = self._load(records[0].path)
        entry = self.manifest["episodes"][records[0].manifest_index]
        validate_episode_payload(first, records[0].path, entry, self.manifest)
        self.rgb_height = int(first["rgb"]["height"])
        self.rgb_width = int(first["rgb"]["width"])

    def _project_boundary(self, boundary: int) -> int:
        if self._length == self._full_length:
            return boundary
        if self._length == 1:
            return int(boundary > 0)
        return (boundary * (self._length - 1) + self._full_length - 2) // (
            self._full_length - 1
        )

    def _project_spans(
        self, spans: list[tuple[int, int]]
    ) -> tuple[tuple[int, int], ...]:
        projected = []
        for start, end in spans:
            left, right = self._project_boundary(start), self._project_boundary(end)
            if left < right:
                projected.append((left, right))
        if not projected or projected[0][0] != 0 or projected[-1][1] != self._length:
            raise ValueError("projected v44 sampling groups do not cover the dataset")
        if any(left[1] != right[0] for left, right in zip(projected, projected[1:])):
            raise ValueError("projected v44 sampling groups are not contiguous")
        return tuple(projected)

    @staticmethod
    def _load(path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def _full_index(self, index: int) -> int:
        if not 0 <= index < self._length:
            raise IndexError(index)
        if self._length == self._full_length:
            return index
        if self._length == 1:
            return 0
        return index * (self._full_length - 1) // (self._length - 1)

    def _locate(self, index: int) -> tuple[_ChunkRecord, int]:
        full_index = self._full_index(index)
        record_index = bisect.bisect_right(self._prefix, full_index)
        previous = self._prefix[record_index - 1] if record_index else 0
        return self._records[record_index], full_index - previous

    def _decode(
        self, rgb: dict, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blob, offsets = rgb["jpeg_bytes"], rgb["jpeg_offsets"].long()
        frames = torch.stack(
            [
                decode_jpeg(
                    blob[int(offsets[index]) : int(offsets[index + 1])],
                    mode=ImageReadMode.RGB,
                )
                for index in indices.tolist()
            ]
        )
        if frames.shape[-2:] != (self.rgb_height, self.rgb_width):
            raise ValueError("decoded RGB shape differs from v44 cache metadata")
        return frames, torch.ones(
            len(indices), self.rgb_height, self.rgb_width, dtype=torch.bool
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        base_index, chunk_length = (
            index
            if isinstance(index, tuple)
            else (
                index,
                self.dynamic_history_lengths[-1],
            )
        )
        if chunk_length not in self.dynamic_history_lengths:
            raise ValueError(f"unsupported v44 chunk length: {chunk_length}")
        record, ordinal = self._locate(int(base_index))
        stride = self.temporal_strides[
            _stable_integer(self.seed, record.manifest_index, ordinal, chunk_length)
            % len(self.temporal_strides)
        ]
        span = (chunk_length - 1) * stride
        maximum_start = record.frame_count - 1 - span
        if maximum_start < 0:
            valid_strides = [
                value
                for value in self.temporal_strides
                if (chunk_length - 1) * value < record.frame_count
            ]
            if not valid_strides:
                raise ValueError("episode cannot supply the requested v44 chunk")
            stride = valid_strides[-1]
            span = (chunk_length - 1) * stride
            maximum_start = record.frame_count - 1 - span
        fraction = ordinal / max(record.start_count - 1, 1)
        start = round(fraction * maximum_start)
        controls = start + torch.arange(chunk_length, dtype=torch.long) * stride
        cache = self._load(record.path)
        entry = self.manifest["episodes"][record.manifest_index]
        validate_episode_payload(cache, record.path, entry, self.manifest)
        rgb, valid = self._decode(cache["rgb"], controls)
        generator = torch.Generator().manual_seed(
            _stable_integer(
                "mask", self.seed, record.manifest_index, ordinal, chunk_length
            )
        )
        observed = torch.rand(chunk_length, generator=generator) >= (
            self.observation_mask_probability
        )
        observed[0] = True
        observed[-1] = True
        return {
            "video_rgb": rgb,
            "video_pixel_valid": valid,
            "observation_mask": observed,
            "frame_times": torch.arange(chunk_length).float()
            * stride
            / self.control_hz,
            "control_indices": controls,
            "sequence_index": torch.tensor(record.manifest_index, dtype=torch.long),
            "chunk_length": torch.tensor(chunk_length, dtype=torch.long),
            "temporal_stride": torch.tensor(stride, dtype=torch.long),
        }
