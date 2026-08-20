"""Task-diverse, source-balanced RGB clips from native robot-video sources."""

from __future__ import annotations

import bisect
import hashlib
import math
import mmap
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .multisource_video_index import load_multisource_index
from .temporal_object_dataset import parse_int_choices
from .video_file_decoder import VideoDecodeError, decode_video_frames, square_dino_rgb


MULTISOURCE_POINT_TRACK_CONTRACT = "multisource_point_track_object_video_v1"
MULTISOURCE_VIDEO_CONTRACT = "multisource_native_robot_video_v1"


def _stable_integer(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _parse_milliseconds(value: str) -> tuple[int, ...]:
    choices = parse_int_choices(value, "temporal step milliseconds")
    if choices[-1] > 2000:
        raise ValueError("temporal steps exceed the v52 clip contract")
    return choices


@dataclass(frozen=True)
class _EpisodeRecord:
    source_index: int
    episode_index: int
    sequence_index: int
    group_index: int
    adapter: str
    path: str
    fps: float
    frame_count: int
    frame_offset: int
    start_stride: int
    regular_starts: int
    include_tail: bool

    @property
    def start_count(self) -> int:
        return self.regular_starts + int(self.include_tail)


class MultiSourcePointTrackObjectVideoDataset(Dataset):
    """Read native videos without duplicating them into a frame cache."""

    def __init__(
        self,
        index_path: str,
        split: str,
        chunk_lengths: str = "8,16,24,32",
        temporal_step_ms: str = "33,67,100,133",
        max_items: int = 0,
        seed: int = 17,
    ):
        self.index_path = os.path.abspath(index_path)
        self.sources, episodes, payload = load_multisource_index(
            self.index_path, skip_missing_payloads=True
        )
        self.dynamic_history_lengths = parse_int_choices(chunk_lengths, "chunk lengths")
        self.temporal_step_ms = _parse_milliseconds(temporal_step_ms)
        if self.dynamic_history_lengths[0] < 3 or self.dynamic_history_lengths[-1] > 32:
            raise ValueError("v52 chunk lengths must stay within [3,32]")
        self.start_step_seconds = float(payload.get("start_step_seconds", 0.5))
        self.samples_per_task = int(payload.get("samples_per_task", 4096))
        if self.start_step_seconds <= 0.0 or self.samples_per_task < 1:
            raise ValueError("multisource sampling configuration is invalid")
        self.seed = int(seed)
        self.balance_sampling = True
        self.sampling_group_targets_may_undersample = True
        self.contract_label = "task-diverse multisource RGB-only Object State clips"
        self.condition_dim = 0
        self.teacher_sidecar_sha256 = ""
        self.data_sha256 = ""
        self.paths = [self.index_path]
        self.source_names = tuple(source.name for source in self.sources)
        self.runtime_missing_video_count = int(
            payload.get("runtime_missing_video_count", 0)
        )
        self._build_index(episodes, split, max_items)
        self._open_decode_quarantine(split)

    def _build_index(self, episodes, split: str, max_items: int) -> None:
        selected = [episode for episode in episodes if episode.split == split]
        if not selected:
            raise ValueError(f"multisource index has no {split} episodes")
        selected.sort(key=lambda item: (item.source_index, item.group, item.episode_index))
        group_keys = sorted({(item.source_index, item.group) for item in selected})
        group_ids = {key: index for index, key in enumerate(group_keys)}
        self._group_source_indices = tuple(source for source, _ in group_keys)
        self.sampling_group_names = tuple(group for _, group in group_keys)
        records = []
        required_frames = self.dynamic_history_lengths[-1]
        for sequence_index, episode in enumerate(selected):
            minimum_stride = max(1, round(self.temporal_step_ms[0] * episode.fps / 1000.0))
            last_start = episode.frame_count - 1 - (required_frames - 1) * minimum_stride
            if last_start < 0:
                continue
            start_stride = max(1, round(self.start_step_seconds * episode.fps))
            regular_starts = last_start // start_stride + 1
            records.append(
                _EpisodeRecord(
                    source_index=episode.source_index,
                    episode_index=episode.episode_index,
                    sequence_index=sequence_index,
                    group_index=group_ids[(episode.source_index, episode.group)],
                    adapter=self.sources[episode.source_index].adapter,
                    path=episode.path,
                    fps=episode.fps,
                    frame_count=episode.frame_count,
                    frame_offset=episode.frame_offset,
                    start_stride=start_stride,
                    regular_starts=regular_starts,
                    include_tail=last_start % start_stride != 0,
                )
            )
        if not records:
            raise ValueError("multisource index has no usable episode")
        self._records = records
        self._build_replacement_pools()
        self._prefix = []
        source_probe_full_indices = {}
        source_offset_probe_full_indices = {}
        total = 0
        group_bounds = []
        active_group = records[0].group_index
        group_start = 0
        for record in records:
            source_probe_full_indices.setdefault(record.source_index, total)
            if record.frame_offset > 0:
                source_offset_probe_full_indices.setdefault(record.source_index, total)
            if record.group_index != active_group:
                group_bounds.append((group_start, total, active_group))
                group_start, active_group = total, record.group_index
            total += record.start_count
            self._prefix.append(total)
        group_bounds.append((group_start, total, active_group))
        self._full_length = total
        self._length = min(int(max_items), total) if max_items > 0 else total
        self.source_probe_indices = (
            tuple(source_probe_full_indices[index] for index in range(len(self.sources)))
            if self._length == self._full_length else ()
        )
        self.source_offset_probe_indices = (
            tuple(
                source_offset_probe_full_indices.get(index, source_probe_full_indices[index])
                for index in range(len(self.sources))
            )
            if self._length == self._full_length else ()
        )
        projected = self._project_group_spans(group_bounds)
        self.sampling_group_spans = tuple((left, right) for left, right, _ in projected)
        self.sampling_group_targets = self._task_diverse_targets(projected)
        self.minimum_balanced_samples = sum(self.sampling_group_targets)
        self.source_episode_counts = tuple(
            sum(record.source_index == source for record in records)
            for source in range(len(self.sources))
        )
        self.source_task_counts = tuple(
            sum(
                self._group_source_indices[group] == source
                for _, _, group in projected
            )
            for source in range(len(self.sources))
        )
        self.source_target_samples = tuple(
            sum(
                target
                for (_, _, group), target in zip(projected, self.sampling_group_targets)
                if self._group_source_indices[group] == source
            )
            for source in range(len(self.sources))
        )
        self.rgb_height = self.rgb_width = 518

    def _build_replacement_pools(self) -> None:
        group_records: dict[int, list[int]] = {}
        source_records: dict[int, list[int]] = {}
        path_records: dict[str, list[int]] = {}
        for record_index, record in enumerate(self._records):
            group_records.setdefault(record.group_index, []).append(record_index)
            source_records.setdefault(record.source_index, []).append(record_index)
            path_records.setdefault(record.path, []).append(record_index)
        self._group_replacement_records = {
            group: tuple(indices) for group, indices in group_records.items()
        }
        self._source_replacement_records = {
            source: tuple(indices) for source, indices in source_records.items()
        }
        self._path_replacement_records = {
            path: tuple(indices) for path, indices in path_records.items()
        }
        self._all_replacement_records = tuple(range(len(self._records)))

    def _open_decode_quarantine(self, split: str) -> None:
        index_stat = os.stat(self.index_path)
        namespace = f"{index_stat.st_size}_{index_stat.st_mtime_ns}_{split}"
        root = os.environ.get("V53_DECODE_QUARANTINE_ROOT", "/dev/shm")
        path = os.path.join(root, f"igsw_v53_decode_quarantine_{namespace}.bin")
        self._decode_quarantine_path = path
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(descriptor, len(self._records))
        self._decode_quarantine = mmap.mmap(descriptor, len(self._records))
        os.close(descriptor)

    def _task_diverse_targets(self, group_bounds) -> tuple[int, ...]:
        groups_by_source = {}
        for _, _, group_index in group_bounds:
            source_index = self._group_source_indices[group_index]
            groups_by_source.setdefault(source_index, []).append(group_index)
        largest_task_count = max(len(groups) for groups in groups_by_source.values())
        base_source_budget = largest_task_count * self.samples_per_task
        targets_by_group = {}
        for source_index, groups in groups_by_source.items():
            weight = self.sources[source_index].weight
            source_budget = max(len(groups), round(base_source_budget * weight))
            per_group, remainder = divmod(source_budget, len(groups))
            for position, group_index in enumerate(groups):
                targets_by_group[group_index] = per_group + int(position < remainder)
        return tuple(targets_by_group[group] for _, _, group in group_bounds)

    def _project_boundary(self, boundary: int) -> int:
        if self._length == self._full_length:
            return boundary
        if self._length == 1:
            return int(boundary > 0)
        return (boundary * (self._length - 1) + self._full_length - 2) // (
            self._full_length - 1
        )

    def _project_group_spans(self, group_bounds) -> tuple[tuple[int, int, int], ...]:
        spans = []
        for start, end, group_index in group_bounds:
            left, right = self._project_boundary(start), self._project_boundary(end)
            if left < right:
                spans.append((left, right, group_index))
        if not spans or spans[0][0] != 0 or spans[-1][1] != self._length:
            raise ValueError("projected multisource groups do not cover the dataset")
        return tuple(spans)

    def __len__(self) -> int:
        return self._length

    def _full_index(self, index: int) -> int:
        if not 0 <= index < self._length:
            raise IndexError(index)
        if self._length == self._full_length:
            return index
        if self._length == 1:
            return 0
        return index * (self._full_length - 1) // (self._length - 1)

    def _locate(self, index: int) -> tuple[int, _EpisodeRecord, int]:
        full_index = self._full_index(index)
        record_index = bisect.bisect_right(self._prefix, full_index)
        previous = self._prefix[record_index - 1] if record_index else 0
        return record_index, self._records[record_index], full_index - previous

    def _replacement_records(
        self, record_index: int, sample_index: int
    ) -> tuple[int, ...]:
        record = self._records[record_index]
        candidates = [record_index]
        seen_records = {record_index}
        pools = (
            (self._group_replacement_records[record.group_index], 16),
            (self._source_replacement_records[record.source_index], 64),
            (self._all_replacement_records, 64),
        )
        for scope_index, (pool, scope_limit) in enumerate(pools):
            if len(pool) < 2:
                continue
            start = _stable_integer(
                "decode-replacement-start", self.seed, sample_index, scope_index
            ) % len(pool)
            step = _stable_integer(
                "decode-replacement-step", self.seed, sample_index, scope_index
            ) % len(pool) or 1
            while math.gcd(step, len(pool)) != 1:
                step = step % len(pool) + 1
            accepted = 0
            for offset in range(len(pool)):
                candidate = pool[(start + offset * step) % len(pool)]
                if candidate in seen_records:
                    continue
                seen_records.add(candidate)
                candidates.append(candidate)
                accepted += 1
                if accepted == scope_limit:
                    break
        return tuple(candidates)

    def _quarantine_decode_failure(
        self, record_index: int, error: VideoDecodeError
    ) -> None:
        record = self._records[record_index]
        affected = (
            self._path_replacement_records[record.path]
            if error.path_unusable
            else (record_index,)
        )
        for affected_index in affected:
            self._decode_quarantine[affected_index] = 1

    @staticmethod
    def _decode_cache(path: str, indices: torch.Tensor) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        rgb = payload["rgb"]
        offsets = rgb["jpeg_offsets"].long()
        return torch.stack(
            [
                decode_jpeg(
                    rgb["jpeg_bytes"][int(offsets[index]) : int(offsets[index + 1])],
                    mode=ImageReadMode.RGB,
                )
                for index in indices.tolist()
            ]
        )

    def _decode(
        self, record: _EpisodeRecord, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        absolute = indices + record.frame_offset
        if record.adapter == "rgb_episode_cache":
            frames = self._decode_cache(record.path, absolute).permute(0, 2, 3, 1)
        else:
            frames = decode_video_frames(record.path, absolute, record.fps)
        return square_dino_rgb(frames, self.rgb_height)

    def _sample_record(
        self,
        record: _EpisodeRecord,
        ordinal: int,
        chunk_length: int,
    ) -> dict[str, torch.Tensor]:
        strides = sorted(
            {
                max(1, round(milliseconds * record.fps / 1000.0))
                for milliseconds in self.temporal_step_ms
            }
        )
        valid_strides = [
            stride for stride in strides if (chunk_length - 1) * stride < record.frame_count
        ]
        if not valid_strides:
            raise ValueError("episode cannot supply the requested multisource clip")
        stride = valid_strides[
            _stable_integer(
                self.seed, record.source_index, record.episode_index, ordinal, chunk_length
            ) % len(valid_strides)
        ]
        span = (chunk_length - 1) * stride
        maximum_start = record.frame_count - 1 - span
        fraction = ordinal / max(record.start_count - 1, 1)
        start = round(fraction * maximum_start)
        controls = start + torch.arange(chunk_length, dtype=torch.long) * stride
        rgb, valid = self._decode(record, controls)
        return {
            "video_rgb": rgb,
            "video_pixel_valid": valid,
            "observation_mask": torch.ones(chunk_length, dtype=torch.bool),
            "frame_times": torch.arange(chunk_length).float() * stride / record.fps,
            "control_indices": controls,
            "sequence_index": torch.tensor(record.sequence_index, dtype=torch.long),
            "source_index": torch.tensor(record.source_index, dtype=torch.long),
            "task_group_index": torch.tensor(record.group_index, dtype=torch.long),
            "chunk_length": torch.tensor(chunk_length, dtype=torch.long),
            "temporal_stride": torch.tensor(stride, dtype=torch.long),
            "temporal_step_seconds": torch.tensor(stride / record.fps),
        }

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        base_index, chunk_length = (
            index if isinstance(index, tuple) else (index, self.dynamic_history_lengths[-1])
        )
        if chunk_length not in self.dynamic_history_lengths:
            raise ValueError(f"unsupported v52 chunk length: {chunk_length}")
        record_index, original, ordinal = self._locate(int(base_index))
        last_error = None
        for replacement_index in self._replacement_records(
            record_index, int(base_index)
        ):
            record = self._records[replacement_index]
            if self._decode_quarantine[replacement_index]:
                continue
            replacement_ordinal = (
                ordinal
                if replacement_index == record_index
                else _stable_integer(
                    "decode-replacement-ordinal",
                    self.seed,
                    int(base_index),
                    record.episode_index,
                )
                % record.start_count
            )
            try:
                sample = self._sample_record(record, replacement_ordinal, chunk_length)
            except VideoDecodeError as error:
                self._quarantine_decode_failure(replacement_index, error)
                last_error = error
                continue
            sample["decode_replaced"] = torch.tensor(
                replacement_index != record_index, dtype=torch.bool
            )
            sample["requested_sequence_index"] = torch.tensor(
                original.sequence_index, dtype=torch.long
            )
            return sample
        raise VideoDecodeError(
            f"no decodable replacement for source={original.source_index} "
            f"group={original.group_index}; last error: {last_error}"
        )


MultiSourceRobotVideoDataset = MultiSourcePointTrackObjectVideoDataset
