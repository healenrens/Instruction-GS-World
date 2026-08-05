"""Causal dynamic-history samples with fixed-short and terminal targets."""

from __future__ import annotations

import bisect
import hashlib
import os
from dataclasses import dataclass

import torch

from .episode_sequence_dataset import CausalVisualEpisodeDataset
from .group_balanced_sampler import sqrt_coverage_targets
from .rgb_episode_cache_contract import validate_episode_payload


DYNAMIC_DUAL_HORIZON_CONTRACT = "dynamic_dual_horizon_v1"


@dataclass(frozen=True)
class _DynamicEpisodeRecord:
    path: str
    episode_index: int
    sampling_group: str
    frame_count: int
    first_anchor: int
    last_anchor: int
    regular_anchors: int
    include_tail: bool

    @property
    def anchor_count(self) -> int:
        return self.regular_anchors + int(self.include_tail)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not values or values[0] <= 0:
        raise ValueError("history spans must be positive integers")
    return values


def _stable_choice(values: tuple[int, ...], *parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    token = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return values[token % len(values)]


def _inclusive_indices(start: int, end: int, count: int) -> torch.Tensor:
    if count < 1 or end < start:
        raise ValueError("invalid causal history request")
    if count == 1:
        return torch.tensor([end], dtype=torch.long)
    span = end - start
    if span < count - 1:
        raise ValueError("history span cannot provide unique frames")
    values = [start + (index * span) // (count - 1) for index in range(count)]
    values[-1] = end
    result = torch.tensor(values, dtype=torch.long)
    if len(torch.unique(result)) != count:
        raise ValueError("dynamic history contains duplicate frames")
    return result


class DynamicDualHorizonEpisodeDataset(CausalVisualEpisodeDataset):
    """Sample history independently from fixed-short and terminal targets."""

    def __init__(
        self,
        cache_root: str,
        split: str,
        history_frames_min: int = 1,
        history_frames_max: int = 4,
        history_span_frames: str = "15,30,45",
        short_horizon_frames: int = 30,
        goal_query_seconds: float = 6.0,
        goal_tail_guard_frames: int = 0,
        goal_probe_frames: int = 3,
        minimum_goal_tail_frames: int = 1,
        max_items: int = 0,
        teacher_sidecar: str = "",
        feature_source: str = "jit",
    ):
        cache_root = os.path.abspath(cache_root)
        if feature_source != "jit":
            raise ValueError("dynamic dual-horizon training requires JIT DINO")
        if not 1 <= history_frames_min <= history_frames_max <= 4:
            raise ValueError("dynamic history length must stay within [1,4]")
        spans = _parse_positive_ints(history_span_frames)
        if spans[-1] < history_frames_max - 1:
            raise ValueError("largest history span cannot provide four frames")
        if short_horizon_frames < 1:
            raise ValueError("short horizon must be positive")
        if goal_tail_guard_frames < 0 or goal_probe_frames < 2:
            raise ValueError("invalid terminal target configuration")
        if minimum_goal_tail_frames < 1:
            raise ValueError("minimum goal tail must be positive")

        super().__init__(
            cache_root=cache_root,
            split=split,
            history_frames=4,
            future_frames=2,
            anchors="3",
            max_items=0,
            load_rgb=False,
            explicit_goal=False,
            teacher_sidecar=teacher_sidecar,
            feature_source=feature_source,
        )
        if not self.jit_dino:
            raise ValueError("dynamic dual-horizon data requires RGB-only episodes")
        short_seconds = short_horizon_frames / self.control_hz
        if goal_query_seconds <= short_seconds:
            raise ValueError("goal query scale must exceed the fixed short horizon")

        self.temporal_contract = DYNAMIC_DUAL_HORIZON_CONTRACT
        self.cache_root = os.path.abspath(cache_root)
        self.dynamic_history_lengths = tuple(
            range(history_frames_min, history_frames_max + 1)
        )
        self.history_frames_min = history_frames_min
        self.history_frames_max = history_frames_max
        self.history_frames = history_frames_max
        self.future_frames = 2
        self.history_span_frames = spans
        self.short_horizon_frames = short_horizon_frames
        self.goal_query_seconds = float(goal_query_seconds)
        self.minimum_goal_horizon_frames = (
            short_horizon_frames + minimum_goal_tail_frames
        )
        self.goal_tail_guard_frames = goal_tail_guard_frames
        self.goal_probe_frames = goal_probe_frames
        self.contract_label = (
            "language-free 30 Hz dynamic history with fixed-short and terminal targets"
        )
        self._build_dynamic_index(split, max_items)

    def _build_dynamic_index(self, split: str, max_items: int) -> None:
        episodes = [
            entry for entry in self._manifest["episodes"] if entry["split"] == split
        ]
        grouped: dict[str, list[tuple[int, str, int]]] = {}
        for episode_index, entry in enumerate(episodes):
            path = os.path.join(self.cache_root, entry["filename"])
            group = str(entry["sampling_group"])
            grouped.setdefault(group, []).append(
                (episode_index, path, int(entry["frame_count"]))
            )

        records: list[_DynamicEpisodeRecord] = []
        max_history_span = self.history_span_frames[-1]
        for group, group_episodes in grouped.items():
            for episode_index, path, frame_count in group_episodes:
                goal = frame_count - 1 - self.goal_tail_guard_frames
                first = max_history_span
                last = goal - self.minimum_goal_horizon_frames
                if last < first or goal - (self.goal_probe_frames - 1) < 0:
                    continue
                regular = (last - first) // self.sample_stride + 1
                records.append(
                    _DynamicEpisodeRecord(
                        path=path,
                        episode_index=episode_index,
                        sampling_group=group,
                        frame_count=frame_count,
                        first_anchor=first,
                        last_anchor=last,
                        regular_anchors=regular,
                        include_tail=((last - first) % self.sample_stride != 0),
                    )
                )
        if not records:
            raise ValueError("episode cache has no dynamic dual-horizon samples")

        self._dynamic_records = records
        self._dynamic_prefix: list[int] = []
        group_spans: dict[str, list[int]] = {}
        total = 0
        for record in records:
            start = total
            total += record.anchor_count
            self._dynamic_prefix.append(total)
            span = group_spans.get(record.sampling_group)
            if span is None:
                group_spans[record.sampling_group] = [start, total]
            elif span[1] == start:
                span[1] = total
            else:
                raise ValueError("dynamic sampling group is not contiguous")
        self._full_length = total
        self._selected_length = max_items if 0 < max_items < total else total
        selected = []
        for name, (start, end) in group_spans.items():
            selected_start = self._selected_index_boundary(start)
            selected_end = self._selected_index_boundary(end)
            if selected_start < selected_end:
                selected.append((name, selected_start, selected_end))
        self.sampling_group_names = tuple(name for name, _, _ in selected)
        self.sampling_group_spans = tuple((start, end) for _, start, end in selected)
        if (
            not self.sampling_group_spans
            or self.sampling_group_spans[0][0] != 0
            or self.sampling_group_spans[-1][1] != self._selected_length
            or any(
                left[1] != right[0]
                for left, right in zip(
                    self.sampling_group_spans,
                    self.sampling_group_spans[1:],
                )
            )
        ):
            raise ValueError("selected dynamic sampling groups do not partition data")
        self.sampling_group_targets = sqrt_coverage_targets(
            tuple(end - start for start, end in self.sampling_group_spans)
        )
        self.minimum_balanced_samples = sum(self.sampling_group_targets)
        self.paths = [
            (
                f"{record.path}#dynamic={record.anchor_count}"
                f"#short={self.short_horizon_frames}"
                f"#goal_guard={self.goal_tail_guard_frames}"
                f"#group={record.sampling_group}"
            )
            for record in records
        ]
        if self.teacher_sidecar is not None:
            self.paths.extend(self.teacher_sidecar.paths)

    def _selected_index_boundary(self, boundary: int) -> int:
        if self._selected_length == self._full_length:
            return boundary
        if self._selected_length == 1:
            return int(boundary > 0)
        numerator = self._selected_length - 1
        denominator = self._full_length - 1
        return (boundary * numerator + denominator - 1) // denominator

    def _locate_dynamic(self, index: int) -> tuple[_DynamicEpisodeRecord, int]:
        full_index = self._full_index(index)
        record_index = bisect.bisect_right(self._dynamic_prefix, full_index)
        previous = self._dynamic_prefix[record_index - 1] if record_index else 0
        ordinal = full_index - previous
        record = self._dynamic_records[record_index]
        anchor = (
            record.first_anchor + ordinal * self.sample_stride
            if ordinal < record.regular_anchors
            else record.last_anchor
        )
        return record, anchor

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        if isinstance(index, tuple):
            base_index, history_count = index
        else:
            base_index, history_count = index, self.history_frames_max
        if history_count not in self.dynamic_history_lengths:
            raise ValueError(f"unsupported dynamic history length: {history_count}")
        record, anchor = self._locate_dynamic(int(base_index))
        cache = self._load_cache(record.path)
        validate_episode_payload(
            cache,
            record.path,
            self._entries_by_path[record.path],
            self._manifest,
        )
        span = _stable_choice(
            self.history_span_frames,
            record.episode_index,
            anchor,
            history_count,
        )
        history_controls = _inclusive_indices(anchor - span, anchor, history_count)
        goal_control = record.frame_count - 1 - self.goal_tail_guard_frames
        future_controls = torch.tensor(
            [anchor + self.short_horizon_frames, goal_control], dtype=torch.long
        )
        short_seconds = self.short_horizon_frames / self.control_hz
        observation_times = (future_controls - anchor).float() / self.control_hz
        history_times = (history_controls - anchor).float() / self.control_hz
        history_rgb, history_valid = self._decode_rgb(
            cache["rgb"], history_controls
        )
        future_rgb, future_valid = self._decode_rgb(cache["rgb"], future_controls)
        probe_controls = torch.arange(
            goal_control - self.goal_probe_frames + 1,
            goal_control,
            dtype=torch.long,
        )
        probe_rgb, probe_valid = self._decode_rgb(cache["rgb"], probe_controls)
        result = {
            "history_jit_rgb": history_rgb,
            "future_jit_rgb": future_rgb,
            "goal_probe_jit_rgb": probe_rgb,
            "history_jit_valid": history_valid,
            "future_jit_valid": future_valid,
            "goal_probe_jit_valid": probe_valid,
            "history_times": history_times,
            "future_times": torch.tensor(
                [short_seconds, self.goal_query_seconds], dtype=torch.float32
            ),
            "future_observation_times": observation_times,
            "history_control_indices": history_controls,
            "future_control_indices": future_controls,
            "anchor_control_index": torch.tensor(anchor, dtype=torch.long),
            "history_length": torch.tensor(history_count, dtype=torch.long),
            "history_span_controls": torch.tensor(span, dtype=torch.long),
            "future_horizon_valid": torch.ones(2, dtype=torch.bool),
            "sequence_index": torch.tensor(record.episode_index, dtype=torch.long),
        }
        if self.teacher_sidecar is not None:
            sampled = torch.cat((history_controls, future_controls))
            result.update(
                self.teacher_sidecar.sample(
                    record.path,
                    sampled,
                    torch.arange(history_count),
                    torch.arange(history_count, history_count + 2),
                )
            )
        return result
