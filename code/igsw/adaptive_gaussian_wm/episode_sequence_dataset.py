"""Dense causal samples from memory-mapped, language-free episode caches."""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .sequence_contract import (
    CONTROL_HZ,
    EPISODE_CACHE_VERSION,
    EPISODE_MANIFEST_NAME,
    EXPECTED_FRAME_COUNT,
    GROUP_SAMPLER_VERSION,
    control_frame_indices,
    parse_anchor_indices,
    parse_control_windows,
    temporal_layout,
)
from .group_balanced_sampler import sqrt_coverage_targets
from .teacher_sidecar import TeacherSidecarStore


@dataclass(frozen=True)
class _EpisodeWindowRecord:
    path: str
    episode_index: int
    sampling_group: str
    frame_count: int
    window: int
    regular_starts: int
    include_tail: bool

    @property
    def start_count(self) -> int:
        return self.regular_starts + int(self.include_tail)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CausalVisualEpisodeDataset(Dataset):
    """Enumerate dense multi-duration windows without duplicating episode frames."""

    def __init__(
        self,
        cache_root: str,
        split: str,
        history_frames: int = 4,
        future_frames: int = 4,
        anchors: str = "3,5,8",
        max_items: int = 0,
        load_rgb: bool = False,
        explicit_goal: bool = False,
        rgb_short_side: int = 256,
        rgb_pad_multiple: int = 16,
        teacher_sidecar: str = "",
    ):
        manifest_path = os.path.join(cache_root, EPISODE_MANIFEST_NAME)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self._validate_manifest(manifest, manifest_path)
        self.data_sha256 = _file_sha256(manifest_path)
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.anchors = parse_anchor_indices(anchors)
        for anchor in self.anchors:
            temporal_layout(
                EXPECTED_FRAME_COUNT,
                anchor,
                history_frames,
                future_frames,
            )
        self.window_lengths = parse_control_windows(
            manifest["sampling"]["window_lengths"]
        )
        self.sample_stride = int(manifest["sampling"]["sample_stride"])
        if self.sample_stride < 1:
            raise ValueError("episode sample stride must be positive")
        self.load_rgb = load_rgb
        self.explicit_goal = explicit_goal
        group_balance = manifest["sampling"].get("group_balance", "none")
        if group_balance not in ("none", "task_sqrt_coverage"):
            raise ValueError(f"unknown episode group balance: {group_balance}")
        if (
            group_balance != "none"
            and manifest["sampling"].get("group_sampler_version")
            != GROUP_SAMPLER_VERSION
        ):
            raise ValueError("episode group sampler version mismatch")
        if (
            group_balance == "task_sqrt_coverage"
            and manifest["sampling"].get("group_sampling_temperature") != 0.5
        ):
            raise ValueError("episode task sampling temperature mismatch")
        self.condition_store = None
        self.condition_dim = 0
        self.feature_contract = str(manifest["cache"].get("feature_contract", ""))
        self.contract_label = (
            "language-free dense causal visual episodes with backbone-native DINO"
        )

        episodes = [
            entry
            for entry in manifest["episodes"]
            if entry["split"] == split
        ]
        if not episodes:
            raise ValueError(f"no {split} episodes in {manifest_path}")
        records = []
        episode_paths = []
        grouped_episodes: dict[str, list[tuple[int, str, int]]] = {}
        for episode_index, entry in enumerate(episodes):
            path = os.path.join(cache_root, entry["filename"])
            if not os.path.isfile(path):
                raise ValueError(f"episode cache is missing: {path}")
            episode_paths.append(path)
            frame_count = int(entry["frame_count"])
            group = str(entry["sampling_group"])
            grouped_episodes.setdefault(group, []).append(
                (episode_index, path, frame_count)
            )
        for group, group_episodes in grouped_episodes.items():
            for episode_index, path, frame_count in group_episodes:
                for window in self.window_lengths:
                    last_start = frame_count - 1 - window
                    if last_start < 0:
                        continue
                    regular_starts = last_start // self.sample_stride + 1
                    records.append(
                        _EpisodeWindowRecord(
                            path=path,
                            episode_index=episode_index,
                            sampling_group=group,
                            frame_count=frame_count,
                            window=window,
                            regular_starts=regular_starts,
                            include_tail=(
                                last_start % self.sample_stride != 0
                            ),
                        )
                    )
        if not records:
            raise ValueError("episode cache has no valid temporal windows")
        self._records = records
        self._prefix = []
        group_spans: dict[str, list[int]] = {}
        total = 0
        for record in records:
            start = total
            total += record.start_count * len(self.anchors)
            self._prefix.append(total)
            span = group_spans.get(record.sampling_group)
            if span is None:
                group_spans[record.sampling_group] = [start, total]
            elif span[1] == start:
                span[1] = total
            else:
                raise ValueError("episode sampling group is not contiguous")
        self._full_length = total
        self._selected_length = (
            max_items if 0 < max_items < total else total
        )
        selected_groups = []
        for name, (start, end) in group_spans.items():
            if self._selected_length == self._full_length:
                selected_start, selected_end = start, end
            elif self._selected_length == 1:
                selected_start = 0
                selected_end = int(start == 0)
            else:
                numerator = self._selected_length - 1
                denominator = self._full_length - 1
                selected_start = (
                    start * numerator + denominator - 1
                ) // denominator
                selected_end = (
                    end * numerator + denominator - 1
                ) // denominator
            if selected_start < selected_end:
                selected_groups.append((name, selected_start, selected_end))
        self.sampling_group_names = tuple(
            name for name, _, _ in selected_groups
        )
        self.sampling_group_spans = tuple(
            (start, end) for _, start, end in selected_groups
        )
        self.balance_sampling = group_balance == "task_sqrt_coverage"
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
            raise ValueError("selected sampling groups do not partition the dataset")
        self.sampling_group_targets = sqrt_coverage_targets(
            tuple(end - start for start, end in self.sampling_group_spans)
        )
        self.minimum_balanced_samples = sum(self.sampling_group_targets)
        self.paths = [
            (
                f"{record.path}#window={record.window}"
                f"#starts={record.start_count}"
                f"#anchors={','.join(map(str, self.anchors))}"
                f"#group={record.sampling_group}"
            )
            for record in records
        ]

        first = self._load_cache(episode_paths[0])
        self._validate_cache(first, episode_paths[0])
        features = first["dino"]
        self.feature_dim = int(first["feature_dim"])
        if first.get("feature_contract") != self.feature_contract:
            raise ValueError("episode feature contract differs from manifest")
        self.grid_height = int(features.shape[1])
        self.grid_width = int(features.shape[2])
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.grid_height),
            torch.linspace(-1.0, 1.0, self.grid_width),
            indexing="ij",
        )
        self.coordinates = torch.stack((x, y), dim=-1).reshape(-1, 2)
        rgb_meta = first["rgb"]
        if (
            int(rgb_meta["short_side"]) != rgb_short_side
            or int(rgb_meta["pad_multiple"]) != rgb_pad_multiple
        ):
            raise ValueError(
                "episode RGB cache and runtime resize configuration differ"
            )
        self.rgb_height = int(rgb_meta["padded_height"])
        self.rgb_width = int(rgb_meta["padded_width"])
        self.teacher_sidecar = (
            TeacherSidecarStore(
                teacher_sidecar,
                manifest,
                self.data_sha256,
                self.grid_height,
                self.grid_width,
            )
            if teacher_sidecar
            else None
        )
        self.teacher_sidecar_sha256 = (
            self.teacher_sidecar.manifest_sha256 if self.teacher_sidecar else ""
        )
        if self.teacher_sidecar is not None:
            self.paths.extend(self.teacher_sidecar.paths)

    @staticmethod
    def _validate_manifest(manifest: dict, path: str) -> None:
        if manifest.get("episode_cache_version") != EPISODE_CACHE_VERSION:
            raise ValueError(f"episode manifest version mismatch: {path}")
        if manifest.get("complete") is not True:
            raise ValueError(f"episode cache is incomplete: {path}")
        if abs(float(manifest.get("control_hz", 0.0)) - CONTROL_HZ) > 1e-9:
            raise ValueError(f"episode manifest control frequency mismatch: {path}")
        if int(manifest.get("sample_frame_count", 0)) != EXPECTED_FRAME_COUNT:
            raise ValueError(f"episode manifest sample frame count mismatch: {path}")
        if not isinstance(manifest.get("episodes"), list):
            raise ValueError(f"episode manifest has no episode index: {path}")
        sampling = manifest.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError(f"episode manifest has no sampling contract: {path}")
        cache = manifest.get("cache")
        if not isinstance(cache, dict) or cache.get("feature_contract") != (
            "backbone_native"
        ):
            raise ValueError(f"episode manifest is not backbone-native DINO: {path}")

    @staticmethod
    def _load_cache(path: str) -> dict:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

    @staticmethod
    def _validate_cache(cache: dict, path: str) -> None:
        if cache.get("episode_version") != EPISODE_CACHE_VERSION:
            raise ValueError(f"visual episode version mismatch: {path}")
        if cache.get("feature_contract") != "backbone_native":
            raise ValueError(f"visual episode is not backbone-native DINO: {path}")
        if abs(float(cache.get("control_hz", 0.0)) - CONTROL_HZ) > 1e-9:
            raise ValueError(f"visual episode control frequency mismatch: {path}")
        forbidden = {
            "instruction",
            "condition_feature",
            "condition_tokens",
            "task",
            "task_index",
            "language",
        }
        present = sorted(forbidden.intersection(cache))
        if present:
            raise ValueError(f"semantic fields leaked into episode cache: {present}")
        features = cache.get("dino")
        controls = cache.get("frame_control_indices")
        if (
            not torch.is_tensor(features)
            or features.ndim != 4
            or features.shape[-1] != int(cache.get("feature_dim", -1))
            or not torch.is_tensor(controls)
            or controls.shape != (len(features),)
            or not torch.equal(
                controls,
                torch.arange(len(features), dtype=controls.dtype),
            )
        ):
            raise ValueError(f"invalid episode feature/timestamp tensors: {path}")
        rgb = cache.get("rgb")
        if not isinstance(rgb, dict):
            raise ValueError(f"invalid episode RGB metadata: {path}")
        blob = rgb.get("jpeg_bytes")
        offsets = rgb.get("jpeg_offsets")
        if (
            not torch.is_tensor(blob)
            or blob.dtype != torch.uint8
            or blob.ndim != 1
            or not torch.is_tensor(offsets)
            or offsets.shape != (len(features) + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(blob)
            or not bool((offsets[1:] > offsets[:-1]).all())
        ):
            raise ValueError(f"invalid packed episode JPEG storage: {path}")

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(features.float(), (features.shape[-1],))
        return normalized.to(features.dtype)

    def _full_index(self, index: int) -> int:
        if not 0 <= index < self._selected_length:
            raise IndexError(index)
        if self._selected_length == self._full_length:
            return index
        if self._selected_length == 1:
            return 0
        return (
            index * (self._full_length - 1)
        ) // (self._selected_length - 1)

    def _locate(self, index: int) -> tuple[_EpisodeWindowRecord, int, int]:
        full_index = self._full_index(index)
        record_index = bisect.bisect_right(self._prefix, full_index)
        previous = self._prefix[record_index - 1] if record_index else 0
        local = full_index - previous
        record = self._records[record_index]
        start_ordinal, anchor_offset = divmod(local, len(self.anchors))
        if start_ordinal < record.regular_starts:
            start = start_ordinal * self.sample_stride
        else:
            start = record.frame_count - 1 - record.window
        return record, start, self.anchors[anchor_offset]

    def _decode_rgb(
        self,
        rgb_cache: dict,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_height = int(rgb_cache["content_height"])
        content_width = int(rgb_cache["content_width"])
        blob = rgb_cache["jpeg_bytes"]
        offsets = rgb_cache["jpeg_offsets"].long()
        decoded = torch.stack(
            [
                decode_jpeg(
                    blob[int(offsets[index]) : int(offsets[index + 1])],
                    mode=ImageReadMode.RGB,
                )
                for index in indices.tolist()
            ]
        )
        if decoded.shape[-2:] != (content_height, content_width):
            raise ValueError("decoded episode RGB shape differs from metadata")
        output = torch.zeros(
            len(indices),
            3,
            self.rgb_height,
            self.rgb_width,
            dtype=torch.uint8,
        )
        output[:, :, :content_height, :content_width] = decoded
        valid = torch.zeros(
            len(indices),
            self.rgb_height,
            self.rgb_width,
            dtype=torch.bool,
        )
        valid[:, :content_height, :content_width] = True
        return output, valid

    def __len__(self) -> int:
        return self._selected_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record, start, anchor = self._locate(index)
        cache = self._load_cache(record.path)
        self._validate_cache(cache, record.path)
        if len(cache["dino"]) != record.frame_count:
            raise ValueError("episode frame count differs from manifest")
        sampled_controls = control_frame_indices(
            start,
            record.window,
            EXPECTED_FRAME_COUNT,
        )
        history_index, future_index = temporal_layout(
            EXPECTED_FRAME_COUNT,
            anchor,
            self.history_frames,
            self.future_frames,
        )
        anchor_control = sampled_controls[anchor]
        relative_seconds = (
            sampled_controls - anchor_control
        ).float() / float(CONTROL_HZ)
        sampled_features = cache["dino"][sampled_controls]
        history = self._normalize(sampled_features[history_index]).flatten(1, 2)
        future = self._normalize(sampled_features[future_index]).flatten(1, 2)
        valid = torch.ones(len(self.coordinates), dtype=torch.bool)
        result = {
            "history_features": history,
            "history_coordinates": self.coordinates[None].expand(
                len(history_index), -1, -1
            ),
            "history_valid": valid[None].expand(len(history_index), -1),
            "history_times": relative_seconds[history_index],
            "future_features": future,
            "future_coordinates": self.coordinates[None].expand(
                len(future_index), -1, -1
            ),
            "future_valid": valid[None].expand(len(future_index), -1),
            "future_times": relative_seconds[future_index],
            "feature_grid_hw": torch.tensor(
                [self.grid_height, self.grid_width],
                dtype=torch.long,
            ),
            "anchor_frame_index": torch.tensor(anchor, dtype=torch.long),
            "history_frame_indices": history_index,
            "future_frame_indices": future_index,
            "history_control_indices": sampled_controls[history_index],
            "future_control_indices": sampled_controls[future_index],
            "sequence_index": torch.tensor(
                record.episode_index,
                dtype=torch.long,
            ),
            "window_start_control": torch.tensor(start, dtype=torch.long),
            "window_control_span": torch.tensor(
                record.window,
                dtype=torch.long,
            ),
        }
        if self.explicit_goal:
            goal_index = future_index[-1]
            result.update(
                goal_features=future[-1],
                goal_coordinates=self.coordinates,
                goal_valid=valid,
                goal_time=relative_seconds[goal_index],
                goal_frame_index=goal_index,
                goal_control_index=sampled_controls[goal_index],
            )
        if self.load_rgb:
            history_rgb, history_rgb_valid = self._decode_rgb(
                cache["rgb"],
                sampled_controls[history_index],
            )
            future_rgb, future_rgb_valid = self._decode_rgb(
                cache["rgb"],
                sampled_controls[future_index],
            )
            result.update(
                history_rgb=history_rgb,
                history_rgb_valid=history_rgb_valid,
                future_rgb=future_rgb,
                future_rgb_valid=future_rgb_valid,
            )
            if self.explicit_goal:
                result.update(
                    goal_rgb=future_rgb[-1],
                    goal_rgb_valid=future_rgb_valid[-1],
                )
        if self.teacher_sidecar is not None:
            result.update(
                self.teacher_sidecar.sample(
                    record.path,
                    sampled_controls,
                    history_index,
                    future_index,
                )
            )
        return result
