"""Language-free multi-frame visual sequence dataset with physical timestamps."""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .sequence_contract import (
    CONTROL_HZ,
    EPISODE_MANIFEST_NAME,
    SEQUENCE_CACHE_VERSION,
    parse_anchor_indices,
    temporal_layout,
)


class WindowCausalVisualSequenceDataset(Dataset):
    """Expose visual history and multi-horizon targets without semantic conditions."""

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
    ):
        clip_paths = [
            path
            for path in sorted(glob.glob(os.path.join(cache_root, "*.pt")))
            if os.path.basename(path).endswith(f"_{split}.pt")
        ]
        if not clip_paths:
            raise ValueError(f"no {split} visual sequence caches in {cache_root}")
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.anchors = parse_anchor_indices(anchors)
        self.load_rgb = load_rgb
        self.explicit_goal = explicit_goal
        self.condition_store = None
        self.condition_dim = 0
        self.data_sha256 = ""
        self.contract_label = "language-free causal visual sequences"
        self.clip_indices = {
            path: index for index, path in enumerate(clip_paths)
        }
        examples = [
            (path, anchor)
            for path in clip_paths
            for anchor in self.anchors
        ]
        if max_items > 0 and max_items < len(examples):
            if max_items == 1:
                examples = [examples[0]]
            else:
                last = len(examples) - 1
                examples = [
                    examples[(index * last) // (max_items - 1)]
                    for index in range(max_items)
                ]
        self.examples = examples
        self.paths = [f"{path}#anchor={anchor}" for path, anchor in examples]

        first = self._load_cache(examples[0][0])
        self._validate_cache(first, examples[0][0])
        features = first["dino"]
        self.feature_dim = int(first["feature_dim"])
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
                "sequence RGB cache and runtime resize configuration differ"
            )
        self.rgb_height = int(rgb_meta["padded_height"])
        self.rgb_width = int(rgb_meta["padded_width"])

    @staticmethod
    def _load_cache(path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def _validate_cache(self, cache: dict, path: str) -> None:
        if cache.get("sequence_version") != SEQUENCE_CACHE_VERSION:
            raise ValueError(f"visual sequence version mismatch: {path}")
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
            raise ValueError(f"semantic fields leaked into sequence cache: {present}")
        features = cache.get("dino")
        if (
            not torch.is_tensor(features)
            or features.ndim != 4
            or features.shape[-1] != int(cache.get("feature_dim", -1))
        ):
            raise ValueError(f"invalid sequence DINO tensor: {path}")
        controls = cache.get("frame_control_indices")
        if (
            not torch.is_tensor(controls)
            or controls.ndim != 1
            or len(controls) != len(features)
            or not bool((controls[1:] > controls[:-1]).all())
        ):
            raise ValueError(f"invalid sequence control timestamps: {path}")
        if abs(float(cache.get("control_hz", 0.0)) - CONTROL_HZ) > 1e-9:
            raise ValueError(f"control frequency mismatch: {path}")
        rgb = cache.get("rgb")
        if not isinstance(rgb, dict) or len(rgb.get("jpeg_frames", ())) != len(features):
            raise ValueError(f"invalid sequence RGB cache: {path}")

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(features.float(), (features.shape[-1],))

    def _decode_rgb(
        self,
        rgb_cache: dict,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_height = int(rgb_cache["content_height"])
        content_width = int(rgb_cache["content_width"])
        decoded = torch.stack(
            [
                decode_jpeg(
                    rgb_cache["jpeg_frames"][int(index)],
                    mode=ImageReadMode.RGB,
                )
                for index in indices
            ]
        )
        if decoded.shape[-2:] != (content_height, content_width):
            raise ValueError("decoded sequence RGB shape differs from cache metadata")
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
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, anchor = self.examples[index]
        cache = self._load_cache(path)
        self._validate_cache(cache, path)
        history_index, future_index = temporal_layout(
            len(cache["dino"]),
            anchor,
            self.history_frames,
            self.future_frames,
        )
        controls = cache["frame_control_indices"].long()
        anchor_control = controls[anchor]
        relative_seconds = (controls - anchor_control).float() / float(
            cache["control_hz"]
        )
        history = self._normalize(cache["dino"][history_index]).flatten(1, 2)
        future = self._normalize(cache["dino"][future_index]).flatten(1, 2)
        valid = torch.ones(
            len(self.coordinates),
            dtype=torch.bool,
        )
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
            "history_control_indices": controls[history_index],
            "future_control_indices": controls[future_index],
            "sequence_index": torch.tensor(
                self.clip_indices[path],
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
                goal_control_index=controls[goal_index],
            )
        if self.load_rgb:
            history_rgb, history_rgb_valid = self._decode_rgb(
                cache["rgb"],
                history_index,
            )
            future_rgb, future_rgb_valid = self._decode_rgb(
                cache["rgb"],
                future_index,
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
        return result


def CausalVisualSequenceDataset(
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
):
    """Select the versioned window or dense-episode sequence backend."""
    arguments = dict(
        cache_root=cache_root,
        split=split,
        history_frames=history_frames,
        future_frames=future_frames,
        anchors=anchors,
        max_items=max_items,
        load_rgb=load_rgb,
        explicit_goal=explicit_goal,
        rgb_short_side=rgb_short_side,
        rgb_pad_multiple=rgb_pad_multiple,
    )
    if os.path.isfile(os.path.join(cache_root, EPISODE_MANIFEST_NAME)):
        from .episode_sequence_dataset import CausalVisualEpisodeDataset

        return CausalVisualEpisodeDataset(**arguments)
    return WindowCausalVisualSequenceDataset(**arguments)
