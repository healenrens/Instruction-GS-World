"""Five-frame causal video clips layered onto the v39 temporal sampler."""
from __future__ import annotations

import torch

from .dynamic_dual_horizon_dataset import (
    DynamicDualHorizonEpisodeDataset,
    _inclusive_indices,
)
from .rgb_episode_cache_contract import validate_episode_payload


DUAL_ENCODER_TEMPORAL_CONTRACT = "dynamic_dual_horizon_video_v2"


def _left_pad_history(value: torch.Tensor, frames: int) -> torch.Tensor:
    if value.shape[0] > frames:
        raise ValueError("observed history exceeds the video clip length")
    if value.shape[0] == frames:
        return value
    padding = value[:1].expand(frames - value.shape[0], *value.shape[1:])
    return torch.cat((padding, value), dim=0)


def _assemble_target_clip(
    start: torch.Tensor,
    middle: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    if start.shape != end.shape or middle.shape[0] != 3:
        raise ValueError("dual-encoder target clip has an invalid shape")
    return torch.cat((start[None], middle, end[None]), dim=0)


class DualEncoderDynamicEpisodeDataset(DynamicDualHorizonEpisodeDataset):
    """Add motion-bearing clips without changing online DINO observations."""

    def __init__(self, *args, video_clip_frames: int = 5, **kwargs):
        if video_clip_frames != 5:
            raise ValueError("Wan2.2 v44 requires exactly five-frame clips")
        if "minimum_goal_tail_frames" in kwargs:
            raise ValueError("v44 fixes its minimum tail to a unique VAE clip")
        kwargs["minimum_goal_tail_frames"] = video_clip_frames - 1
        super().__init__(*args, **kwargs)
        self.video_clip_frames = video_clip_frames
        self.temporal_contract = DUAL_ENCODER_TEMPORAL_CONTRACT
        self.contract_label = (
            "language-free 30 Hz dual-encoder history/short/goal clips"
        )

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        result = super().__getitem__(index)
        base_index = index[0] if isinstance(index, tuple) else index
        record, anchor = self._locate_dynamic(int(base_index))
        cache = self._load_cache(record.path)
        validate_episode_payload(
            cache,
            record.path,
            self._entries_by_path[record.path],
            self._manifest,
        )
        history_rgb = _left_pad_history(
            result["history_jit_rgb"], self.video_clip_frames
        )
        history_valid = _left_pad_history(
            result["history_jit_valid"], self.video_clip_frames
        )
        history_controls = _left_pad_history(
            result["history_control_indices"], self.video_clip_frames
        )

        short_end = anchor + self.short_horizon_frames
        short_controls = _inclusive_indices(anchor, short_end, self.video_clip_frames)
        short_middle_rgb, short_middle_valid = self._decode_rgb(
            cache["rgb"], short_controls[1:-1]
        )
        short_rgb = _assemble_target_clip(
            result["history_jit_rgb"][-1],
            short_middle_rgb,
            result["future_jit_rgb"][0],
        )
        short_valid = _assemble_target_clip(
            result["history_jit_valid"][-1],
            short_middle_valid,
            result["future_jit_valid"][0],
        )

        goal_end = int(result["future_control_indices"][1])
        goal_controls = _inclusive_indices(
            short_end, goal_end, self.video_clip_frames
        )
        goal_middle_rgb, goal_middle_valid = self._decode_rgb(
            cache["rgb"], goal_controls[1:-1]
        )
        goal_rgb = _assemble_target_clip(
            result["future_jit_rgb"][0],
            goal_middle_rgb,
            result["future_jit_rgb"][1],
        )
        goal_valid = _assemble_target_clip(
            result["future_jit_valid"][0],
            goal_middle_valid,
            result["future_jit_valid"][1],
        )
        result.update(
            history_video_rgb=history_rgb,
            history_video_valid=history_valid,
            history_video_control_indices=history_controls,
            short_video_rgb=short_rgb,
            short_video_valid=short_valid,
            short_video_control_indices=short_controls,
            goal_video_rgb=goal_rgb,
            goal_video_valid=goal_valid,
            goal_video_control_indices=goal_controls,
        )
        return result
