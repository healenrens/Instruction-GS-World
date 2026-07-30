"""Versioned visual-sequence and physical-time contract for RoboTwin clips."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SEQUENCE_CACHE_VERSION = "rt2_visual_sequence_dino_v1"
EPISODE_CACHE_VERSION = "rt2_visual_episode_dino_native_v2"
EPISODE_MANIFEST_NAME = "episode_manifest.json"
EPISODE_VERIFIED_NAME = "episode_manifest.verified.sha256"
GROUP_SAMPLER_VERSION = 2
CONTROL_HZ = 250.0 / 15.0
EXPECTED_FRAME_COUNT = 13
DEFAULT_SEQUENCE_ANCHORS = (3, 5, 8)
DEFAULT_EPISODE_WINDOWS = (25, 50, 75, 100)
RT2_HELDSEED_FRACTION = 0.20
RT2_HELDTASKS = ("handover_block", "place_object_basket", "stack_blocks_two")


def stable_rt2_episode_split(task: str, episode: int) -> str:
    if task in RT2_HELDTASKS:
        return "heldtask"
    import hashlib

    value = int(hashlib.md5(f"{task}:{episode}".encode()).hexdigest(), 16)
    return (
        "heldseed"
        if value % 1000 < int(RT2_HELDSEED_FRACTION * 1000)
        else "train"
    )


def parse_anchor_indices(text: str) -> tuple[int, ...]:
    anchors = tuple(sorted({int(value) for value in text.split(",") if value}))
    if not anchors:
        raise ValueError("sequence anchors cannot be empty")
    return anchors


def parse_control_windows(values) -> tuple[int, ...]:
    if isinstance(values, str):
        windows = tuple(
            sorted({int(value) for value in values.split(",") if value})
        )
    else:
        windows = tuple(sorted({int(value) for value in values}))
    if not windows or windows[0] < EXPECTED_FRAME_COUNT - 1:
        raise ValueError(
            "control windows must contain at least one span with 12 control steps"
        )
    return windows


def control_frame_indices(
    start: int,
    window: int,
    frame_count: int,
) -> torch.Tensor:
    """Reproduce np.linspace(...).astype(int) for non-negative control indices."""
    if start < 0 or window < 1 or frame_count < 2:
        raise ValueError("invalid source timing metadata")
    denominator = frame_count - 1
    offsets = [
        (index * window) // denominator
        for index in range(frame_count)
    ]
    offsets[-1] = window
    return torch.tensor(offsets, dtype=torch.long) + start


def temporal_layout(
    frame_count: int,
    anchor: int,
    history_frames: int,
    future_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select monotonic history and multi-horizon future indices around an anchor."""
    if history_frames < 1 or future_frames < 2:
        raise ValueError("sequence training requires history and multi-horizon future")
    if not history_frames - 1 <= anchor < frame_count - future_frames:
        raise ValueError(
            f"anchor {anchor} cannot support H={history_frames}, "
            f"Q={future_frames}, K={frame_count}"
        )

    def inclusive_indices(start: int, end: int, count: int) -> torch.Tensor:
        if count == 1:
            return torch.tensor([end], dtype=torch.long)
        span = end - start
        if span < count - 1:
            raise ValueError("temporal selection would contain duplicate frames")
        values = [
            start + (index * span) // (count - 1)
            for index in range(count)
        ]
        values[-1] = end
        result = torch.tensor(values, dtype=torch.long)
        if len(torch.unique(result)) != count:
            raise ValueError("temporal selection contains duplicate frames")
        return result

    history = inclusive_indices(0, anchor, history_frames)
    future = inclusive_indices(anchor + 1, frame_count - 1, future_frames)
    return history, future


def preprocess_vggt_rgb(
    frames: torch.Tensor,
    target_size: int = 518,
) -> torch.Tensor:
    """Match SpaTracker VGGT crop preprocessing for [T,H,W,3] uint8 frames."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("source RGB must have shape [T,H,W,3]")
    if frames.dtype != torch.uint8 or target_size < 14:
        raise ValueError("VGGT preprocessing requires uint8 RGB and target_size >= 14")
    height, width = frames.shape[1:3]
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    channels = frames.permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        channels,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
    if new_height > target_size:
        start = (new_height - target_size) // 2
        resized = resized[:, :, start : start + target_size]
    return (
        resized.round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def rgb_resize_shape(
    height: int,
    width: int,
    short_side: int,
    pad_multiple: int,
) -> tuple[int, int, int, int]:
    if min(height, width, short_side, pad_multiple) < 1:
        raise ValueError("invalid RGB shape configuration")
    scale = short_side / min(height, width)
    content_height = max(1, round(height * scale))
    content_width = max(1, round(width * scale))
    padded_height = math.ceil(content_height / pad_multiple) * pad_multiple
    padded_width = math.ceil(content_width / pad_multiple) * pad_multiple
    return content_height, content_width, padded_height, padded_width
