"""Exact indexed MP4 decoding and the established 518px visual transform."""

from __future__ import annotations

import os

import torch
from .sequence_contract import preprocess_vggt_rgb


VIDEO_DECODER_CONTRACT = "pyav_single_thread_exact_index_v1"


class VideoDecodeError(ValueError):
    """The requested frames cannot be read from an otherwise valid index entry."""

    def __init__(self, message: str, *, path_unusable: bool = False):
        super().__init__(message)
        self.path_unusable = path_unusable


def decode_video_frames(path: str, indices: torch.Tensor, fps: float) -> torch.Tensor:
    import av

    if not os.path.isfile(path):
        raise VideoDecodeError(
            f"video payload is missing: {path}", path_unusable=True
        )
    wanted = [int(value) for value in indices.tolist()]
    targets = set(wanted)
    first, last = min(wanted), max(wanted)
    decoded = {}
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            stream.thread_type = "NONE"
            time_base = stream.time_base
            start_pts = int(stream.start_time or 0)
            seek_time = max(first / fps - 1.0, 0.0)
            container.seek(
                start_pts + int(seek_time / float(time_base)),
                backward=True,
                any_frame=False,
                stream=stream,
            )
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise VideoDecodeError(f"video frame has no timestamp: {path}")
                frame_index = int(
                    round(float((frame.pts - start_pts) * time_base) * fps)
                )
                if frame_index in targets:
                    decoded[frame_index] = torch.from_numpy(
                        frame.to_ndarray(format="rgb24")
                    )
                if frame_index >= last:
                    break
    except (av.error.FFmpegError, OSError) as error:
        raise VideoDecodeError(
            f"video decoder rejected {path}: {error}", path_unusable=True
        ) from error
    missing = [index for index in wanted if index not in decoded]
    if missing:
        raise VideoDecodeError(
            f"video decode missed frames {missing[:8]} in {path}"
        )
    return torch.stack([decoded[index] for index in wanted])


def square_dino_rgb(
    frames: torch.Tensor, target_size: int = 518
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = preprocess_vggt_rgb(frames, target_size)
    height, width = prepared.shape[1:3]
    if height > target_size or width > target_size:
        raise ValueError("preprocessed RGB exceeds the square visual contract")
    top = (target_size - height) // 2
    left = (target_size - width) // 2
    square = torch.zeros(
        len(prepared), 3, target_size, target_size, dtype=torch.uint8
    )
    square[:, :, top : top + height, left : left + width] = prepared.permute(0, 3, 1, 2)
    valid = torch.zeros(len(prepared), target_size, target_size, dtype=torch.bool)
    valid[:, top : top + height, left : left + width] = True
    return square, valid
