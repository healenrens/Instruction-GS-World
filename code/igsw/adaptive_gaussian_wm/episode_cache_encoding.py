"""DINO projection and packed RGB helpers for dense episode caches."""
from __future__ import annotations

from concurrent.futures import Executor, Future

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

from .rgb_supervision import resize_and_pad_rgb


def projection_matrix(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("DINO projection dimension cannot exceed backbone dimension")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def encoded_feature_sequence(
    extractor,
    projection: torch.Tensor | None,
    frames,
) -> torch.Tensor:
    """Encode normalized backbone features, optionally through a legacy projection."""
    grid, _ = extractor.grid_batch(frames)
    grid = F.normalize(grid, dim=-1)
    if projection is not None:
        grid = F.normalize(grid @ projection, dim=-1)
    return grid.to(torch.bfloat16).cpu()


def decode_source_frame(dataset, index: int) -> torch.Tensor:
    encoded = torch.from_numpy(
        np.frombuffer(bytes(dataset[index]), dtype=np.uint8).copy()
    )
    return decode_jpeg(encoded, mode=ImageReadMode.RGB).permute(1, 2, 0)


def decode_source_batch(dataset, start: int, end: int) -> torch.Tensor:
    if not 0 <= start < end <= len(dataset):
        raise ValueError("invalid source JPEG batch")
    encoded = [
        torch.from_numpy(
            np.frombuffer(bytes(dataset[index]), dtype=np.uint8).copy()
        )
        for index in range(start, end)
    ]
    decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB)
    if not isinstance(decoded, list) or len(decoded) != len(encoded):
        raise RuntimeError("batched JPEG decoder returned an invalid result")
    return torch.stack([frame.permute(1, 2, 0) for frame in decoded])


def packed_rgb_batch(
    frames: torch.Tensor,
    short_side: int,
    pad_multiple: int,
    jpeg_quality: int,
    executor: Executor | None = None,
) -> tuple[list[torch.Tensor | Future[torch.Tensor]], dict[str, int]]:
    resized, valid = resize_and_pad_rgb(frames, short_side, pad_multiple)
    content_height = int(valid[0].any(dim=-1).sum())
    content_width = int(valid[0].any(dim=-2).sum())
    content = resized[:, :, :content_height, :content_width]
    inputs = [frame.contiguous() for frame in content]
    encoded = (
        [
            executor.submit(encode_jpeg, frame, jpeg_quality)
            for frame in inputs
        ]
        if executor is not None
        else [
            encode_jpeg(frame, quality=jpeg_quality)
            for frame in inputs
        ]
    )
    return encoded, {
        "content_height": content_height,
        "content_width": content_width,
        "padded_height": int(resized.shape[-2]),
        "padded_width": int(resized.shape[-1]),
        "short_side": short_side,
        "pad_multiple": pad_multiple,
        "jpeg_quality": jpeg_quality,
    }


def pack_jpegs(
    encoded: list[torch.Tensor | Future[torch.Tensor]],
    metadata: dict[str, int],
) -> dict:
    values = [
        value.result() if isinstance(value, Future) else value
        for value in encoded
    ]
    if not values:
        raise ValueError("cannot pack an empty JPEG sequence")
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0))
    )
    return {
        "jpeg_bytes": torch.cat(values),
        "jpeg_offsets": offsets,
        **metadata,
    }
