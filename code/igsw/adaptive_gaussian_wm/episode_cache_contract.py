"""Integrity checks shared by dense episode cache generation and verification."""
from __future__ import annotations

import hashlib
import os

import torch

from .sequence_contract import CONTROL_HZ, EPISODE_CACHE_VERSION


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_episode_cache_header(
    cache: dict,
    path: str,
    episode: dict,
    cache_contract: dict,
    projection_sha256: str | None,
) -> None:
    expected = {
        "episode_version": EPISODE_CACHE_VERSION,
        "source_name": episode["filename"],
        "split": episode["split"],
        "model": cache_contract["model"],
        "image_size": int(cache_contract["image_size"]),
        "feature_dim": int(cache_contract["feature_dim"]),
        "feature_contract": cache_contract["feature_contract"],
        "projection_seed": int(cache_contract["projection_seed"]),
    }
    mismatches = {
        name: (cache.get(name), value)
        for name, value in expected.items()
        if cache.get(name) != value
    }
    if abs(float(cache.get("control_hz", 0.0)) - CONTROL_HZ) > 1e-9:
        mismatches["control_hz"] = (cache.get("control_hz"), CONTROL_HZ)
    if (
        projection_sha256 is not None
        and cache.get("projection_sha256") != projection_sha256
    ):
        mismatches["projection_sha256"] = (
            cache.get("projection_sha256"),
            projection_sha256,
        )
    features = cache.get("dino")
    controls = cache.get("frame_control_indices")
    frame_count = int(episode["frame_count"])
    if (
        not torch.is_tensor(features)
        or features.ndim != 4
        or len(features) != frame_count
        or features.shape[-1] != int(cache_contract["feature_dim"])
    ):
        mismatches["dino"] = (
            getattr(features, "shape", None),
            (frame_count, "...", int(cache_contract["feature_dim"])),
        )
    if (
        not torch.is_tensor(controls)
        or controls.shape != (frame_count,)
        or not torch.equal(
            controls,
            torch.arange(frame_count, dtype=controls.dtype),
        )
    ):
        mismatches["frame_control_indices"] = (
            getattr(controls, "shape", None),
            (frame_count,),
        )
    rgb = cache.get("rgb")
    rgb_expected = {
        "short_side": int(cache_contract["rgb_short_side"]),
        "pad_multiple": int(cache_contract["rgb_pad_multiple"]),
        "jpeg_quality": int(cache_contract["jpeg_quality"]),
    }
    if not isinstance(rgb, dict):
        mismatches["rgb"] = (type(rgb).__name__, "dict")
    else:
        for name, value in rgb_expected.items():
            if rgb.get(name) != value:
                mismatches[f"rgb.{name}"] = (rgb.get(name), value)
    if mismatches:
        raise ValueError(f"episode cache header differs at {path}: {mismatches}")
    if os.path.getsize(path) <= 0:
        raise ValueError(f"episode cache is empty: {path}")
