"""Integrity checks shared by dense episode cache generation and verification."""

from __future__ import annotations

import hashlib
import os

import torch

from .sequence_contract import EPISODE_CACHE_VERSION, EXPECTED_FRAME_COUNT


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
    control_hz = float(episode.get("control_hz", 0.0))
    if control_hz <= 0 or abs(float(cache.get("control_hz", 0.0)) - control_hz) > 1e-9:
        mismatches["control_hz"] = (cache.get("control_hz"), control_hz)
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


def validate_episode_manifest(manifest: dict, path: str) -> float:
    if manifest.get("episode_cache_version") != EPISODE_CACHE_VERSION:
        raise ValueError(f"episode manifest version mismatch: {path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"episode cache is incomplete: {path}")
    control_hz = float(manifest.get("control_hz", 0.0))
    if control_hz <= 0:
        raise ValueError(f"episode manifest control frequency is invalid: {path}")
    if int(manifest.get("sample_frame_count", 0)) != EXPECTED_FRAME_COUNT:
        raise ValueError(f"episode manifest sample frame count mismatch: {path}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"episode manifest has no episode index: {path}")
    if any(
        abs(float(episode.get("control_hz", 0.0)) - control_hz) > 1e-9
        for episode in episodes
    ):
        raise ValueError(f"episode manifest contains mixed frequencies: {path}")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError(f"episode manifest has no sampling contract: {path}")
    cache = manifest.get("cache")
    if not isinstance(cache, dict) or cache.get("feature_contract") != (
        "backbone_native"
    ):
        raise ValueError(f"episode manifest is not backbone-native DINO: {path}")
    return control_hz


def validate_episode_payload(cache: dict, path: str, control_hz: float) -> None:
    if cache.get("episode_version") != EPISODE_CACHE_VERSION:
        raise ValueError(f"visual episode version mismatch: {path}")
    if cache.get("feature_contract") != "backbone_native":
        raise ValueError(f"visual episode is not backbone-native DINO: {path}")
    if abs(float(cache.get("control_hz", 0.0)) - control_hz) > 1e-9:
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
        or not torch.equal(controls, torch.arange(len(features), dtype=controls.dtype))
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
