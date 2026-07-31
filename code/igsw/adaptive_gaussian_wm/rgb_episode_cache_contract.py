"""Contracts for RGB-only episode storage used by JIT DINO training."""

from __future__ import annotations

import hashlib
import os
import string

import torch

from .robotwin_lerobot_source import source_index_sha256
from .sequence_contract import (
    EXPECTED_FRAME_COUNT,
    GROUP_SAMPLER_VERSION,
    RT2_HELDSEED_FRACTION,
    RT2_HELDTASKS,
    parse_control_windows,
)


RGB_EPISODE_CACHE_VERSION = "rt2_visual_episode_rgb_jit_v1"
JIT_DINO_FEATURE_CONTRACT = "jit_backbone_native"
JIT_DINO_MODEL = "vit_large_patch14_dinov2.lvd142m"
JIT_DINO_IMAGE_SIZE = 518
JIT_DINO_FEATURE_DIM = 1024
PENDING_MANIFEST_NAME = "episode_manifest.pending.json"
SOURCE_INDEX_NAME = "episode_source_index.json"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def manifest_payload(
    episodes: list[dict],
    source_root: str,
    source_variants: tuple[str, ...],
    expected_source_fps: float,
    window_lengths: tuple[int, ...],
    sample_stride: int,
    jpeg_quality: int,
    complete: bool,
) -> dict:
    frequencies = {round(float(item["control_hz"]), 9) for item in episodes}
    if len(frequencies) != 1:
        raise ValueError(f"RGB episode sources use mixed frequencies: {frequencies}")
    control_hz = frequencies.pop()
    return {
        "episode_cache_version": RGB_EPISODE_CACHE_VERSION,
        "complete": complete,
        "source": {
            "kind": "robotwin2_lerobot_v3",
            "path": os.path.abspath(source_root),
            "variants": list(source_variants),
            "expected_source_fps": float(expected_source_fps),
            "source_frame_stride": 1,
            "index_sha256": source_index_sha256(episodes),
        },
        "split_contract": {
            "name": "rt2_task_md5_v1",
            "heldseed_fraction": RT2_HELDSEED_FRACTION,
            "heldtasks": list(RT2_HELDTASKS),
        },
        "control_hz": control_hz,
        "sample_frame_count": EXPECTED_FRAME_COUNT,
        "sampling": {
            "window_lengths": list(window_lengths),
            "sample_stride": int(sample_stride),
            "group_balance": "task_sqrt_coverage",
            "group_sampling_temperature": 0.5,
            "group_sampler_version": GROUP_SAMPLER_VERSION,
        },
        "feature": {
            "source": "jit",
            "model": JIT_DINO_MODEL,
            "image_size": JIT_DINO_IMAGE_SIZE,
            "feature_dim": JIT_DINO_FEATURE_DIM,
            "feature_contract": JIT_DINO_FEATURE_CONTRACT,
        },
        "rgb_cache": {
            "preprocess": "spatracker_vggt_then_square_v1",
            "height": JIT_DINO_IMAGE_SIZE,
            "width": JIT_DINO_IMAGE_SIZE,
            "jpeg_quality": int(jpeg_quality),
        },
        "episodes": [
            {
                "filename": episode["filename"],
                "frame_count": int(episode["frame_count"]),
                "split": episode["split"],
                "control_hz": float(episode["control_hz"]),
                "sampling_group": hashlib.sha256(episode["task"].encode()).hexdigest()[
                    :16
                ],
            }
            for episode in episodes
        ],
    }


def validate_manifest(manifest: dict, path: str) -> float:
    if manifest.get("episode_cache_version") != RGB_EPISODE_CACHE_VERSION:
        raise ValueError(f"RGB episode manifest version mismatch: {path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"RGB episode manifest is incomplete: {path}")
    control_hz = float(manifest.get("control_hz", 0.0))
    if control_hz <= 0:
        raise ValueError(f"RGB episode control frequency is invalid: {path}")
    if int(manifest.get("sample_frame_count", 0)) != EXPECTED_FRAME_COUNT:
        raise ValueError(f"RGB episode sample count differs: {path}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"RGB episode manifest has no episodes: {path}")
    if any(
        abs(float(episode.get("control_hz", 0.0)) - control_hz) > 1e-9
        for episode in episodes
    ):
        raise ValueError(f"RGB episode manifest contains mixed frequencies: {path}")
    sampling = manifest.get("sampling", {})
    if (
        int(sampling.get("sample_stride", 0)) < 1
        or sampling.get("group_balance") != "task_sqrt_coverage"
        or float(sampling.get("group_sampling_temperature", -1.0)) != 0.5
        or int(sampling.get("group_sampler_version", 0)) != GROUP_SAMPLER_VERSION
    ):
        raise ValueError(f"RGB episode sampling contract differs: {sampling}")
    feature = manifest.get("feature", {})
    expected_feature = {
        "source": "jit",
        "model": JIT_DINO_MODEL,
        "image_size": JIT_DINO_IMAGE_SIZE,
        "feature_dim": JIT_DINO_FEATURE_DIM,
        "feature_contract": JIT_DINO_FEATURE_CONTRACT,
    }
    if feature != expected_feature:
        raise ValueError(f"JIT DINO feature contract differs: {feature}")
    rgb = manifest.get("rgb_cache", {})
    if (
        rgb.get("preprocess") != "spatracker_vggt_then_square_v1"
        or int(rgb.get("height", 0)) != JIT_DINO_IMAGE_SIZE
        or int(rgb.get("width", 0)) != JIT_DINO_IMAGE_SIZE
        or not 1 <= int(rgb.get("jpeg_quality", 0)) <= 100
    ):
        raise ValueError(f"RGB episode storage contract differs: {rgb}")
    parse_control_windows(sampling["window_lengths"])
    return control_hz


def validate_episode_payload(
    payload: dict,
    path: str,
    entry: dict,
    manifest: dict,
) -> None:
    expected = {
        "episode_version": RGB_EPISODE_CACHE_VERSION,
        "source_name": entry["filename"],
        "split": entry["split"],
        "control_hz": float(entry["control_hz"]),
        "visual_preprocess": manifest["rgb_cache"]["preprocess"],
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if "dino" in payload:
        mismatches["dino"] = ("present", "forbidden")
    frame_count = int(entry["frame_count"])
    controls = payload.get("frame_control_indices")
    if (
        not torch.is_tensor(controls)
        or controls.dtype != torch.int64
        or controls.shape != (frame_count,)
        or not torch.equal(controls, torch.arange(frame_count))
    ):
        mismatches["frame_control_indices"] = (
            getattr(controls, "shape", None),
            (frame_count,),
        )
    rgb = payload.get("rgb")
    if not isinstance(rgb, dict):
        mismatches["rgb"] = (type(rgb).__name__, "dict")
    else:
        blob = rgb.get("jpeg_bytes")
        offsets = rgb.get("jpeg_offsets")
        rgb_contract = manifest["rgb_cache"]
        if (
            not torch.is_tensor(blob)
            or blob.dtype != torch.uint8
            or blob.ndim != 1
            or not torch.is_tensor(offsets)
            or offsets.dtype != torch.int64
            or offsets.shape != (frame_count + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(blob)
            or not bool((offsets[1:] > offsets[:-1]).all())
        ):
            mismatches["rgb.storage"] = ("invalid", "packed JPEG")
        for name in ("height", "width", "jpeg_quality"):
            if int(rgb.get(name, 0)) != int(rgb_contract[name]):
                mismatches[f"rgb.{name}"] = (
                    rgb.get(name),
                    rgb_contract[name],
                )
    forbidden = {
        "instruction",
        "condition_feature",
        "condition_tokens",
        "task",
        "task_index",
        "language",
    }
    present = sorted(forbidden.intersection(payload))
    if present:
        mismatches["semantic_fields"] = (present, [])
    if os.path.getsize(path) <= 0:
        mismatches["file_size"] = (0, ">0")
    if mismatches:
        raise ValueError(f"RGB episode payload differs at {path}: {mismatches}")


def validate_final_entry(path: str, entry: dict) -> None:
    if int(entry.get("cache_bytes", 0)) != os.path.getsize(path):
        raise ValueError(f"RGB episode byte count differs: {path}")
    expected_hash = entry.get("cache_sha256")
    if not _valid_sha256(expected_hash) or file_sha256(path) != expected_hash:
        raise ValueError(f"RGB episode checksum differs: {path}")
