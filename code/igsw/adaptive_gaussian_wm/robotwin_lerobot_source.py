"""Read language-free visual episodes from RoboTwin2 LeRobot-v3 repos."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Iterator, Sequence

import h5py
import pyarrow.parquet as pq
import torch
import torchvision

from .episode_cache_encoding import decode_source_batch
from .sequence_contract import CONTROL_HZ, stable_rt2_episode_split


LEROBOT_SOURCE_KIND = "robotwin2_lerobot_v3"
LEROBOT_CAMERA_KEY = "observation.images.cam_high"
LEROBOT_DEFAULT_ROOT = "/mnt/pfs/public/fanyupeng/dataset/robotwin2_lerobot"
LEROBOT_DEFAULT_VARIANTS = ("demo_clean", "demo_randomized")
LEROBOT_EXPECTED_FPS = 30.0
HDF5_SOURCE_KIND = "robotwin_hdf5"
HDF5_ROOT_SOURCE_KIND = "robotwin_hdf5_root"


def parse_source_variants(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value.replace(",", " ").split() if isinstance(value, str) else value
    variants = tuple(
        dict.fromkeys(str(item).strip() for item in raw if str(item).strip())
    )
    if not variants:
        raise ValueError("RoboTwin LeRobot source variants cannot be empty")
    if any(os.path.basename(item) != item for item in variants):
        raise ValueError(f"invalid RoboTwin LeRobot source variants: {variants}")
    return variants


def _format_path(template: str, *, video_key: str = "", chunk: int, file: int) -> str:
    normalized = template.replace("{chunk_index", "{episode_chunk").replace(
        "{file_index", "{episode_file"
    )
    return normalized.format(
        video_key=video_key,
        episode_chunk=chunk,
        episode_file=file,
    )


def _metadata_rows(split_dir: str) -> list[tuple[str, dict]]:
    pattern = os.path.join(split_dir, "meta", "episodes", "chunk-*", "file-*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"LeRobot episode metadata is missing: {pattern}")
    rows = []
    for path in paths:
        table = pq.read_table(path)
        rows.extend((os.path.abspath(path), row) for row in table.to_pylist())
    rows.sort(key=lambda item: int(item[1]["episode_index"]))
    return rows


def _required_integer(row: dict, key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"LeRobot episode metadata is missing {key}")
    return int(value)


def _required_float(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"LeRobot episode metadata is missing {key}")
    return float(value)


def discover_lerobot_episodes(
    source_root: str,
    variants: str | Sequence[str] = LEROBOT_DEFAULT_VARIANTS,
    expected_fps: float = LEROBOT_EXPECTED_FPS,
) -> list[dict]:
    """Build one deterministic visual-only source record per LeRobot episode."""
    source_root = os.path.abspath(source_root)
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"RoboTwin LeRobot root is missing: {source_root}")
    variants = parse_source_variants(variants)
    episodes = []
    variant_counts = {variant: 0 for variant in variants}
    identities = set()
    for task in sorted(os.listdir(source_root)):
        task_dir = os.path.join(source_root, task)
        if not os.path.isdir(task_dir):
            continue
        for variant in variants:
            split_dir = os.path.join(task_dir, variant)
            if not os.path.isdir(split_dir):
                continue
            info_path = os.path.join(split_dir, "meta", "info.json")
            if not os.path.isfile(info_path):
                raise FileNotFoundError(f"LeRobot info.json is missing: {info_path}")
            with open(info_path, encoding="utf-8") as handle:
                info = json.load(handle)
            source_fps = float(info.get("fps", 0.0))
            if abs(source_fps - expected_fps) > 1e-6:
                raise ValueError(
                    f"RoboTwin source must remain {expected_fps:g} Hz: "
                    f"{info_path} declares {source_fps:g} Hz"
                )
            data_template = info.get(
                "data_path",
                "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            )
            video_template = info.get(
                "video_path",
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            )
            metadata = _metadata_rows(split_dir)
            declared_episodes = int(info.get("total_episodes", -1))
            declared_frames = int(info.get("total_frames", -1))
            metadata_frames = sum(
                _required_integer(row, "length") for _, row in metadata
            )
            if declared_episodes != len(metadata) or declared_frames != metadata_frames:
                raise ValueError(
                    f"LeRobot metadata totals differ at {split_dir}: "
                    f"episodes={len(metadata)}/{declared_episodes} "
                    f"frames={metadata_frames}/{declared_frames}"
                )
            referenced_data = set()
            split_episodes = []
            for metadata_path, row in metadata:
                episode = _required_integer(row, "episode_index")
                identity = (task, variant, episode)
                if identity in identities:
                    raise ValueError(f"duplicate LeRobot episode identity: {identity}")
                identities.add(identity)
                length = _required_integer(row, "length")
                from_index = _required_integer(row, "dataset_from_index")
                to_index = _required_integer(row, "dataset_to_index")
                if length < 2 or to_index - from_index != length:
                    raise ValueError(
                        f"invalid LeRobot episode bounds for {identity}: "
                        f"length={length} range=[{from_index},{to_index})"
                    )
                data_path = os.path.join(
                    split_dir,
                    _format_path(
                        str(data_template),
                        chunk=int(row.get("data/chunk_index", 0)),
                        file=int(row.get("data/file_index", 0)),
                    ),
                )
                chunk_key = f"videos/{LEROBOT_CAMERA_KEY}/chunk_index"
                file_key = f"videos/{LEROBOT_CAMERA_KEY}/file_index"
                from_key = f"videos/{LEROBOT_CAMERA_KEY}/from_timestamp"
                video_path = os.path.join(
                    split_dir,
                    _format_path(
                        str(video_template),
                        video_key=LEROBOT_CAMERA_KEY,
                        chunk=int(row.get(chunk_key, 0)),
                        file=int(row.get(file_key, episode)),
                    ),
                )
                for path in (data_path, video_path):
                    if not os.path.isfile(path):
                        raise FileNotFoundError(
                            f"LeRobot source file is missing: {path}"
                        )
                referenced_data.add(os.path.abspath(data_path))
                split_episodes.append(
                    {
                        "source_format": LEROBOT_SOURCE_KIND,
                        "task": task,
                        "source_variant": variant,
                        "episode": episode,
                        "metadata_path": metadata_path,
                        "data_path": os.path.abspath(data_path),
                        "video_path": os.path.abspath(video_path),
                        "video_from_timestamp": _required_float(row, from_key),
                        "source_from_index": from_index,
                        "source_to_index": to_index,
                        "source_fps": source_fps,
                        "source_frame_stride": 1,
                        "frame_count": length,
                        "control_hz": source_fps,
                        "split": stable_rt2_episode_split(task, episode),
                    }
                )
                variant_counts[variant] += 1
            parquet_rows = sum(
                int(pq.ParquetFile(path).metadata.num_rows)
                for path in sorted(referenced_data)
            )
            if parquet_rows != declared_frames:
                raise ValueError(
                    f"LeRobot data parquet totals differ at {split_dir}: "
                    f"rows={parquet_rows} declared={declared_frames}"
                )
            episodes.extend(split_episodes)
    missing = [variant for variant, count in variant_counts.items() if count == 0]
    if missing:
        raise ValueError(f"RoboTwin LeRobot variants contain no episodes: {missing}")
    return episodes


def discover_hdf5_episodes(source_root: str) -> list[dict]:
    """Preserve the original source-root contract for legacy cache launchers."""
    episodes = []
    pattern = re.compile(r"episode(\d+)\.hdf5$")
    for task in sorted(os.listdir(source_root)):
        data_dir = os.path.join(source_root, task, "demo_clean", "data")
        if not os.path.isdir(data_dir):
            continue
        for name in sorted(os.listdir(data_dir)):
            match = pattern.fullmatch(name)
            if match is None:
                continue
            episode = int(match.group(1))
            path = os.path.abspath(os.path.join(data_dir, name))
            with h5py.File(path, "r") as handle:
                frame_count = len(handle["observation/head_camera/rgb"])
            episodes.append(
                {
                    "source_format": HDF5_SOURCE_KIND,
                    "task": task,
                    "source_variant": "demo_clean",
                    "episode": episode,
                    "hdf5": path,
                    "frame_count": frame_count,
                    "source_fps": CONTROL_HZ,
                    "source_frame_stride": 1,
                    "control_hz": CONTROL_HZ,
                    "split": stable_rt2_episode_split(task, episode),
                }
            )
    if not episodes:
        raise ValueError(f"RoboTwin HDF5 source is empty: {source_root}")
    return episodes


def assign_episode_filenames(episodes: list[dict]) -> list[dict]:
    filenames = set()
    for episode in episodes:
        if episode.get("source_format") == LEROBOT_SOURCE_KIND:
            prefix = (
                f"{episode['task']}_{episode['source_variant']}_"
                f"ep{int(episode['episode']):04d}"
            )
        else:
            prefix = f"{episode['task']}_ep{int(episode['episode']):02d}"
        filename = f"{prefix}_{episode['split']}.pt"
        if filename in filenames or os.path.basename(filename) != filename:
            raise ValueError(f"invalid or duplicate episode cache filename: {filename}")
        filenames.add(filename)
        episode["filename"] = filename
    return episodes


def source_index_sha256(episodes: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(
            json.dumps(episode, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def source_files(episode: dict) -> tuple[str, ...]:
    if episode.get("source_format") == LEROBOT_SOURCE_KIND:
        return tuple(
            str(episode[name]) for name in ("metadata_path", "data_path", "video_path")
        )
    return (str(episode["hdf5"]),)


def decode_lerobot_batch(episode: dict, start: int, end: int) -> torch.Tensor:
    fps = float(episode["source_fps"])
    first = float(episode["video_from_timestamp"])
    timestamps = [first + index / fps for index in range(start, end)]
    tolerance = 0.5 / fps + 1e-4
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(episode["video_path"], "video")
    reader.seek(timestamps[0], keyframes_only=True)
    loaded_frames = []
    loaded_timestamps = []
    for frame in reader:
        timestamp = float(frame["pts"])
        loaded_frames.append(frame["data"])
        loaded_timestamps.append(timestamp)
        if timestamp >= timestamps[-1]:
            break
    reader.container.close()
    if not loaded_frames:
        raise RuntimeError(f"video seek decoded no frames: {episode['video_path']}")
    query = torch.tensor(timestamps, dtype=torch.float64)
    decoded = torch.tensor(loaded_timestamps, dtype=torch.float64)
    distances = torch.cdist(query[:, None], decoded[:, None], p=1)
    minimum, closest = distances.min(dim=1)
    if not bool((minimum <= tolerance).all()):
        raise RuntimeError(
            "LeRobot video timestamps exceed tolerance: "
            f"max={float(minimum.max()):.6f}s tolerance={tolerance:.6f}s "
            f"video={episode['video_path']}"
        )
    frames = torch.stack([loaded_frames[int(index)] for index in closest])
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise RuntimeError(f"LeRobot decoder returned invalid RGB: {frames.shape}")
    return frames.permute(0, 2, 3, 1).contiguous()


def iter_episode_batches(episode: dict, batch_size: int) -> Iterator[torch.Tensor]:
    frame_count = int(episode["frame_count"])
    if episode.get("source_format") == LEROBOT_SOURCE_KIND:
        for start in range(0, frame_count, batch_size):
            yield decode_lerobot_batch(
                episode,
                start,
                min(start + batch_size, frame_count),
            )
        return
    if episode.get("source_format", HDF5_SOURCE_KIND) != HDF5_SOURCE_KIND:
        raise ValueError(f"unsupported episode source: {episode.get('source_format')}")
    with h5py.File(episode["hdf5"], "r") as handle:
        source = handle["observation/head_camera/rgb"]
        if len(source) != frame_count:
            raise ValueError(f"episode frame count differs: {episode['hdf5']}")
        for start in range(0, frame_count, batch_size):
            yield decode_source_batch(
                source,
                start,
                min(start + batch_size, frame_count),
            )


def hdf5_plan_episode(window: dict) -> dict:
    return {
        "source_format": HDF5_SOURCE_KIND,
        "task": str(window["task"]),
        "source_variant": "demo_clean",
        "episode": int(window["ep"]),
        "hdf5": os.path.abspath(window["hdf5"]),
        "frame_count": int(window["T"]),
        "source_fps": CONTROL_HZ,
        "source_frame_stride": 1,
        "control_hz": CONTROL_HZ,
        "split": str(window["split"]),
    }
