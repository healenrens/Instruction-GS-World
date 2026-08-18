#!/usr/bin/env python3
"""Build a lightweight path index for diverse robot-video sources."""

from __future__ import annotations

import argparse
from collections import Counter
from glob import glob
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.multisource_video_index import (  # noqa: E402
    MULTISOURCE_VIDEO_CONTRACT,
)


SOURCE_SPEC_CONTRACT = "multisource_robot_video_source_spec_v1"


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cache_episodes(source: dict, source_index: int) -> list[dict]:
    root = os.path.abspath(source["root"])
    manifest = read_json(os.path.join(root, "episode_manifest.json"))
    fps = float(manifest["control_hz"])
    result = []
    for episode_index, entry in enumerate(manifest["episodes"]):
        result.append(
            {
                "source_index": source_index,
                "episode_index": episode_index,
                "split": entry["split"],
                "group": str(entry["sampling_group"]),
                "path": os.path.join(root, entry["filename"]),
                "fps": fps,
                "frame_count": int(entry["frame_count"]),
                "frame_offset": 0,
            }
        )
    return result


def find_task_roots(root: str) -> list[str]:
    info_paths = [os.path.join(root, "meta", "info.json")]
    info_paths.extend(glob(os.path.join(root, "*", "meta", "info.json")))
    return sorted(
        os.path.dirname(os.path.dirname(path))
        for path in info_paths
        if os.path.isfile(path)
    )


def select_camera(info: dict, source: dict, task_root: str) -> str:
    priorities = source.get("camera_priority")
    if priorities is None:
        priorities = [source["camera"]]
    video_keys = {
        key for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    }
    for camera in priorities:
        if camera in video_keys:
            return str(camera)
    raise ValueError(f"none of the requested cameras exist in {task_root}: {priorities}")


def format_video_path(task_root: str, info: dict, camera: str, episode: int) -> str:
    chunk_size = int(info["chunks_size"])
    template = str(info["video_path"])
    values = {
        "episode_chunk": episode // chunk_size,
        "episode_index": episode,
        "video_key": camera,
    }
    return os.path.join(task_root, template.format(**values))


def lerobot_v21_episodes(source: dict, source_index: int) -> list[dict]:
    root = os.path.abspath(source["root"])
    result = []
    for task_root in find_task_roots(root):
        info = read_json(os.path.join(task_root, "meta", "info.json"))
        camera = select_camera(info, source, task_root)
        episodes_path = os.path.join(task_root, "meta", "episodes.jsonl")
        if not os.path.isfile(episodes_path):
            raise ValueError(f"LeRobot v2.1 episode metadata is missing: {episodes_path}")
        group = os.path.relpath(task_root, root)
        fps = float(info["fps"])
        for raw in read_jsonl(episodes_path):
            local_index = int(raw["episode_index"])
            path = format_video_path(task_root, info, camera, local_index)
            result.append(
                {
                    "source_index": source_index,
                    "episode_index": len(result),
                    "split": str(raw.get("split", "train")),
                    "group": group,
                    "path": path,
                    "fps": fps,
                    "frame_count": int(raw["length"]),
                    "frame_offset": 0,
                }
            )
    return result


def parquet_rows(paths: list[str], required: tuple[str, ...], optional: tuple[str, ...]) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = set(required) - available
        if missing:
            raise ValueError(f"episode metadata is missing {sorted(missing)}: {path}")
        columns = list(required) + [name for name in optional if name in available]
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def episode_group(row: dict, fallback: str) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if tasks:
        return str(tasks)
    if "task_index" in row:
        return f"task_{int(row['task_index'])}"
    return fallback


def lerobot_v30_episodes(source: dict, source_index: int) -> list[dict]:
    root = os.path.abspath(source["root"])
    result = []
    for task_root in find_task_roots(root):
        info = read_json(os.path.join(task_root, "meta", "info.json"))
        if not str(info.get("codebase_version", "")).startswith("v3"):
            raise ValueError(f"LeRobot v3 metadata expected in {task_root}")
        camera = select_camera(info, source, task_root)
        video_root = os.path.join(task_root, "videos", camera)
        if not os.path.isdir(video_root):
            continue
        available_videos = set(
            glob(os.path.join(video_root, "chunk-*", "file-*.mp4"))
        )
        if not available_videos:
            continue
        episode_paths = sorted(
            glob(os.path.join(task_root, "meta", "episodes", "chunk-*", "file-*.parquet"))
        )
        if not episode_paths:
            raise ValueError(f"LeRobot v3 episode metadata is missing in {task_root}")
        fps = float(info["fps"])
        template = str(info["video_path"])
        fallback_group = os.path.relpath(task_root, root)
        video_columns = (
            f"videos/{camera}/chunk_index",
            f"videos/{camera}/file_index",
            f"videos/{camera}/from_timestamp",
            f"videos/{camera}/to_timestamp",
        )
        for row in parquet_rows(
            episode_paths,
            ("episode_index", "length", *video_columns),
            ("split", "tasks", "task_index"),
        ):
            chunk = int(row[f"videos/{camera}/chunk_index"])
            file_index = int(row[f"videos/{camera}/file_index"])
            start_time = float(row[f"videos/{camera}/from_timestamp"])
            end_time = float(row[f"videos/{camera}/to_timestamp"])
            length = int(row["length"])
            if round((end_time - start_time) * fps) != length:
                raise ValueError(f"video episode duration differs from length in {task_root}")
            path = os.path.join(
                task_root,
                template.format(video_key=camera, chunk_index=chunk, file_index=file_index),
            )
            if path not in available_videos:
                continue
            result.append(
                {
                    "source_index": source_index,
                    "episode_index": len(result),
                    "split": str(row.get("split", "train")),
                    "group": episode_group(row, fallback_group),
                    "path": path,
                    "fps": fps,
                    "frame_count": length,
                    "frame_offset": round(start_time * fps),
                }
            )
    return result


def _numeric_video_key(path: str) -> tuple[int, int]:
    match = re.search(r"/chunk-(\d+)/file-(\d+)\.mp4$", path)
    if match is None:
        raise ValueError(f"unknown converted video path: {path}")
    return int(match.group(1)), int(match.group(2))


def _video_frame_count(path: str, expected_fps: float) -> int:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = int(stream.frames)
        rate = float(stream.average_rate)
    if frames <= 0:
        raise ValueError(f"video container has no indexed frame count: {path}")
    if abs(rate - expected_fps) > 1e-3:
        raise ValueError(f"video fps {rate} differs from metadata {expected_fps}: {path}")
    return frames


def _hy_episode_tasks(converted_table: str, minimum_age_seconds: float) -> dict[int, int]:
    import pyarrow.parquet as pq

    cutoff = time.time() - minimum_age_seconds
    result = {}
    paths = sorted(glob(os.path.join(converted_table, "data", "chunk-*", "file-*.parquet")))
    for path in paths:
        if os.path.getmtime(path) > cutoff:
            continue
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        episode_ids = set(int(value) for value in table["episode_index"].to_pylist())
        if len(episode_ids) != 1:
            raise ValueError(f"HY converted parquet mixes episodes: {path}")
        counts = Counter(int(value) for value in table["task_index"].to_pylist())
        if not counts:
            raise ValueError(f"HY converted parquet has no task rows: {path}")
        episode_index = episode_ids.pop()
        result[episode_index] = min(
            counts, key=lambda value: (-counts[value], value)
        )
    return result


def _hy_episode_rows(table_root: str) -> list[dict]:
    paths = sorted(glob(os.path.join(table_root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not paths:
        raise ValueError(f"HY episode metadata is missing: {table_root}")
    return parquet_rows(
        paths,
        ("episode_index", "length", "dataset_from_index", "dataset_to_index"),
        ("tasks", "task_index"),
    )


def hy_converted_video_episodes(source: dict, source_index: int) -> list[dict]:
    raw_root = os.path.abspath(source["root"])
    converted_root = os.path.abspath(source["converted_root"])
    camera = str(source["camera"])
    minimum_age_seconds = float(source.get("minimum_file_age_seconds", 300.0))
    cutoff = time.time() - minimum_age_seconds
    result = []
    converted_tables = sorted(
        path for path in glob(os.path.join(converted_root, "table_*"))
        if os.path.isdir(path)
    )
    for converted_table in converted_tables:
        table_name = os.path.basename(converted_table)
        raw_table = os.path.join(raw_root, table_name)
        info = read_json(os.path.join(raw_table, "meta", "info.json"))
        fps = float(info["fps"])
        episode_tasks = _hy_episode_tasks(converted_table, minimum_age_seconds)
        rows = _hy_episode_rows(raw_table)
        rows.sort(key=lambda row: int(row["dataset_from_index"]))
        videos = sorted(
            (
                path
                for path in glob(
                    os.path.join(converted_table, "videos", camera, "chunk-*", "file-*.mp4")
                )
                if os.path.getmtime(path) <= cutoff
            ),
            key=_numeric_video_key,
        )
        if not videos:
            continue
        row_position = 0
        shard_start = 0
        for video_path in videos:
            shard_end = shard_start + _video_frame_count(video_path, fps)
            while row_position < len(rows):
                row = rows[row_position]
                episode_start = int(row["dataset_from_index"])
                episode_end = int(row["dataset_to_index"])
                if episode_start >= shard_end:
                    break
                if episode_start < shard_start or episode_end > shard_end:
                    raise ValueError(
                        f"HY video shard cuts through episode {row['episode_index']}: {video_path}"
                    )
                original_episode = int(row["episode_index"])
                if original_episode in episode_tasks:
                    task_index = episode_tasks[original_episode]
                    result.append(
                        {
                            "source_index": source_index,
                            "episode_index": len(result),
                            "split": "train",
                            "group": f"{table_name}/task_{task_index}",
                            "path": video_path,
                            "fps": fps,
                            "frame_count": int(row["length"]),
                            "frame_offset": episode_start - shard_start,
                        }
                    )
                row_position += 1
            shard_start = shard_end
    return result


def external_index_episodes(source: dict, source_index: int) -> list[dict]:
    root = os.path.abspath(source["root"])
    rows = read_jsonl(os.path.abspath(source["episode_index"]))
    result = []
    for row in rows:
        path = str(row["path"])
        result.append(
            {
                "source_index": source_index,
                "episode_index": len(result),
                "split": str(row.get("split", "train")),
                "group": str(row["group"]),
                "path": path if os.path.isabs(path) else os.path.join(root, path),
                "fps": float(row.get("fps", source["fps"])),
                "frame_count": int(row["frame_count"]),
                "frame_offset": int(row.get("frame_offset", 0)),
            }
        )
    return result


def build(spec: dict) -> dict:
    if spec.get("contract") != SOURCE_SPEC_CONTRACT:
        raise ValueError("unknown multisource source-spec contract")
    sources, episodes = [], []
    builders = {
        "rgb_episode_cache": cache_episodes,
        "lerobot_v21_task_tree": lerobot_v21_episodes,
        "lerobot_v30_task_tree": lerobot_v30_episodes,
        "hy_converted_video": hy_converted_video_episodes,
        "external_episode_index": external_index_episodes,
    }
    for source_index, raw in enumerate(spec.get("sources", [])):
        source = dict(raw)
        builder_name = str(source.pop("builder"))
        if builder_name not in builders:
            raise ValueError(f"unsupported source builder: {builder_name}")
        sources.append(
            {
                "name": str(source["name"]),
                "adapter": "rgb_episode_cache" if builder_name == "rgb_episode_cache" else "video_file",
                "weight": float(source.get("weight", 1.0)),
                "builder": builder_name,
                "root": os.path.abspath(source["root"]),
                "camera": source.get("camera", ""),
                "camera_priority": list(source.get("camera_priority", ())),
            }
        )
        source_episodes = builders[builder_name](source, source_index)
        if not source_episodes:
            raise ValueError(f"source produced no episodes: {source['name']}")
        episodes.extend(source_episodes)
    if not sources or not episodes:
        raise ValueError("source spec produced an empty video index")
    return {
        "contract": MULTISOURCE_VIDEO_CONTRACT,
        "start_step_seconds": float(spec.get("start_step_seconds", 0.5)),
        "samples_per_task": int(spec.get("samples_per_task", 4096)),
        "sources": sources,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec_path, output_path = os.path.abspath(args.spec), os.path.abspath(args.output)
    payload = build(read_json(spec_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    counts = {
        source["name"]: sum(
            int(episode["source_index"] == index) for episode in payload["episodes"]
        )
        for index, source in enumerate(payload["sources"])
    }
    print(json.dumps({"output": output_path, "episodes": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
