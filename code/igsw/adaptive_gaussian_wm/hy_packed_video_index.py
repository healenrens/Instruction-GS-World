"""Build HY episode records from packed videos and raw frame intervals."""

from __future__ import annotations

from glob import glob
import json
import os
import re
import subprocess
import time


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _numeric_key(path: str) -> tuple[int, int]:
    match = re.search(r"/chunk-(\d+)/file-(\d+)\.(?:mp4|parquet)$", path)
    if match is None:
        raise ValueError(f"unknown HY shard path: {path}")
    return int(match.group(1)), int(match.group(2))


def _raw_episode_rows(table_root: str) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted(
        glob(
            os.path.join(table_root, "meta", "episodes", "**", "*.parquet"),
            recursive=True,
        )
    )
    if not paths:
        raise ValueError(f"HY episode metadata is missing: {table_root}")
    required = (
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
    )
    rows = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = set(required) - available
        if missing:
            raise ValueError(f"HY episode metadata is missing {sorted(missing)}: {path}")
        rows.extend(pq.read_table(path, columns=list(required)).to_pylist())
    rows.sort(key=lambda row: int(row["dataset_from_index"]))
    previous_end = 0
    episode_ids = set()
    for row in rows:
        episode = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        if episode in episode_ids:
            raise ValueError(f"HY episode metadata repeats episode {episode}")
        if start != previous_end or end - start != length or length <= 0:
            raise ValueError(f"HY episode frame intervals are not contiguous: {table_root}")
        episode_ids.add(episode)
        previous_end = end
    return rows


def _column_index(parquet_file, column: str, path: str) -> int:
    names = parquet_file.schema_arrow.names
    if column not in names:
        raise ValueError(f"HY converted parquet is missing {column}: {path}")
    return names.index(column)


def _constant_row_group_value(parquet_file, row_group: int, column_index: int):
    statistics = (
        parquet_file.metadata.row_group(row_group).column(column_index).statistics
    )
    if statistics is None or not statistics.has_min_max:
        return None
    minimum, maximum = int(statistics.min), int(statistics.max)
    return minimum if minimum == maximum else None


def _row_group_episode_tasks(
    parquet_file,
    row_group: int,
    path: str,
) -> list[tuple[int, int]]:
    episode_index = _column_index(parquet_file, "episode_index", path)
    task_index = _column_index(parquet_file, "task_index", path)
    episode = _constant_row_group_value(parquet_file, row_group, episode_index)
    task = _constant_row_group_value(parquet_file, row_group, task_index)
    if episode is not None and task is not None:
        return [(episode, task)]

    table = parquet_file.read_row_group(
        row_group,
        columns=["episode_index", "task_index"],
    )
    grouped = table.group_by("episode_index").aggregate(
        [("task_index", "min"), ("task_index", "max")]
    )
    result = []
    for row in grouped.to_pylist():
        minimum = int(row["task_index_min"])
        maximum = int(row["task_index_max"])
        if minimum != maximum:
            raise ValueError(
                f"HY episode {row['episode_index']} mixes task_index values: {path}"
            )
        result.append((int(row["episode_index"]), minimum))
    return result


def _episode_task_labels(
    converted_table: str,
    required_episodes: set[int],
    cutoff: float,
) -> tuple[dict[int, int], int]:
    import pyarrow.parquet as pq

    if not required_episodes:
        return {}, 0
    labels = {}
    recent_files = 0
    paths = sorted(
        glob(os.path.join(converted_table, "data", "chunk-*", "file-*.parquet")),
        key=_numeric_key,
    )
    for position, path in enumerate(paths, start=1):
        if os.path.getmtime(path) > cutoff:
            recent_files += 1
            continue
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.num_row_groups):
            for episode, task in _row_group_episode_tasks(
                parquet_file, row_group, path
            ):
                if episode not in required_episodes:
                    continue
                previous = labels.setdefault(episode, task)
                if previous != task:
                    raise ValueError(
                        f"HY episode {episode} has conflicting task labels"
                    )
        if len(labels) == len(required_episodes):
            break
        if position % 2000 == 0:
            print(
                json.dumps(
                    {
                        "event": "hy_task_label_progress",
                        "table": os.path.basename(converted_table),
                        "files_scanned": position,
                        "labels_found": len(labels),
                        "labels_required": len(required_episodes),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return labels, recent_files


def _frame_rate(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _probe_video_header(path: str, expected_fps: float) -> tuple[str, int, dict]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "ffprobe_error", 0, {"stderr": result.stderr.strip()[-500:]}
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        return "missing_video_stream", 0, {}
    stream = streams[0]
    rate = _frame_rate(str(stream.get("avg_frame_rate", "0/0")))
    if rate <= 0.0:
        return "invalid_frame_rate", 0, {"frame_rate": rate}
    if abs(rate - expected_fps) > 1e-3:
        return "fps_mismatch", 0, {"frame_rate": rate}
    raw_frames = str(stream.get("nb_frames", "0"))
    frame_count = int(raw_frames) if raw_frames.isdigit() else 0
    if frame_count <= 0:
        return "missing_frame_count", 0, {"frame_rate": rate}
    return (
        "usable",
        frame_count,
        {
            "codec": str(stream.get("codec_name", "")),
            "duration": float(stream.get("duration", 0.0)),
            "frame_rate": rate,
        },
    )


def _stable_video_prefix(
    converted_table: str,
    camera: str,
    fps: float,
    cutoff: float,
) -> tuple[list[tuple[str, int]], dict | None, int]:
    paths = sorted(
        glob(os.path.join(converted_table, "videos", camera, "chunk-*", "file-*.mp4")),
        key=_numeric_key,
    )
    prefix = []
    first_gap = None
    for position, path in enumerate(paths):
        if os.path.getmtime(path) > cutoff:
            status, frame_count, details = "recent_video", 0, {}
        else:
            status, frame_count, details = _probe_video_header(path, fps)
        if status != "usable":
            first_gap = {
                "status": status,
                "path": path,
                "details": details,
                "later_video_count": len(paths) - position - 1,
            }
            break
        prefix.append((path, frame_count))
    return prefix, first_gap, len(paths)


def _map_video_prefix(rows: list[dict], videos: list[tuple[str, int]]) -> list[dict]:
    mapped = []
    row_position = 0
    video_start = 0
    for path, frame_count in videos:
        video_end = video_start + frame_count
        while row_position < len(rows):
            row = rows[row_position]
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            if start >= video_end:
                break
            if start < video_start or end > video_end:
                raise ValueError(
                    f"HY packed video cuts through episode {row['episode_index']}: {path}"
                )
            mapped.append(
                {
                    "episode": int(row["episode_index"]),
                    "path": path,
                    "frame_count": int(row["length"]),
                    "frame_offset": start - video_start,
                }
            )
            row_position += 1
        expected_end = (
            int(rows[row_position]["dataset_from_index"])
            if row_position < len(rows)
            else int(rows[-1]["dataset_to_index"])
        )
        if expected_end != video_end:
            raise ValueError(f"HY packed video boundary does not match episodes: {path}")
        video_start = video_end
    return mapped


def build_hy_packed_video_episodes(source: dict, source_index: int) -> list[dict]:
    raw_root = os.path.abspath(source["root"])
    converted_root = os.path.abspath(source["converted_root"])
    camera = str(source["camera"])
    minimum_age = float(source.get("minimum_file_age_seconds", 300.0))
    minimum_coverage = float(source.get("minimum_episode_coverage", 0.98))
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("HY minimum episode coverage must stay within (0,1]")
    cutoff = time.time() - minimum_age
    result = []
    total_raw_episodes = 0
    total_mapped_episodes = 0
    table_summaries = []
    converted_tables = sorted(
        path
        for path in glob(os.path.join(converted_root, "table_*"))
        if os.path.isdir(path)
    )
    for converted_table in converted_tables:
        table_name = os.path.basename(converted_table)
        raw_table = os.path.join(raw_root, table_name)
        fps = float(_read_json(os.path.join(raw_table, "meta", "info.json"))["fps"])
        rows = _raw_episode_rows(raw_table)
        videos, first_gap, discovered_videos = _stable_video_prefix(
            converted_table, camera, fps, cutoff
        )
        mapped = _map_video_prefix(rows, videos)
        required_episodes = {item["episode"] for item in mapped}
        task_labels, recent_labels = _episode_task_labels(
            converted_table, required_episodes, cutoff
        )
        missing_labels = sorted(required_episodes - task_labels.keys())
        coverage = len(mapped) / len(rows)
        summary = {
            "event": "hy_packed_video_audit",
            "table": table_name,
            "raw_episode_count": len(rows),
            "mapped_episode_count": len(mapped),
            "episode_coverage": coverage,
            "discovered_video_count": discovered_videos,
            "usable_prefix_video_count": len(videos),
            "first_unusable_video": first_gap,
            "missing_task_label_count": len(missing_labels),
            "recent_task_label_file_count": recent_labels,
        }
        table_summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if missing_labels:
            raise ValueError(
                f"HY {table_name} lacks task labels for mapped episodes: {missing_labels[:8]}"
            )
        total_raw_episodes += len(rows)
        total_mapped_episodes += len(mapped)
        for item in mapped:
            result.append(
                {
                    "source_index": source_index,
                    "episode_index": len(result),
                    "split": "train",
                    "group": f"{table_name}/task_{task_labels[item['episode']]}",
                    "path": item["path"],
                    "fps": fps,
                    "frame_count": item["frame_count"],
                    "frame_offset": item["frame_offset"],
                }
            )
    source_coverage = total_mapped_episodes / total_raw_episodes
    source_summary = {
        "event": "hy_packed_video_source_audit",
        "raw_episode_count": total_raw_episodes,
        "mapped_episode_count": total_mapped_episodes,
        "episode_coverage": source_coverage,
        "minimum_episode_coverage": minimum_coverage,
        "tables_with_truncated_prefix": [
            item["table"]
            for item in table_summaries
            if item["first_unusable_video"] is not None
        ],
    }
    print(json.dumps(source_summary, sort_keys=True), flush=True)
    if source_coverage < minimum_coverage:
        raise ValueError(
            f"HY source episode coverage {source_coverage:.4f} is below required "
            f"{minimum_coverage:.4f}"
        )
    return result
