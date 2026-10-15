"""RGB-only sequences from the existing relocatable video collection."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.utils.data import Dataset

from igsw.adaptive_gaussian_wm.object_video_manifest_v69 import (
    load_object_video_manifest_v69,
    resolve_object_video_case_v69,
)
from igsw.adaptive_gaussian_wm.object_video_rgb_v69 import read_object_video_frames_v69
from igsw.adaptive_gaussian_wm.video_file_decoder import VideoDecodeError


@dataclass(frozen=True)
class SequenceConfig:
    history_seconds: float = 3.0
    future_seconds: float = 5.0
    history_frames: int = 16
    future_frames: int = 25
    sampling: str = "uniform_anchor"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def episode_key(entry):
    return entry["source"], entry["group"], int(entry["episode_index"])


def available_anchors(case, config):
    fps = float(case["record"]["fps"])
    first = int(case["first_frame"]) + int(np.ceil(config.history_seconds * fps))
    last = int(case["last_frame"]) - int(np.ceil(config.future_seconds * fps))
    return first, last


def language_windows(path):
    by_case = defaultdict(list)
    if path:
        payload = json.loads(Path(path).read_text())
        for row in payload["entries"]:
            by_case[row["case_id"]].append({
                "window_id": row["window_id"],
                "instruction": row["instruction"],
                "language_provenance": row["language_provenance"],
                "frame_indices": row["frame_indices"],
                "history_frames": row["history_frames"],
                "original_language_partition": row["partition"],
            })
    return by_case


def prepare_manifest(source_path, output, config, language_path="", media_root=None):
    source = load_object_video_manifest_v69(source_path)
    root = Path(media_root).resolve() if media_root else Path(source["root"])
    languages = language_windows(language_path)
    entries, excluded = [], []
    for original in source["entries"]:
        case = original["case"]
        media = root / case["record"]["path"]
        start, stop = available_anchors(case, config)
        reason = "missing_media" if not media.is_file() else "short_clip" if start > stop else None
        if reason:
            excluded.append({"case_id": original["case_id"], "source": original["source"],
                             "reason": reason, "media": str(media)})
            continue
        rows = []
        for window in languages[original["case_id"]]:
            indices = window["frame_indices"]
            th = window["history_frames"]
            if (th == config.history_frames and len(indices) == th + config.future_frames
                    and min(indices) >= case["first_frame"] and max(indices) <= case["last_frame"]):
                rows.append(window)
        entries.append({
            "case_id": original["case_id"], "source": original["source"],
            "group": original["group"], "episode_index": original["episode_index"],
            "partition": "train" if original["partition"] == "train" else "development",
            "original_partition": original["partition"], "case": case,
            "language_windows": rows,
        })
    counts = Counter(f"{e['source']}/{e['partition']}" for e in entries)
    summaries = {}
    for name in sorted({e["source"] for e in entries}):
        rows = [e for e in entries if e["source"] == name]
        seconds = [(e["case"]["last_frame"] - e["case"]["first_frame"])
                   / float(e["case"]["record"]["fps"]) for e in rows]
        summaries[name] = {
            "clips": len(rows), "episodes": len({episode_key(e) for e in rows}),
            "clips_with_language_windows": sum(bool(e["language_windows"]) for e in rows),
            "language_windows": sum(len(e["language_windows"]) for e in rows),
            "seconds_min": min(seconds), "seconds_median": statistics.median(seconds),
            "seconds_max": max(seconds),
            "fps_counts": dict(Counter(str(e["case"]["record"]["fps"]) for e in rows)),
        }
    report = {
        "input_clips": len(source["entries"]), "rgb_clips": len(entries),
        "episodes": len({episode_key(e) for e in entries}),
        "source_partition_counts": dict(counts), "sources": summaries,
        "excluded": excluded, "exclusion_counts": dict(Counter(e["reason"] for e in excluded)),
        "teacher_files_loaded": 0, "video_decode_performed": False,
        "independent_object_truth_present": False,
        "language_source": str(Path(language_path).resolve()) if language_path else None,
        "language_scope_counts": dict(Counter(
            w["language_provenance"].get("instruction_scope", "unspecified")
            for e in entries for w in e["language_windows"])),
    }
    payload = {
        "contract": "object_event_video_v1", "root": str(root),
        "source_manifest": str(Path(source_path).resolve()),
        "sequence": vars(config), "entries": entries, "report": report,
        "split_use": {"train": "inherited_training_episodes",
                      "development": "historically_reviewed_held_episodes", "test": "not_created"},
        "supervision": "observed_RGB; optional historical language provenance; no tracker required",
    }
    write_json(output, payload)
    return payload


def unique_frame_mask(indices, history_frames):
    valid = torch.ones_like(indices, dtype=torch.bool)
    valid[:history_frames-1] = indices[:history_frames-1] != indices[1:history_frames]
    valid[history_frames:-1] = indices[history_frames:-1] != indices[history_frames+1:]
    valid[history_frames:] &= indices[history_frames:] > indices[history_frames-1]
    return valid


class ObjectEventVideoDataset(Dataset):
    """One index per episode; multiple clips/windows do not increase episode weight."""

    def __init__(self, manifest, partition="train", config=None, seed=17):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.root = (self.path.parent / self.manifest["root"]).resolve()
        self.config = config or SequenceConfig(**self.manifest["sequence"])
        self.seed = seed
        grouped = defaultdict(list)
        for entry in self.manifest["entries"]:
            if entry["partition"] != partition:
                continue
            if self.config.sampling == "language_windows" and not entry["language_windows"]:
                continue
            grouped[episode_key(entry)].append(entry)
        self.episodes = [grouped[key] for key in sorted(grouped)]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        index, epoch, occurrence = item if isinstance(item, tuple) else (item, 0, item)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, occurrence, index]))
        candidates = self.episodes[index]
        entry = candidates[int(rng.integers(len(candidates)))]
        case = resolve_object_video_case_v69(entry["case"], self.root)
        config = self.config
        instruction, provenance, window_id = "", None, None
        if config.sampling == "language_windows":
            windows = entry["language_windows"]
            window = windows[int(rng.integers(len(windows)))]
            indices = torch.tensor(window["frame_indices"], dtype=torch.long)
            instruction, provenance = window["instruction"], window["language_provenance"]
            window_id = window["window_id"]
        else:
            first, last = available_anchors(case, config)
            anchor = int(rng.integers(first, last + 1))
            fps = float(case["record"]["fps"])
            history = np.rint(np.linspace(anchor-config.history_seconds*fps, anchor,
                                          config.history_frames)).astype(np.int64)
            future = np.rint(anchor + np.arange(1, config.future_frames+1)
                            * config.future_seconds*fps/config.future_frames).astype(np.int64)
            indices = torch.from_numpy(np.concatenate((history, future)))
            # Only episode-scoped language is reusable at a newly sampled anchor.
            episode_language = [w for w in entry["language_windows"]
                                if w["language_provenance"].get("instruction_scope")
                                == "episode_goal_not_precise_future_subtask"]
            if len({w["instruction"] for w in episode_language}) == 1:
                instruction = episode_language[0]["instruction"]
                provenance = episode_language[0]["language_provenance"]
        rgb, timestamps, basis = read_object_video_frames_v69(case, indices)
        if isinstance(rgb, VideoDecodeError):
            return {"decode_error": str(rgb), "case_id": entry["case_id"], "source": entry["source"]}
        th = config.history_frames
        return {
            "rgb": rgb, "frame_valid": unique_frame_mask(indices, th),
            "times": (timestamps-timestamps[th-1]).float(), "native_hw": torch.tensor(rgb.shape[-2:]),
            "history_frames": th, "frame_indices": indices, "time_source": basis,
            "instruction": instruction, "language_provenance": provenance, "window_id": window_id,
            "case_id": entry["case_id"], "source": entry["source"],
            "episode_index": entry["episode_index"], "partition": entry["partition"],
            "sample_index": index, "epoch": epoch, "occurrence": occurrence,
        }


def collate_object_event_video(samples):
    errors = [row for row in samples if "decode_error" in row]
    if errors:
        return {"decode_errors": errors}
    height = (max(row["rgb"].shape[-2] for row in samples)+15)//16*16
    width = (max(row["rgb"].shape[-1] for row in samples)+15)//16*16
    b, t = len(samples), len(samples[0]["rgb"])
    rgb = torch.zeros((b, t, 3, height, width), dtype=torch.uint8)
    pixel_valid = torch.zeros((b, t, height, width), dtype=torch.bool)
    for index, row in enumerate(samples):
        h, w = row["rgb"].shape[-2:]
        rgb[index, :, :, :h, :w] = row["rgb"]
        pixel_valid[index, :, :h, :w] = row["frame_valid"][:, None, None]
    batch = {"rgb": rgb, "pixel_valid": pixel_valid, "history_frames": samples[0]["history_frames"]}
    for key in samples[0]:
        if key in ("rgb", "history_frames"):
            continue
        values = [row[key] for row in samples]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return batch
