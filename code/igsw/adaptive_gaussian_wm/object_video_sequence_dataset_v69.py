"""Native RGB sequence and teacher measurements kept in separate dictionaries."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from .object_video_rgb_v69 import read_object_video_frames_v69
from .object_video_manifest_v69 import load_object_video_manifest_v69, resolve_object_video_case_v69
from .v69_config import ObjectVideoConfigV69
from .video_file_decoder import VideoDecodeError
from .grounded_background_motion_v68 import motion_evidence
from .grounded_motion_export_v68 import all_point_motion_selection


def temporal_indices(config, frame_count, fps, rng):
    # Choose the anchor using the maximum contract before sampling requested durations.
    first = int(np.ceil(config.history_seconds * fps))
    stop = frame_count - int(np.ceil(config.future_seconds * fps))
    current = int(rng.integers(first, stop))
    hs = float(rng.uniform(config.history_min_seconds, config.history_seconds))
    fs = float(rng.uniform(config.future_min_seconds, config.future_seconds))
    history = np.rint(np.linspace(current - hs * fps, current, config.history_frames)).astype(np.int64)
    future = np.rint(current + np.arange(1, config.future_frames + 1) * fs * fps / config.future_frames).astype(np.int64)
    indices = torch.from_numpy(np.concatenate((history, future)))
    unique_time = distinct_sequence_frames(indices, config.history_frames)
    return current, indices, unique_time


def distinct_sequence_frames(indices, history_frames):
    history, future = indices[:history_frames], indices[history_frames:]
    history_valid = torch.ones(len(history), dtype=torch.bool)
    future_valid = torch.ones(len(future), dtype=torch.bool)
    history_valid[:-1] = history[:-1] != history[1:]
    future_valid[:-1] = future[:-1] != future[1:]
    future_valid &= future > history[-1]
    return torch.cat((history_valid, future_valid))


def tracker_observations(data):
    native = data["native"]
    tracks = native["tracks"].float()
    finite = torch.isfinite(tracks).all(-1)
    visible = native["visibility"].bool() & native["in_bounds"].bool() & finite
    relay = data["relay_evidence"]
    second = relay["prediction"]
    second_visible = second["visibility"].bool() & second["in_bounds"].bool()
    observed = visible & relay["valid"].bool()
    original_query = native["query_local_frames"].long()
    columns = torch.arange(tracks.shape[1])
    observed[original_query, columns] = False
    # Agreement on absence is still teacher evidence, not calibrated human visibility.
    absent = (~native["visibility"].bool()) & (~second["visibility"].bool())
    absent &= native["in_bounds"].bool() & second["in_bounds"].bool() & finite
    seen_before = (observed.long().cumsum(0) > 0)
    seen_after = (observed.flip(0).long().cumsum(0).flip(0) > 0)
    absent &= seen_before & seen_after
    absent[original_query, columns] = False
    absent[relay["anchor_frames"].long(), columns] = False
    state = torch.full_like(visible, -1, dtype=torch.int8)
    state[observed & second_visible] = 1
    state[absent] = 0
    return tracks, observed, state


class ObjectVideoSequenceDatasetV69(Dataset):
    def __init__(self, manifest, config=None, seed=17, partition="train", annotations=""):
        self.path = str(Path(manifest).resolve())
        self.manifest = load_object_video_manifest_v69(manifest)
        self.root = Path(self.manifest["root"])
        self.entries = [e for e in self.manifest["entries"] if e["partition"] == partition]
        self.config, self.seed = config or ObjectVideoConfigV69(), seed
        self.annotations = {}
        if annotations:
            for line in Path(annotations).read_text().splitlines():
                row = json.loads(line)
                self.annotations[row["case_id"]] = row

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        index, epoch, occurrence = item if isinstance(item, tuple) else (item, 0, item)
        entry = self.entries[index]
        data = torch.load(self.root / entry["path"], map_location="cpu", weights_only=False)
        case = resolve_object_video_case_v69(data["case"], self.root)
        fps = float(case["record"]["fps"])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, occurrence, index]))
        _, indices, frame_valid = temporal_indices(self.config, len(data["native"]["tracks"]), fps, rng)
        annotation = self.annotations.get(entry["case_id"])
        if annotation is not None:
            indices = torch.tensor(annotation["frame_indices"], dtype=torch.long)-case["first_frame"]
            frame_valid = distinct_sequence_frames(indices, self.config.history_frames)
        rgb, timestamps, time_source = read_object_video_frames_v69(case, indices + case["first_frame"])
        if isinstance(rgb, VideoDecodeError):
            return {"decode_error": str(rgb), "case_id": entry["case_id"], "source": entry["source"]}
        th, p = self.config.history_frames, self.config.measurement_points
        tracks, reliable, observation = tracker_observations(data)
        # Point selection changes only supervised measurement locations, never Student queries.
        historical = reliable[indices[:th]]
        eligible = historical.any(0)
        ids = torch.where(eligible)[0].numpy()
        rng.shuffle(ids)
        ids = torch.as_tensor(ids[:p], dtype=torch.long)
        count = len(ids)
        positions = torch.zeros((len(indices), p, 2))
        valid = torch.zeros((len(indices), p), dtype=torch.bool)
        known = torch.full((len(indices), p), -1, dtype=torch.int8)
        measured = tracks[indices[:, None], ids[None]]
        finite = torch.isfinite(measured).all(-1)
        positions[:, :count] = torch.where(finite[..., None], measured, torch.zeros_like(measured))
        valid[:, :count] = reliable[indices[:, None], ids[None]] & frame_valid[:, None]
        known[:, :count] = observation[indices[:, None], ids[None]]
        known[~frame_valid] = -1
        hist_index = torch.arange(th)[:, None].expand(th, count)
        reference_index = torch.where(historical[:, ids], hist_index, -1).max(0).values
        reference = torch.zeros((p, 2))
        reference[:count] = tracks[indices[reference_index], ids]
        reference_indices = torch.zeros(p, dtype=torch.long)
        reference_indices[:count] = reference_index
        if "motion_target_mask" in data:
            selected = data["motion_target_mask"].bool()
        else:
            # Old raw trajectories remain reusable; never reuse the old object-role eligibility mask.
            evidence = motion_evidence(data["native"], data["background_motion"], SimpleNamespace(**data["configuration"]))
            selected = all_point_motion_selection(evidence, .75)[1]
        selection_status = "global_compensated_motion_top75"
        if not bool(selected.any()):
            measurable = torch.where(reliable.any(0))[0].numpy()
            rng.shuffle(measurable)
            selected = torch.zeros_like(selected)
            selected[measurable[:math.ceil(.75*len(measurable))]] = True
            selection_status = "uniform75_motion_ranking_unavailable" if len(measurable) else "no_reliable_measurements"
        transport_weight = torch.zeros(p)
        transport_weight[:count] = selected[ids].float()
        point_ids = torch.full((p,), -1, dtype=torch.long)
        point_ids[:count] = ids
        h, w = rgb.shape[-2:]
        scale = torch.tensor([w - 1, h - 1]).clamp_min(1)
        compensated = data["motion_evidence"]["compensated_coordinates"][indices[:, None], ids[None]].float()
        compensated_positions = torch.zeros_like(positions)
        compensated_positions[:, :count] = compensated / scale * 2 - 1
        compensated_valid = torch.zeros_like(valid)
        compensated_valid[:, :count] = data["motion_evidence"]["valid"][indices[:, None], ids[None]] & valid[:, :count]
        return {"rgb": rgb, "frame_valid": frame_valid, "times": (timestamps-timestamps[th-1]).float(),
                "native_hw": torch.tensor([h, w]), "history_frames": th,
                "teacher": {"xy": positions / scale * 2 - 1, "valid": valid, "observation": known,
                            "reference_xy": reference / scale * 2 - 1, "reference_index": reference_indices,
                            "point_present": point_ids >= 0, "point_ids": point_ids, "transport_weight": transport_weight,
                            "relative_xy": compensated_positions, "relative_valid": compensated_valid},
                "case_id": entry["case_id"], "source": entry["source"], "episode_index": entry["episode_index"],
                "frame_indices": indices + case["first_frame"], "sample_index": index, "epoch": epoch,
                "occurrence": occurrence, "annotation": annotation, "case": case, "time_source": time_source,
                "transport_selection_status": selection_status}


def collate_object_video_v69(samples):
    errors = [s for s in samples if "decode_error" in s]
    if errors:
        return {"decode_errors": errors}
    height = (max(s["rgb"].shape[-2] for s in samples) + 15) // 16 * 16
    width = (max(s["rgb"].shape[-1] for s in samples) + 15) // 16 * 16
    b, t = len(samples), len(samples[0]["rgb"])
    rgb = torch.zeros((b, t, 3, height, width), dtype=torch.uint8)
    pixel_valid = torch.zeros((b, t, height, width), dtype=torch.bool)
    for i, sample in enumerate(samples):
        h, w = sample["rgb"].shape[-2:]
        rgb[i, :, :, :h, :w] = sample["rgb"]
        pixel_valid[i, :, :h, :w] = sample["frame_valid"][:, None, None]
    result = {"rgb": rgb, "pixel_valid": pixel_valid, "history_frames": samples[0]["history_frames"]}
    for key in samples[0]:
        if key in ("rgb", "history_frames"):
            continue
        values = [s[key] for s in samples]
        if key == "teacher":
            result[key] = {name: torch.stack([v[name] for v in values]) for name in values[0]}
        else:
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return result


def move_batch_v69(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
            else move_batch_v69(value, device) if isinstance(value, dict) else value for key, value in batch.items()}
