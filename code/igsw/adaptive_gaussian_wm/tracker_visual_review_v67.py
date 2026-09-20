"""Point-tracker-only review on the existing native multisource video index."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import torch

from .multisource_point_track_dataset import MultiSourcePointTrackObjectVideoDataset
from .tracker_review_cases_v67 import select_long_cases
from .video_file_decoder import VideoDecodeError, decode_video_frames, read_video_frames


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def select_cases(args):
    if args.queries_json:
        manual = read_json(args.queries_json)
        cases = [
            case
            for case in manual["cases"]
            if manual["points"].get(case["case_id"])
            and case.get("clip_seconds", 0) >= max(10.0, args.clip_seconds)
            and case.get("camera_evidence", "").startswith("external")
        ]
        return cases, {
            "mode": "manual queries from long non-wrist review",
            "selected": len(cases),
        }
    return select_long_cases(args)


def window_indices(case, step_ms, query_frames=()):
    record = case["record"]
    stride = max(1, round(step_ms * record["fps"] / 1000.0))
    first, last = case["first_frame"], case["last_frame"]
    sampled = torch.arange(first, last + 1, stride)
    # Multi-time queries and the final frame are shared exactly between both runs.
    supplied = torch.as_tensor(query_frames, dtype=torch.long)
    sampled = torch.cat((sampled, supplied, torch.tensor([last]))).unique(sorted=True)
    native = torch.arange(first, last + 1)
    return native, sampled, stride


def decode_case(case, indices, *, return_error=False):
    record = case["record"]
    absolute = indices + record["frame_offset"]
    if record["adapter"] == "rgb_episode_cache":
        return MultiSourcePointTrackObjectVideoDataset._decode_cache(
            record["path"], absolute
        )
    decoder = read_video_frames if return_error else decode_video_frames
    frames = decoder(record["path"], absolute, record["fps"])
    return frames if isinstance(frames, VideoDecodeError) else frames.permute(0, 3, 1, 2)


def manual_query_points(case, args):
    points = read_json(args.queries_json)["points"][case["case_id"]]
    xy = torch.tensor([[point["x"], point["y"]] for point in points]).float()
    xy *= torch.tensor([case["width"] - 1, case["height"] - 1])
    return {
        "xy": xy,
        "labels": [point.get("label", "manual") for point in points],
        "frames": torch.full((len(xy),), case["anchor_frame"], dtype=torch.long),
    }


def load_tracker(args, device):
    from cotracker.predictor import CoTrackerPredictor

    model = (
        CoTrackerPredictor(
            checkpoint=args.tracker_checkpoint,
            v2=args.tracker_version == "2",
            offline=True,
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    metadata = {
        "checkpoint": str(Path(args.tracker_checkpoint).resolve()),
        "requested_version": args.tracker_version,
        "actual_model_class": type(model.model).__name__,
        "predictor_file": inspect.getfile(CoTrackerPredictor),
        "internal_resolution_hw": list(model.interp_shape),
        "mode": "offline",
        "backward_tracking": True,
        "precision": "float32",
        "visibility": "predictor_returned_boolean_not_calibrated_probability",
        "query_frames": "per-point supplied positions; exclude each point's own query frame from differences",
        "confidence_filter": "none",
        "world_model_used": False,
        "appearance_teacher_used": False,
        "point_identity_is_object_identity": False,
    }
    print(f"[tracker-review] tracker={json.dumps(metadata)}", flush=True)
    return model, metadata


@torch.no_grad()
def predict(model, rgb, indices, points, device, points_per_pass):
    local_anchors = torch.searchsorted(indices, points["frames"])
    tracks_parts, visibility_parts = [], []
    video = rgb[None].to(device=device, dtype=torch.float32)
    for first in range(0, len(points["xy"]), points_per_pass):
        last = first + points_per_pass
        queries = torch.cat(
            (local_anchors[first:last, None].float(), points["xy"][first:last]), -1
        )
        print(
            f"[tracker-review] point_batch={first}:{min(last, len(points['xy']))} frames={len(indices)}",
            flush=True,
        )
        tracks, visible = model(
            video, queries=queries[None].to(device), backward_tracking=True
        )
        tracks_parts.append(tracks[0].float().cpu())
        visibility_parts.append(visible[0].bool().cpu().reshape(len(indices), -1))
    tracks = torch.cat(tracks_parts, 1)
    visible = torch.cat(visibility_parts, 1)
    height, width = rgb.shape[-2:]
    in_bounds = (
        (tracks[..., 0] >= 0)
        & (tracks[..., 0] <= width - 1)
        & (tracks[..., 1] >= 0)
        & (tracks[..., 1] <= height - 1)
    )
    return {
        "tracks": tracks,
        "visibility": visible,
        "in_bounds": in_bounds,
        "frame_indices": indices.cpu(),
        "query_local_frames": local_anchors,
    }


def export_point_rows(path, native, sampled, labels, fps):
    common = torch.searchsorted(native["frame_indices"], sampled["frame_indices"])
    difference = (native["tracks"][common] - sampled["tracks"]).norm(dim=-1)
    paired_visible = native["visibility"][common] & sampled["visibility"]
    selected = paired_visible & torch.isfinite(difference)
    selected[sampled["query_local_frames"], torch.arange(len(labels))] = False
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "seconds",
                "point",
                "query_label",
                "is_query_frame",
                "native_x",
                "native_y",
                "native_visible",
                "native_in_bounds",
                "sampled_x",
                "sampled_y",
                "sampled_visible",
                "sampled_in_bounds",
                "native_sampled_distance_px_not_gt_error",
            ]
        )
        for frame, original in enumerate(sampled["frame_indices"].tolist()):
            for point, label in enumerate(labels):
                continuous_index = int(common[frame])
                writer.writerow(
                    [
                        original,
                        original / fps,
                        point,
                        label,
                        frame == int(sampled["query_local_frames"][point]),
                        *native["tracks"][continuous_index, point].tolist(),
                        bool(native["visibility"][continuous_index, point]),
                        bool(native["in_bounds"][continuous_index, point]),
                        *sampled["tracks"][frame, point].tolist(),
                        bool(sampled["visibility"][frame, point]),
                        bool(sampled["in_bounds"][frame, point]),
                        float(difference[frame, point]),
                    ]
                )
    distances = difference[selected]
    return {
        "nonfinite_pair_count": int((~torch.isfinite(difference)).sum()),
        "paired_visible_nonquery_count": int(selected.sum()),
        "paired_distance_px_p50": float(distances.median()) if len(distances) else None,
        "paired_distance_px_p95": float(distances.quantile(0.95))
        if len(distances)
        else None,
        "interpretation": "sampling consistency only; neither run is ground truth",
    }
