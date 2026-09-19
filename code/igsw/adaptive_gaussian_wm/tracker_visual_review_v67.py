"""Point-tracker-only review on the existing native multisource video index."""

from __future__ import annotations

from dataclasses import asdict
import csv
import inspect
import json
from pathlib import Path

import torch

from .continuous_field_sampling_v67 import stratified_query_coordinates_v67
from .multisource_point_track_dataset import MultiSourcePointTrackObjectVideoDataset
from .video_file_decoder import decode_video_frames


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
        return [
            case for case in manual["cases"] if manual["points"].get(case["case_id"])
        ]
    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        "8",
        str(max(args.steps_ms)),
        seed=args.seed,
        group_partition="held",
        held_group_stride=args.held_group_stride,
        preserve_native_rgb=True,
    )
    records = {record.sequence_index: record for record in dataset._records}
    cases = []
    for source_index, source in enumerate(dataset.source_names):
        indices = dataset.balanced_source_evaluation_indices(
            source_index, args.cases_per_source
        )
        for index in indices:
            print(
                f"[tracker-review] selecting source={source} sample={index}", flush=True
            )
            sample = dataset[(index, 8)]
            record = records[int(sample["sequence_index"])]
            actual_source = dataset.source_names[record.source_index]
            case_id = f"{actual_source}_seq{record.sequence_index}_item{index}"
            case = {
                "case_id": case_id,
                "source": actual_source,
                "requested_source": source,
                "requested_index": index,
                "decode_replaced": bool(sample["decode_replaced"]),
                "record": asdict(record),
                "group": dataset.sampling_group_names[record.group_index],
                "anchor_frame": int(sample["control_indices"][3]),
                "height": sample["video_rgb"].shape[-2],
                "width": sample["video_rgb"].shape[-1],
            }
            cases.append(case)
    return cases


def window_indices(case, step_ms):
    record = case["record"]
    stride = max(1, round(step_ms * record["fps"] / 1000.0))
    first = case["anchor_frame"] - 3 * stride
    sampled = first + torch.arange(8) * stride
    native = torch.arange(first, int(sampled[-1]) + 1)
    return native, sampled, stride


def decode_case(case, indices):
    record = case["record"]
    absolute = indices + record["frame_offset"]
    if record["adapter"] == "rgb_episode_cache":
        return MultiSourcePointTrackObjectVideoDataset._decode_cache(
            record["path"], absolute
        )
    return decode_video_frames(record["path"], absolute, record["fps"]).permute(
        0, 3, 1, 2
    )


def query_points(case, args, device):
    if args.queries_json:
        points = read_json(args.queries_json)["points"][case["case_id"]]
        xy = torch.tensor([[point["x"], point["y"]] for point in points], device=device)
        xy = xy.float() * torch.tensor(
            [case["width"] - 1, case["height"] - 1], device=device
        )
        labels = [point.get("label", "manual") for point in points]
    else:
        normalized = stratified_query_coordinates_v67(
            torch.tensor([case["record"]["sequence_index"]], device=device),
            args.grid_side,
            0.35,
        )[0]
        xy = (normalized + 1.0) * 0.5
        xy = xy * torch.tensor([case["width"] - 1, case["height"] - 1], device=device)
        labels = ["unlabelled_grid_point"] * len(xy)
    return xy, labels


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
        "anchor_frame": "supplied_query; official predictor overwrites it, exclude from accuracy",
        "confidence_filter": "none",
        "world_model_used": False,
        "appearance_teacher_used": False,
        "point_identity_is_object_identity": False,
    }
    print(f"[tracker-review] tracker={json.dumps(metadata)}", flush=True)
    return model, metadata


@torch.no_grad()
def predict(model, rgb, indices, anchor_frame, xy, device):
    local_anchor = int((indices == anchor_frame).nonzero()[0, 0])
    times = torch.full((len(xy), 1), float(local_anchor), device=device)
    queries = torch.cat((times, xy), dim=-1)[None]
    tracks, visible = model(
        rgb[None].to(device=device, dtype=torch.float32),
        queries=queries,
        backward_tracking=True,
    )
    tracks = tracks[0].float().cpu()
    visible = visible[0].bool().cpu().reshape(len(indices), len(xy))
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
        "anchor_local_frame": local_anchor,
    }


def export_point_rows(path, native, sampled, labels, fps):
    common = torch.searchsorted(native["frame_indices"], sampled["frame_indices"])
    difference = (native["tracks"][common] - sampled["tracks"]).norm(dim=-1)
    paired_visible = native["visibility"][common] & sampled["visibility"]
    selected = paired_visible & torch.isfinite(difference)
    selected[sampled["anchor_local_frame"]] = False
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
                        frame == sampled["anchor_local_frame"],
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
