#!/usr/bin/env python3
"""Rerender an existing tracker review on CPU; never run CoTracker or resample."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import av
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.adaptive_gaussian_wm.tracker_motion_display_v67 import (
    select_moving_tracks,
    subset_tracks,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json
from igsw.adaptive_gaussian_wm.tracker_visual_review_media_v67 import render_pair
from igsw.adaptive_gaussian_wm.tracker_visual_review_gallery_v67 import write_gallery
from review_point_tracker_v67 import pack_review, upload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_review", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--minimum_motion_pixels", type=float, default=12.0)
    parser.add_argument("--minimum_motion_fraction", type=float, default=0.02)
    parser.add_argument("--minimum_visible_frames", type=int, default=6)
    parser.add_argument("--trails_seconds", type=float, default=0.0)
    parser.add_argument("--display_width", type=int, default=640)
    parser.add_argument("--reuse_completed", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb_project", default=os.environ.get("WANDB_PROJECT", "instruct-gs-world")
    )
    parser.add_argument(
        "--wandb_entity",
        default=os.environ.get(
            "WANDB_ENTITY", "healenrenss-university-of-chinese-acadmic-and-science"
        ),
    )
    parser.add_argument("--wandb_name", default="tracker_motion_review_v67_moving_only")
    parser.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", "."))
    return parser.parse_args()


def read_display_rgb(path, case):
    # The review's own MP4 makes rerendering independent of the training data mount.
    with av.open(str(path)) as container:
        images = [
            torch.from_numpy(frame.to_ndarray(format="rgb24"))[
                : case["height"], : case["width"]
            ]
            for frame in container.decode(video=0)
        ]
    return torch.stack(images).permute(0, 3, 1, 2)


def main():
    args = parse_args()
    source, out = Path(args.input_review).resolve(), Path(args.out).resolve()
    original = read_json(source / "summary.json")
    out.mkdir(parents=True, exist_ok=True)
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    settings = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("wandb_") and key != "reuse_completed"
    }
    results = []
    for case in original["cases"]:
        case_dir = out / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        source_case = source / case["case_id"]
        for name in ("source.mp4", "anchor.png", "sampling.json", "queries.pt"):
            shutil.copy2(source_case / name, case_dir / name)
        masks = source_case / "motion_masks"
        if masks.is_dir():
            shutil.copytree(masks, case_dir / "motion_masks", dirs_exist_ok=True)
        rgb = None
        for old_row in original["results"]:
            if old_row["case_id"] != case["case_id"]:
                continue
            source_pair, pair_dir = (
                source / old_row["directory"],
                out / old_row["directory"],
            )
            pair_dir.mkdir(parents=True, exist_ok=True)
            result_path = pair_dir / "result.json"
            if (
                args.reuse_completed
                and result_path.is_file()
                and read_json(result_path).get("render_settings") == settings
            ):
                results.append(read_json(result_path))
                print(f"[tracker-moving] reuse={old_row['directory']}", flush=True)
                continue
            if rgb is None:
                rgb = read_display_rgb(source_case / "source.mp4", case)
            cached = torch.load(
                source_pair / "tracks.pt", map_location="cpu", weights_only=False
            )
            point_ids, selection = select_moving_tracks(
                cached["native"],
                case["height"],
                case["width"],
                args.minimum_motion_pixels,
                args.minimum_motion_fraction,
                args.minimum_visible_frames,
            )
            write_json(pair_dir / "motion_filter.json", selection)
            parameters = cached["parameters"]
            labels = [parameters["labels"][point] for point in point_ids.tolist()]
            xy = torch.tensor(parameters["xy"], dtype=torch.float32)[point_ids]
            native = subset_tracks(cached["native"], point_ids)
            sampled = subset_tracks(cached["sampled"], point_ids)
            print(
                f"[tracker-moving] render={old_row['directory']} shown={len(point_ids)}/{selection['raw_point_count']} threshold_px={selection['threshold_px']:.2f}",
                flush=True,
            )
            media, boxes = render_pair(
                pair_dir,
                rgb,
                native,
                sampled,
                case,
                xy,
                labels,
                args.display_width,
                args.trails_seconds,
            )
            for name in ("tracks.pt", "point_rows.csv"):
                shutil.copy2(source_pair / name, pair_dir / name)
            row = deepcopy(old_row)
            row.update(
                {
                    "media": media,
                    "crop_boxes": boxes,
                    "display_filter": selection,
                    "render_settings": settings,
                    "render_source_rgb": "archived source.mp4, display only",
                    "sampling_consistency_population": "all original points, not the display-filtered subset",
                }
            )
            write_json(result_path, row)
            results.append(row)
            write_gallery(out, original["cases"], results, args.source_revision)
        del rgb
    summary = deepcopy(original)
    summary.update(
        {
            "results": results,
            "configuration": {"operation": "CPU display-only rerender", **vars(args)},
            "source_configuration": original["configuration"],
            "input_review": str(source),
            "tracking_recomputed": False,
            "raw_tracks_preserved": True,
        }
    )
    write_json(out / "summary.json", summary)
    shutil.copy2(source / "cases.json", out / "cases.json")
    write_gallery(out, original["cases"], results, args.source_revision)
    pack_review(out)
    print(
        f"[tracker-moving] complete gallery={out / 'index.html'} bundle={out / 'review_bundle.zip'}",
        flush=True,
    )
    upload(args)


if __name__ == "__main__":
    main()
