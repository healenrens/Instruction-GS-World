#!/usr/bin/env python3
"""Reselect grounded motion targets and redraw on CPU from archived review outputs."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.adaptive_gaussian_wm.grounded_tracker_export_v67 import (
    export_training_candidates,
    styled_tracks,
)
from igsw.adaptive_gaussian_wm.grounded_tracker_selection_media_v67 import (
    render_object_selection,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json
from igsw.adaptive_gaussian_wm.tracker_visual_review_media_v67 import render_pair
from igsw.adaptive_gaussian_wm.tracker_visual_review_gallery_v67 import write_gallery
from render_moving_tracker_review_v67 import read_display_rgb
from review_point_tracker_v67 import pack_review, upload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_review", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--motion_top_fraction", type=float, default=0.5)
    parser.add_argument("--display_width", type=int, default=640)
    parser.add_argument("--reuse_completed", type=int, choices=(0, 1), default=1)
    parser.add_argument("--stage", choices=("run", "upload"), default="run")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument(
        "--wandb_entity",
        default="healenrenss-university-of-chinese-acadmic-and-science",
    )
    parser.add_argument("--wandb_name", default="grounded_motion_topk_v67")
    parser.add_argument("--wandb_dir", default=".")
    return parser.parse_args()


def run(args):
    source, out = Path(args.input_review).resolve(), Path(args.out).resolve()
    original = read_json(source / "summary.json")
    out.mkdir(parents=True, exist_ok=True)
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    settings = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("wandb_") and key not in ("reuse_completed", "stage")
    }
    config = {
        **original["configuration"],
        **settings,
        "operation": "CPU target reselection and redraw, not tracking inference",
    }
    selection_args = SimpleNamespace(**config)
    cases = deepcopy(original["cases"])
    results, entries = [], []

    def save_progress():
        summary = {
            **original,
            "cases": cases,
            "results": results,
            "configuration": config,
            "source_configuration": original["configuration"],
            "tracking_recomputed": False,
            "raw_tracks_preserved": True,
            "input_review": str(source),
            "status": "ready_for_human_review",
        }
        write_json(out / "summary.json", summary)
        write_json(
            out / "cases.json",
            {
                "configuration": config,
                "cases": cases,
                "selection_report": original["selection_report"],
            },
        )
        write_json(
            out / "training_manifest.json",
            {
                "contract": "grounded_object_motion_teacher_v3",
                "root": str(out),
                "entries": entries,
                "pseudo_labels": True,
                "teacher_only": True,
                "future_used_for_selection": True,
            },
        )
        write_gallery(out, cases, results, args.source_revision)

    for case in cases:
        source_case, case_dir = source / case["case_id"], out / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        for path in source_case.iterdir():
            if path.is_file() and path.suffix in (".json", ".pt", ".png", ".mp4"):
                shutil.copy2(path, case_dir / path.name)
        shutil.copytree(
            source_case / "grounded_masks",
            case_dir / "grounded_masks",
            dirs_exist_ok=True,
        )
        rgb = None
        for previous in original["results"]:
            if previous["case_id"] != case["case_id"]:
                continue
            source_pair, pair_dir = (
                source / previous["directory"],
                out / previous["directory"],
            )
            pair_dir.mkdir(parents=True, exist_ok=True)
            result_path = pair_dir / "result.json"
            if args.reuse_completed and result_path.is_file():
                completed = read_json(result_path)
                if completed["refilter_settings"] == settings:
                    results.append(completed)
                    entries.append(completed["training_entry"])
                    print(f"[grounded-topk] reuse={previous['directory']}", flush=True)
                    continue
            if rgb is None:
                rgb = read_display_rgb(source_case / "source.mp4", case)
            payload = torch.load(
                source_pair / "training_candidates.pt",
                map_location="cpu",
                weights_only=False,
            )
            queries, native, sampled = (
                payload["queries"],
                payload["native"],
                payload["sampled"],
            )
            parameters = {
                **payload["parameters"],
                "target_selection_configuration": config,
            }
            ids, selection = export_training_candidates(
                pair_dir,
                case,
                queries,
                native,
                sampled,
                selection_args,
                parameters,
                payload["role_evidence"],
            )
            print(
                f"[grounded-topk] {previous['directory']} object_candidates={selection['object_motion_candidate_count_before_topk']} "
                f"retained={selection['object_motion_target_count']} context={selection['context_count']}",
                flush=True,
            )
            media, boxes = render_pair(
                pair_dir,
                rgb,
                styled_tracks(native, queries, ids),
                styled_tracks(sampled, queries, ids),
                case,
                queries["xy"][ids],
                [queries["labels"][i] for i in ids.tolist()],
                args.display_width,
            )
            media.update(
                render_object_selection(
                    pair_dir, rgb, native, case, queries, selection, args.display_width
                )
            )
            for name in ("tracks.pt", "point_rows.csv", "all_queries.mp4"):
                shutil.copy2(source_pair / name, pair_dir / name)
            media["all_queries"] = "all_queries.mp4"
            entry = {
                **previous["training_entry"],
                "object_targets": selection["object_motion_target_count"],
                "object_candidates_before_topk": selection[
                    "object_motion_candidate_count_before_topk"
                ],
                "motion_top_fraction": args.motion_top_fraction,
            }
            row = {
                **previous,
                "configuration": config,
                "refilter_settings": settings,
                "grounded_selection": selection,
                "training_entry": entry,
                "media": media,
                "crop_boxes": boxes,
                "render_source_rgb": "archived source.mp4, display only; all measurements use original tracks",
            }
            write_json(result_path, row)
            results.append(row)
            entries.append(entry)
            save_progress()
        del rgb
        save_progress()
    pack_review(out)
    print(
        f"[grounded-topk] complete gallery={out / 'index.html'} bundle={out / 'review_bundle.zip'}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.stage == "run":
        run(args)
    upload(args)


if __name__ == "__main__":
    main()
