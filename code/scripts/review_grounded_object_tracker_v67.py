#!/usr/bin/env python3
"""Grounded-SAM-2 sampling, CoTracker, role-aware review and shared teacher export."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers

from igsw.adaptive_gaussian_wm.grounded_tracker_masks_v67 import GroundedTrackerMasks
from igsw.adaptive_gaussian_wm.grounded_tracker_sampling_v67 import (
    build_grounded_queries,
    save_tensor,
)
from igsw.adaptive_gaussian_wm.grounded_tracker_export_v67 import (
    export_training_candidates,
    styled_tracks,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import (
    decode_case,
    export_point_rows,
    load_tracker,
    read_json,
    select_cases,
    window_indices,
    write_json,
    predict,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_media_v67 import (
    draw_points,
    render_pair,
    rgb_image,
    write_video,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_gallery_v67 import write_gallery
from review_point_tracker_v67 import pack_review, upload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--case_manifest", default="")
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--grounding_model", required=True)
    parser.add_argument("--sam_model", required=True)
    parser.add_argument("--tracker_version", choices=("2", "3"), default="3")
    parser.add_argument("--stage", choices=("run", "upload"), default="run")
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--cases_per_source", type=int, default=5)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--steps_ms", default="400")
    parser.add_argument("--query_every_seconds", type=float, default=2.0)
    parser.add_argument("--point_budget", type=int, default=2048)
    parser.add_argument("--points_per_pass", type=int, default=256)
    parser.add_argument("--minimum_region_points", type=int, default=4)
    parser.add_argument("--maximum_region_points", type=int, default=96)
    parser.add_argument("--robot_point_fraction", type=float, default=0.15)
    parser.add_argument("--other_context_fraction", type=float, default=0.05)
    parser.add_argument("--sam_grid_side", type=int, default=12)
    parser.add_argument("--sam_crop_divisions", type=int, default=2)
    parser.add_argument("--sam_prompt_batch", type=int, default=16)
    parser.add_argument("--sam_min_area", type=int, default=8)
    parser.add_argument("--sam_score_threshold", type=float, default=0.70)
    parser.add_argument("--sam_stability_threshold", type=float, default=0.90)
    parser.add_argument("--mask_dedup_iou", type=float, default=0.80)
    parser.add_argument("--robot_box_threshold", type=float, default=0.25)
    parser.add_argument("--robot_text_threshold", type=float, default=0.20)
    parser.add_argument("--robot_overlap_threshold", type=float, default=0.10)
    parser.add_argument("--scene_area_fraction", type=float, default=0.40)
    parser.add_argument("--max_masks_per_frame", type=int, default=48)
    parser.add_argument("--max_context_masks", type=int, default=8)
    parser.add_argument("--motion_floor_pixels", type=float, default=1.5)
    parser.add_argument("--motion_region_fraction", type=float, default=0.08)
    parser.add_argument("--motion_noise_multiplier", type=float, default=3.0)
    parser.add_argument("--minimum_visible_frames", type=int, default=6)
    parser.add_argument("--display_width", type=int, default=640)
    parser.add_argument("--reuse_completed", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument(
        "--wandb_entity",
        default="healenrenss-university-of-chinese-acadmic-and-science",
    )
    parser.add_argument("--wandb_name", default="grounded_object_tracker_v67")
    parser.add_argument("--wandb_dir", default=".")
    args = parser.parse_args()
    args.steps_ms = [int(value) for value in args.steps_ms.split(",")]
    args.queries_json = ""
    return args


def choose_cases(args, out, configuration):
    plan_path = out / "cases.json"
    if (
        args.reuse_completed
        and plan_path.is_file()
        and read_json(plan_path)["configuration"] == configuration
    ):
        plan = read_json(plan_path)
        return plan["cases"], plan["selection_report"]
    if args.case_manifest:
        prior = read_json(args.case_manifest)
        cases = deepcopy(prior["cases"])
        selection = {
            "mode": "same cases as baseline",
            "case_manifest": args.case_manifest,
            "selected": len(cases),
            "replacement": "none",
        }
        for case in cases:
            for key in ("point_count", "motion_views", "sampling_kind"):
                case.pop(key, None)
    else:
        cases, selection = select_cases(args)
    write_json(
        plan_path,
        {"configuration": configuration, "cases": cases, "selection_report": selection},
    )
    return cases, selection


def all_queries_video(directory, rgb, prediction, case, queries, display_width):
    ids = torch.arange(len(queries["xy"]))
    styled = styled_tracks(prediction, queries, ids)
    styled["legend"] = (
        "green=object candidate; orange=robot; purple=unknown; blue=scene"
    )

    def frames():
        for frame, image in enumerate(rgb):
            overlay = draw_points(
                rgb_image(image),
                styled,
                frame,
                case["record"]["fps"],
                queries["labels"],
                "ALL queries, including small/static candidates",
            )
            yield overlay.resize(
                (display_width, round(overlay.height * display_width / overlay.width))
            )

    write_video(directory / "all_queries.mp4", frames(), case["record"]["fps"])


def run_review(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    configuration = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("wandb_") and key not in ("reuse_completed", "stage")
    }
    cases, selection = choose_cases(args, out, configuration)
    device = torch.device("cuda:0")
    segmenter = GroundedTrackerMasks(args, device)
    tracker_model, tracker = load_tracker(args, device)
    tracker.update(
        {
            "points_per_pass": args.points_per_pass,
            "temporal_chunking": "none",
            "query_source": "GroundingDINO+SAM2 native mask interiors",
        }
    )
    segmenter_metadata = {
        "grounding_model": args.grounding_model,
        "sam_model": args.sam_model,
        "transformers_version": transformers.__version__,
        "segmentation": "query-frame image masks; CoTracker propagates points, not masks",
        "resolution": "native RGB and overlapping crops; pretrained internal resize remains",
    }
    results, training_entries = [], []

    def save_progress():
        write_json(
            out / "cases.json",
            {
                "configuration": configuration,
                "cases": cases,
                "selection_report": selection,
            },
        )
        write_json(
            out / "summary.json",
            {
                "configuration": configuration,
                "cases": cases,
                "results": results,
                "tracker": tracker,
                "segmenter": segmenter_metadata,
                "selection_report": selection,
                "status": "ready_for_human_review",
                "independent_accuracy": "not_measured",
            },
        )
        write_json(
            out / "training_manifest.json",
            {
                "contract": "grounded_object_motion_teacher_v1",
                "root": str(out.resolve()),
                "entries": training_entries,
                "pseudo_labels": True,
                "teacher_only": True,
                "future_used_for_selection": True,
            },
        )
        write_gallery(out, cases, results, args.source_revision)

    for case in cases:
        directory = out / case["case_id"]
        directory.mkdir(exist_ok=True)
        completed = [
            directory / f"step_{step}ms" / "result.json" for step in args.steps_ms
        ]
        if args.reuse_completed and all(path.is_file() for path in completed):
            rows = [read_json(path) for path in completed]
            if all(row["configuration"] == configuration for row in rows):
                results.extend(rows)
                training_entries.extend(row["training_entry"] for row in rows)
                print(f"[grounded-tracker] reuse={case['case_id']}", flush=True)
                save_progress()
                continue
        indices, _, _ = window_indices(case, args.steps_ms[0])
        print(
            f"[grounded-tracker] decode={case['case_id']} frames={len(indices)}",
            flush=True,
        )
        rgb = decode_case(case, indices)
        case["height"], case["width"] = list(rgb.shape[-2:])
        write_video(
            directory / "source.mp4",
            (rgb_image(frame) for frame in rgb),
            case["record"]["fps"],
        )
        rgb_image(rgb[case["anchor_frame"] - case["first_frame"]]).save(
            directory / "anchor.png"
        )
        sampling_path = directory / "sampling.json"
        if (
            args.reuse_completed
            and sampling_path.is_file()
            and read_json(sampling_path)["configuration"] == configuration
        ):
            queries = torch.load(
                directory / "queries.pt", map_location="cpu", weights_only=False
            )
            sampling = read_json(sampling_path)
        else:
            queries, sampling = build_grounded_queries(
                rgb, case, args, segmenter, directory, configuration
            )
        case.update(
            {
                "point_count": len(queries["xy"]),
                "motion_views": sampling["views"],
                "sampling_kind": sampling["kind"],
            }
        )
        print(
            f"[grounded-tracker] queries={case['case_id']} roles={sampling['role_counts']}",
            flush=True,
        )
        if not len(queries["xy"]):
            save_progress()
            continue
        native_path = directory / "native_tracks.pt"
        native_record = (
            torch.load(native_path, map_location="cpu", weights_only=False)
            if args.reuse_completed and native_path.is_file()
            else None
        )
        if (
            native_record is not None
            and native_record["configuration"] == configuration
        ):
            native = native_record["prediction"]
        else:
            native = predict(
                tracker_model, rgb, indices, queries, device, args.points_per_pass
            )
            save_tensor(
                native_path, {"configuration": configuration, "prediction": native}
            )
        for step in args.steps_ms:
            pair_dir = directory / f"step_{step}ms"
            pair_dir.mkdir(exist_ok=True)
            _, sampled_indices, stride = window_indices(case, step, queries["frames"])
            parameters = {
                "configuration": configuration,
                "tracker": tracker,
                "case": deepcopy(case),
                "step_ms": step,
                "xy": queries["xy"].tolist(),
                "labels": queries["labels"],
                "query_frames": queries["frames"].tolist(),
                "sampled_frame_indices": sampled_indices.tolist(),
                "source_revision": args.source_revision,
            }
            raw_path = pair_dir / "tracks.pt"
            cached = (
                torch.load(raw_path, map_location="cpu", weights_only=False)
                if args.reuse_completed and raw_path.is_file()
                else None
            )
            started = time.monotonic()
            if cached is not None and cached["parameters"] == parameters:
                sampled = cached["sampled"]
            else:
                sampled = predict(
                    tracker_model,
                    rgb[sampled_indices - indices[0]],
                    sampled_indices,
                    queries,
                    device,
                    args.points_per_pass,
                )
                save_tensor(
                    raw_path,
                    {"parameters": parameters, "native": native, "sampled": sampled},
                )
            diagnostic = export_point_rows(
                pair_dir / "point_rows.csv",
                native,
                sampled,
                queries["labels"],
                case["record"]["fps"],
            )
            ids, point_selection = export_training_candidates(
                pair_dir, case, queries, native, sampled, args, parameters
            )
            print(
                f"[grounded-tracker] render={case['case_id']} targets={point_selection['object_motion_target_count']} context={point_selection['context_count']}",
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
            all_queries_video(pair_dir, rgb, native, case, queries, args.display_width)
            media["all_queries"] = "all_queries.mp4"
            entry = {
                "case_id": case["case_id"],
                "source": case["source"],
                "step_ms": step,
                "path": str((pair_dir / "training_candidates.pt").relative_to(out)),
                "queries": len(queries["xy"]),
                "object_targets": point_selection["object_motion_target_count"],
                "context_points": point_selection["context_count"],
                "sampling": str(sampling_path.relative_to(out)),
            }
            row = {
                "case_id": case["case_id"],
                "source": case["source"],
                "group": case["group"],
                "directory": str(pair_dir.relative_to(out)),
                "step_ms": step,
                "actual_step_ms": stride / case["record"]["fps"] * 1000,
                "clip_seconds": case["clip_seconds"],
                "point_count": len(queries["xy"]),
                "sampled_frame_count": len(sampled_indices),
                "native_frame_count": len(indices),
                "native_resolution_hw": [case["height"], case["width"]],
                "parameters": parameters,
                "configuration": configuration,
                "sampling_consistency": diagnostic,
                "grounded_selection": point_selection,
                "media": media,
                "crop_boxes": boxes,
                "training_entry": entry,
                "status": "ready_for_human_review",
                "sampled_inference_and_render_seconds": time.monotonic() - started,
            }
            write_json(pair_dir / "result.json", row)
            results.append(row)
            training_entries.append(entry)
            save_progress()
        del rgb
    save_progress()
    pack_review(out)
    print(
        f"[grounded-tracker] complete gallery={out / 'index.html'} bundle={out / 'review_bundle.zip'}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.stage == "run":
        run_review(args)
    upload(args)


if __name__ == "__main__":
    main()
