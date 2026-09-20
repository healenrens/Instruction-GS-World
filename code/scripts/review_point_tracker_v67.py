#!/usr/bin/env python3
"""Single-GPU tracker visualization; no world-model or feature teacher is loaded."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import (
    decode_case,
    export_point_rows,
    load_tracker,
    predict,
    manual_query_points,
    read_json,
    select_cases,
    window_indices,
    write_json,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_media_v67 import (
    render_pair,
    rgb_image,
    write_video,
)
from igsw.adaptive_gaussian_wm.tracker_visual_review_gallery_v67 import write_gallery
from igsw.adaptive_gaussian_wm.tracker_motion_masks_v67 import build_motion_queries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "upload"), default="run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--tracker_version", choices=("2", "3"), default="3")
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--cases_per_source", type=int, default=5)
    parser.add_argument("--steps_ms", default="400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--point_budget", type=int, default=1024)
    parser.add_argument("--points_per_pass", type=int, default=256)
    parser.add_argument("--query_every_seconds", type=float, default=2.0)
    parser.add_argument("--motion_pair_seconds", type=float, default=0.2)
    parser.add_argument("--motion_min_px", type=float, default=0.75)
    parser.add_argument("--mask_min_area", type=int, default=9)
    parser.add_argument("--queries_json", default="")
    parser.add_argument("--display_width", type=int, default=640)
    parser.add_argument("--reuse_completed", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "--wandb_project", default=os.environ.get("WANDB_PROJECT", "instruct-gs-world")
    )
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_name", default="tracker_visual_review_v67")
    parser.add_argument("--wandb_dir", default=".")
    args = parser.parse_args()
    args.steps_ms = [int(value) for value in args.steps_ms.split(",")]
    return args


def pack_review(out):
    files = [
        path
        for path in out.rglob("*")
        if path.is_file()
        and path.suffix in (".html", ".mp4", ".png", ".csv", ".json", ".pt", ".npz")
    ]
    with zipfile.ZipFile(
        out / "review_bundle.zip", "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(out))


def upload(args):
    import wandb

    out = Path(args.out)
    summary = read_json(out / "summary.json")
    if args.wandb_mode == "disabled":
        return
    # Reviews must not attach to an inherited training run, including on reupload.
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group="point-tracker-visual-review-v67",
        job_type="tracker-visual-review",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=summary["configuration"],
    )
    table = wandb.Table(
        columns=[
            "case",
            "source",
            "group",
            "step_ms",
            "clip_seconds",
            "query_points",
            "displayed_points",
            "camera",
            "native_hw",
            "model_hw",
            "native_video",
            "sampled_video",
            "comparison_video",
            "full_trajectories",
            "manual_query_reference_frame",
            "paired_visible_nonquery_count",
            "distance_p50_px_not_accuracy",
            "distance_p95_px_not_accuracy",
        ]
    )
    for row in summary["results"]:
        directory = out / row["directory"]
        diagnostic = row["sampling_consistency"]
        table.add_data(
            row["case_id"],
            row["source"],
            row["group"],
            row["actual_step_ms"],
            row["clip_seconds"],
            row["point_count"],
            row.get("grounded_selection", row.get("display_filter", {})).get(
                "shown_point_count", row["point_count"]
            ),
            row["parameters"]["case"]["camera"],
            row["native_resolution_hw"],
            summary["tracker"]["internal_resolution_hw"],
            wandb.Video(str(directory / "native.mp4"), format="mp4"),
            wandb.Video(str(directory / "sampled.mp4"), format="mp4"),
            wandb.Video(str(directory / "comparison.mp4"), format="mp4"),
            wandb.Image(str(directory / "trajectories.png"))
            if "trajectories" in row.get("media", {})
            else None,
            wandb.Image(str(directory / "anchor.png")),
            diagnostic["paired_visible_nonquery_count"],
            diagnostic["paired_distance_px_p50"],
            diagnostic["paired_distance_px_p95"],
        )
    run.log({"tracker_review/cases": table})
    grounded_rows = [row for row in summary["results"] if "grounded_selection" in row]
    if grounded_rows:
        roles = wandb.Table(
            columns=[
                "case",
                "source",
                "object_motion_targets",
                "robot_queries",
                "unknown_queries",
                "uncertain_moving_candidates",
                "scene_queries",
                "raw_queries",
                "all_queries_video",
            ]
        )
        for row in grounded_rows:
            selection = row["grounded_selection"]
            counts = selection["role_counts"]
            roles.add_data(
                row["case_id"],
                row["source"],
                selection["object_motion_target_count"],
                counts.get("robot_context", 0),
                counts.get("unknown", 0),
                selection.get("uncertain_motion_candidate_count"),
                counts.get("scene_context", 0),
                selection["raw_point_count"],
                wandb.Video(
                    str(out / row["directory"] / "all_queries.mp4"), format="mp4"
                ),
            )
        run.log({"tracker_review/grounded_roles": roles})
    source_rows = [case for case in summary["cases"] if "source_review" in case]
    if source_rows:
        sources = wandb.Table(
            columns=[
                "case",
                "source",
                "indexed_path",
                "episode_file_frames",
                "tracked_file_frames",
                "original_provenance",
                "whole_episode_overview",
            ]
        )
        for case in source_rows:
            source = case["source_review"]
            sources.add_data(
                case["case_id"],
                case["source"],
                source["indexed_record"]["path"],
                source["episode_file_frame_range"],
                source["selected_file_frames"],
                source["raw_original_provenance"],
                wandb.Image(str(out / case["case_id"] / "episode_overview.png")),
            )
        run.log({"tracker_review/indexed_sources": sources})
    masks = wandb.Table(
        columns=[
            "case",
            "source",
            "frame",
            "mask_fraction",
            "points",
            "motion_mask_and_queries",
        ]
    )
    for case in summary["cases"]:
        directory = out / case["case_id"]
        proposals = read_json(directory / "sampling.json")
        for view in proposals["views"]:
            masks.add_data(
                case["case_id"],
                case["source"],
                view["frame"],
                view["mask_fraction"],
                view["point_count"],
                wandb.Image(str(directory / view["overlay"])),
            )
    run.log({"tracker_review/motion_masks": masks})
    artifact = wandb.Artifact(
        name=f"tracker-visual-review-{run.id}", type="tracker-review"
    )
    artifact.add_file(str(out / "review_bundle.zip"))
    artifact.add_file(str(out / "summary.json"))
    run.log_artifact(artifact)
    run.summary.update(
        {
            "review/case_count": len(summary["cases"]),
            "review/comparison_count": len(summary["results"]),
            "review/empty_motion_cases": sum(
                case.get("point_count", 0) == 0 for case in summary["cases"]
            ),
            "review/independent_accuracy": "not_measured",
            "review/local_gallery": str(out / "index.html"),
        }
    )
    write_json(out / "wandb_run.json", {"id": run.id, "url": run.url})
    print(f"[tracker-review] wandb={run.url}", flush=True)
    run.finish()


def run_review(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    selection = {
        name: configuration[name]
        for name in (
            "data_index",
            "cases_per_source",
            "steps_ms",
            "held_group_stride",
            "seed",
            "queries_json",
            "clip_seconds",
        )
    }
    plan_path = out / "cases.json"
    if (
        args.reuse_completed
        and not args.queries_json
        and plan_path.is_file()
        and read_json(plan_path)["selection"] == selection
    ):
        plan = read_json(plan_path)
        cases, selection_report = plan["cases"], plan["selection_report"]
    else:
        cases, selection_report = select_cases(args)
        write_json(
            plan_path,
            {
                "selection": selection,
                "cases": cases,
                "selection_report": selection_report,
            },
        )
    print(f"[tracker-review] selection={selection_report}", flush=True)
    print(
        f"[tracker-review] selected={len(cases)} cases; native/sampled pairs={len(cases) * len(args.steps_ms)}",
        flush=True,
    )
    device = torch.device("cuda:0")
    model, tracker = load_tracker(args, device)
    tracker["points_per_pass"] = args.points_per_pass
    tracker["temporal_chunking"] = "none; each point batch sees the entire >=10s clip"
    tracker["query_batching_note"] = (
        "joint attention is within each point batch plus official support queries"
    )
    results = []
    for case in cases:
        case_id = case["case_id"]
        all_indices, _, _ = window_indices(case, max(args.steps_ms))
        case_dir = out / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[tracker-review] decode={case_id} path={case['record']['path']} seconds={case['clip_seconds']:.3f}",
            flush=True,
        )
        video = decode_case(case, all_indices)
        case["height"], case["width"] = list(video.shape[-2:])
        source_video = case_dir / "source.mp4"
        if not args.reuse_completed or not source_video.is_file():
            write_video(
                source_video,
                (rgb_image(frame) for frame in video),
                case["record"]["fps"],
            )
        sampling_config = {
            key: configuration[key]
            for key in (
                "point_budget",
                "query_every_seconds",
                "motion_pair_seconds",
                "motion_min_px",
                "mask_min_area",
                "seed",
                "source_revision",
                "queries_json",
            )
        }
        sampling_path, points_path = case_dir / "sampling.json", case_dir / "queries.pt"
        if args.queries_json:
            points = manual_query_points(case, args)
            proposal = {
                "kind": "manual",
                "views": [],
                "actual_points": len(points["xy"]),
            }
        elif (
            args.reuse_completed
            and points_path.is_file()
            and sampling_path.is_file()
            and read_json(sampling_path)["configuration"] == sampling_config
        ):
            points = torch.load(points_path, map_location="cpu", weights_only=False)
            proposal = read_json(sampling_path)
        else:
            print(
                f"[tracker-review] motion_masks={case_id} budget={args.point_budget}",
                flush=True,
            )
            points, proposal = build_motion_queries(video, case, args, case_dir)
        proposal["configuration"] = sampling_config
        write_json(sampling_path, proposal)
        temporary_points = points_path.with_suffix(".pt.tmp")
        torch.save(points, temporary_points)
        temporary_points.replace(points_path)
        xy, labels = points["xy"], points["labels"]
        case["point_count"] = len(labels)
        case["motion_views"] = proposal["views"]
        rgb_image(video[case["anchor_frame"] - case["first_frame"]]).save(
            case_dir / "anchor.png"
        )
        write_json(
            plan_path,
            {
                "selection": selection,
                "cases": cases,
                "selection_report": selection_report,
            },
        )
        if len(labels) == 0:
            print(
                f"[tracker-review] no_motion_queries={case_id}; raw video and empty masks retained",
                flush=True,
            )
            write_gallery(out, cases, results, args.source_revision)
            continue
        native = None
        for step_ms in args.steps_ms:
            native_indices, sampled_indices, stride = window_indices(
                case, step_ms, points["frames"]
            )
            directory = case_dir / f"step_{step_ms}ms"
            directory.mkdir(parents=True, exist_ok=True)
            parameters = {
                "tracker": tracker,
                "case": case,
                "step_ms": step_ms,
                "xy": xy.cpu().tolist(),
                "labels": labels,
                "query_frames": points["frames"].tolist(),
                "sampled_frame_indices": sampled_indices.tolist(),
                "sampling_configuration": sampling_config,
                "source_revision": args.source_revision,
                "display_width": args.display_width,
            }
            result_path = directory / "result.json"
            if (
                args.reuse_completed
                and result_path.is_file()
                and read_json(result_path)["parameters"] == parameters
            ):
                print(f"[tracker-review] reuse={case_id}/{step_ms}ms", flush=True)
                results.append(read_json(result_path))
                continue
            native_rgb = video
            sampled_rgb = video[sampled_indices - all_indices[0]]
            raw_path = directory / "tracks.pt"
            cached = (
                torch.load(raw_path, map_location="cpu", weights_only=False)
                if args.reuse_completed and raw_path.is_file()
                else None
            )
            if cached is not None and cached["parameters"] == parameters:
                native, sampled = cached["native"], cached["sampled"]
            else:
                print(
                    f"[tracker-review] inference={case_id} step_ms={step_ms} native_frames={len(native_rgb)} sampled_frames={len(sampled_rgb)} points={len(xy)}",
                    flush=True,
                )
                started = time.monotonic()
                if native is None:
                    native = predict(
                        model,
                        native_rgb,
                        native_indices,
                        points,
                        device,
                        args.points_per_pass,
                    )
                sampled = predict(
                    model,
                    sampled_rgb,
                    sampled_indices,
                    points,
                    device,
                    args.points_per_pass,
                )
                cached = {
                    "parameters": parameters,
                    "native": native,
                    "sampled": sampled,
                    "seconds": time.monotonic() - started,
                }
                temporary = raw_path.with_suffix(".pt.tmp")
                torch.save(cached, temporary)
                temporary.replace(raw_path)
            diagnostic = export_point_rows(
                directory / "point_rows.csv",
                native,
                sampled,
                labels,
                case["record"]["fps"],
            )
            print(f"[tracker-review] rendering={case_id}/{step_ms}ms", flush=True)
            media, boxes = render_pair(
                directory,
                native_rgb,
                native,
                sampled,
                case,
                xy.cpu(),
                labels,
                args.display_width,
            )
            row = {
                "case_id": case_id,
                "source": case["source"],
                "group": case["group"],
                "directory": str(directory.relative_to(out)),
                "step_ms": step_ms,
                "actual_step_ms": stride / case["record"]["fps"] * 1000.0,
                "clip_seconds": case["clip_seconds"],
                "point_count": len(labels),
                "sampled_frame_count": len(sampled_indices),
                "native_frame_count": len(native_indices),
                "sampled_intervals_ms": (
                    (sampled_indices[1:] - sampled_indices[:-1])
                    / case["record"]["fps"]
                    * 1000
                ).tolist(),
                "native_resolution_hw": [case["height"], case["width"]],
                "sampling_consistency": diagnostic,
                "parameters": parameters,
                "inference_seconds": cached["seconds"],
                "media": media,
                "crop_boxes": boxes,
                "status": "ready_for_human_review",
            }
            write_json(result_path, row)
            results.append(row)
            write_gallery(out, cases, results, args.source_revision)
            print(
                f"[tracker-review] completed={len(results)}/{len(cases) * len(args.steps_ms)} gallery={out / 'index.html'}",
                flush=True,
            )
        del video
    summary = {
        "configuration": configuration,
        "tracker": tracker,
        "selection_report": selection_report,
        "cases": cases,
        "results": results,
        "status": "ready_for_human_review",
        "independent_accuracy": "not_measured; no human point ground truth supplied",
    }
    write_json(out / "summary.json", summary)
    write_gallery(out, cases, results, args.source_revision)
    pack_review(out)
    print(
        f"[tracker-review] local_done gallery={out / 'index.html'} bundle={out / 'review_bundle.zip'}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.stage == "run":
        run_review(args)
    upload(args)


if __name__ == "__main__":
    main()
