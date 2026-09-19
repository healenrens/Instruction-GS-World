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
    query_points,
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "upload"), default="run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--tracker_version", choices=("2", "3"), default="3")
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--cases_per_source", type=int, default=5)
    parser.add_argument("--steps_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--grid_side", type=int, default=16)
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
        and path.suffix in (".html", ".mp4", ".png", ".csv", ".json", ".pt")
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
            "native_hw",
            "model_hw",
            "native_video",
            "sampled_video",
            "comparison_video",
            "query_frame",
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
            row["native_resolution_hw"],
            summary["tracker"]["internal_resolution_hw"],
            wandb.Video(str(directory / "native.mp4"), format="mp4"),
            wandb.Video(str(directory / "sampled.mp4"), format="mp4"),
            wandb.Video(str(directory / "comparison.mp4"), format="mp4"),
            wandb.Image(str(directory / "anchor.png")),
            diagnostic["paired_visible_nonquery_count"],
            diagnostic["paired_distance_px_p50"],
            diagnostic["paired_distance_px_p95"],
        )
    run.log({"tracker_review/cases": table})
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
        )
    }
    plan_path = out / "cases.json"
    if (
        args.reuse_completed
        and not args.queries_json
        and plan_path.is_file()
        and read_json(plan_path)["selection"] == selection
    ):
        cases = read_json(plan_path)["cases"]
    else:
        cases = select_cases(args)
        write_json(plan_path, {"selection": selection, "cases": cases})
    print(
        f"[tracker-review] selected={len(cases)} cases; native/sampled pairs={len(cases) * len(args.steps_ms)}",
        flush=True,
    )
    device = torch.device("cuda:0")
    model, tracker = load_tracker(args, device)
    results = []
    for case in cases:
        case_id = case["case_id"]
        xy, labels = query_points(case, args, device)
        all_indices, _, _ = window_indices(case, max(args.steps_ms))
        case_dir = out / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        video = None
        for step_ms in args.steps_ms:
            native_indices, sampled_indices, stride = window_indices(case, step_ms)
            directory = case_dir / f"step_{step_ms}ms"
            directory.mkdir(parents=True, exist_ok=True)
            parameters = {
                "tracker": tracker,
                "case": case,
                "step_ms": step_ms,
                "xy": xy.cpu().tolist(),
                "labels": labels,
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
            if video is None:
                print(
                    f"[tracker-review] decode={case_id} path={case['record']['path']}",
                    flush=True,
                )
                video = decode_case(case, all_indices)
                write_video(
                    case_dir / "source.mp4",
                    (rgb_image(frame) for frame in video),
                    case["record"]["fps"],
                )
            native_rgb = video[native_indices - all_indices[0]]
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
                    f"[tracker-review] inference={case_id} step_ms={step_ms} native_frames={len(native_rgb)} sampled_frames=8 points={len(xy)}",
                    flush=True,
                )
                started = time.monotonic()
                native = predict(
                    model, native_rgb, native_indices, case["anchor_frame"], xy, device
                )
                sampled = predict(
                    model,
                    sampled_rgb,
                    sampled_indices,
                    case["anchor_frame"],
                    xy,
                    device,
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
