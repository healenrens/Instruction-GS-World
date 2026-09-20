#!/usr/bin/env python3
"""Resumable offline generation with one independent shard per visible GPU."""

from copy import deepcopy
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from review_grounded_object_tracker_v67 import parse_args
from igsw.adaptive_gaussian_wm.grounded_background_motion_v68 import reference_queries, fit_background, motion_evidence, relay_evidence
from igsw.adaptive_gaussian_wm.grounded_motion_sources_v68 import select_data_cases
from igsw.adaptive_gaussian_wm.grounded_motion_refinement_v68 import refine_and_densify
from igsw.adaptive_gaussian_wm.grounded_motion_export_v68 import CONTRACT, export_motion_teacher
from igsw.adaptive_gaussian_wm.grounded_motion_review_v68 import render_motion_data, write_data_gallery, upload_data, camera_overview
from igsw.adaptive_gaussian_wm.grounded_tracker_masks_v67 import GroundedTrackerMasks
from igsw.adaptive_gaussian_wm.grounded_tracker_roles_v67 import resolve_track_roles
from igsw.adaptive_gaussian_wm.grounded_tracker_sampling_v67 import build_grounded_queries, save_tensor
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json, decode_case, load_tracker, predict


def configure(parser):
    parser.set_defaults(motion_top_fraction=.75, cases_per_source=80, clip_seconds=10.0)
    parser.add_argument("--operation", choices=("build", "cameras"), default="build")
    parser.add_argument("--partition", choices=("train", "held"), default="held")
    parser.add_argument("--camera_overrides", default="")
    parser.add_argument("--all_episode_windows", type=int, default=0)
    parser.add_argument("--pilot_point_budget", type=int, default=512)
    parser.add_argument("--background_grid_side", type=int, default=16)
    parser.add_argument("--background_ransac_px", type=float, default=2.0)
    parser.add_argument("--relay_max_error_px", type=float, default=3.0)
    parser.add_argument("--render", type=int, choices=(0, 1), default=1)


def main():
    args = parse_args(configure)
    rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    out = Path(args.out) / f"shard_{rank:04d}"
    out.mkdir(parents=True, exist_ok=True)
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    args.wandb_name += f"_shard{rank:04d}"
    configuration = {k: v for k, v in vars(args).items() if not k.startswith("wandb_") and k not in ("stage", "reuse_completed")}
    configuration["worker_count"] = world
    if rank == 0:
        write_json(Path(args.out) / "workers.json", {"count": world, "configuration": configuration})
    if args.stage == "upload":
        manifest = read_json(out / "training_manifest.json")
        upload_data(args, out, manifest["entries"], manifest["configuration"])
        return
    case_plan = out / "cases.json"
    if args.reuse_completed and case_plan.is_file() and read_json(case_plan)["configuration"] == configuration:
        cases = read_json(case_plan)["cases"]
        selection = read_json(out / "selection.json")
    else:
        cases, selection = select_data_cases(args)
        cases = cases[rank::world]
        selection.update(shard_rank=rank, shard_count=world, selected_in_shard=len(cases))
    write_json(out / "selection.json", selection)
    print(f"[motion-data-v68] planned_global_by_source={selection['selected_by_source']} shard={rank} local_clips={len(cases)}", flush=True)
    write_json(out / "cases.json", {"cases": cases, "configuration": configuration})
    if args.operation == "cameras":
        links = ["<!doctype html><meta charset='utf-8'><h1>Camera catalog</h1>"]
        for case in cases:
            directory = out / case["case_id"]
            directory.mkdir(exist_ok=True)
            camera_overview(case, directory)
            links.append(f"<h2>{case['case_id']}</h2><img style='max-width:100%' src='{case['case_id']}/camera_overview.png'>")
        (out / "index.html").write_text("\n".join(links))
        return
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    segmenter = GroundedTrackerMasks(args, device)
    model, tracker = load_tracker(args, device)
    entries = []

    def save_progress(status):
        write_json(out / "progress.json", {"status": status, "completed_clips": len(entries), "planned_clips": len(cases),
                   "last_case": entries[-1]["case_id"] if entries else None})
        # Full production manifests are written once, not rewritten after each clip.
        if status == "completed" or not entries:
            write_json(out / "training_manifest.json", {"contract": CONTRACT, "root": str(out.resolve()), "entries": entries,
                       "configuration": configuration, "tracker": tracker, "status": status,
                       "partition": args.partition, "shard_rank": rank, "shard_count": world,
                       "teacher_only": True, "future_used_for_selection": True, "source_revision": args.source_revision})
        if args.render:
            write_data_gallery(out, entries, selection)

    save_progress("building")
    for ordinal, case in enumerate(cases):
        directory = out / case["case_id"]
        directory.mkdir(exist_ok=True)
        completed = directory / "complete.json"
        if args.reuse_completed and completed.is_file() and read_json(completed)["configuration"] == configuration:
            entries.append(read_json(completed)["entry"])
            print(f"[motion-data-v68] shard={rank} reuse={case['case_id']}", flush=True)
            save_progress("building")
            continue
        print(f"[motion-data-v68] shard={rank} case={ordinal+1}/{len(cases)} id={case['case_id']} camera={case['camera']}", flush=True)
        indices = torch.arange(case["first_frame"], case["last_frame"] + 1)
        rgb = decode_case(case, indices)
        case["height"], case["width"] = rgb.shape[-2:]
        ref_path = directory / "background.pt"
        if args.reuse_completed and ref_path.is_file() and torch.load(ref_path, weights_only=False)["configuration"] == configuration:
            background = torch.load(ref_path, weights_only=False)["value"]
        else:
            references = predict(model, rgb, indices, reference_queries(case, args.background_grid_side), device, args.points_per_pass)
            background = fit_background(references, case, args)
            save_tensor(ref_path, {"configuration": configuration, "value": background})
        pilot_args = deepcopy(args)
        pilot_args.point_budget = args.pilot_point_budget
        queries, sampling = build_grounded_queries(rgb, case, pilot_args, segmenter, directory, configuration)
        if not len(queries["xy"]):
            queries = reference_queries(case, args.background_grid_side)
            queries["metadata"] = [{"point_id": i, "xy": xy, "frame": case["first_frame"],
                                    "region_id": "background_reference", "role": "scene_context",
                                    "region_area_px": case["height"] * case["width"],
                                    "region_diagonal_px": float((case["height"]**2 + case["width"]**2)**.5),
                                    "sam_score": 0.0, "robot_overlap": 0.0} for i, xy in enumerate(queries["xy"].tolist())]
        pilot_path = directory / "pilot_tracks.pt"
        if args.reuse_completed and pilot_path.is_file() and torch.load(pilot_path, weights_only=False)["configuration"] == configuration:
            pilot = torch.load(pilot_path, weights_only=False)["value"]
        else:
            pilot = predict(model, rgb, indices, queries, device, args.points_per_pass)
            save_tensor(pilot_path, {"configuration": configuration, "value": pilot})
        queries, _, _ = resolve_track_roles(pilot, queries, sampling, directory, args)
        pilot_evidence = motion_evidence(pilot, background, args)
        refinement_path = directory / "refined_query_cache.pt"
        if args.reuse_completed and refinement_path.is_file() and torch.load(refinement_path, weights_only=False)["configuration"] == configuration:
            cached = torch.load(refinement_path, weights_only=False)
            queries = cached["queries"]
        else:
            queries, _ = refine_and_densify(rgb, case, queries, pilot, pilot_evidence, background, sampling, segmenter, directory, args)
            save_tensor(refinement_path, {"configuration": configuration, "queries": queries})
        dense_path = directory / "dense_tracks.pt"
        if args.reuse_completed and dense_path.is_file() and torch.load(dense_path, weights_only=False)["configuration"] == configuration:
            native = torch.load(dense_path, weights_only=False)["value"]
        else:
            native = predict(model, rgb, indices, queries, device, args.points_per_pass)
            save_tensor(dense_path, {"configuration": configuration, "value": native})
        queries, roles, _ = resolve_track_roles(native, queries, sampling, directory, args)
        evidence = motion_evidence(native, background, args)
        relay_path = directory / "relay.pt"
        if args.reuse_completed and relay_path.is_file() and torch.load(relay_path, weights_only=False)["configuration"] == configuration:
            relay = torch.load(relay_path, weights_only=False)["value"]
        else:
            relay = relay_evidence(model, rgb, indices, queries, native, device, args)
            save_tensor(relay_path, {"configuration": configuration, "value": relay})
        report = export_motion_teacher(directory, case, queries, native, evidence, relay, background, sampling, roles, args)
        if args.render:
            render_motion_data(directory, rgb, native, queries, background, report, case, args, evidence["valid"] & relay["valid"])
        entry = {"case_id": case["case_id"], "source": case["source"], "camera": case["camera"],
                 "path": str((directory / "teacher.pt").relative_to(out)), "object_targets": report["object_motion_target_count"],
                 "background_valid_fraction": report["background_usable_frame_fraction"], "rendered": bool(args.render),
                 "partition": args.partition, "raw_video": case["record"]["path"]}
        write_json(completed, {"configuration": configuration, "entry": entry})
        entries.append(entry)
        save_progress("building")
        print(f"[motion-data-v68] saved={directory / 'teacher.pt'} targets={entry['object_targets']}", flush=True)
        del rgb
    save_progress("completed")
    if args.render:
        with zipfile.ZipFile(out / "review_bundle.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path.suffix in (".html", ".png", ".mp4", ".json"):
                    archive.write(path, path.relative_to(out))
    upload_data(args, out, entries, configuration)
    print(f"[motion-data-v68] completed shard={rank} manifest={out / 'training_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
