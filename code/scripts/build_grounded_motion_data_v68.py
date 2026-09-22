#!/usr/bin/env python3
"""Resumable offline generation with independent workers sharing visible GPUs."""

from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch

from review_grounded_object_tracker_v67 import parse_args
from igsw.adaptive_gaussian_wm.grounded_background_motion_v68 import reference_queries, fit_background, motion_evidence, relay_evidence
from igsw.adaptive_gaussian_wm.grounded_motion_sources_v68 import select_data_cases
from igsw.adaptive_gaussian_wm.grounded_motion_jobs_v68 import compatible_configuration, reserve_arguments, split_reserve, append_replacement, target_counts, interleave_sources
from igsw.adaptive_gaussian_wm.grounded_motion_recovery_v68 import recover_plan, recover_tracks, reprocess_only
from igsw.adaptive_gaussian_wm.grounded_motion_refinement_v68 import refine_and_densify
from igsw.adaptive_gaussian_wm.grounded_motion_export_v68 import CONTRACT, SELECTION_POLICY, export_motion_teacher
from igsw.adaptive_gaussian_wm.grounded_motion_review_v68 import render_motion_data, write_data_gallery, write_review_bundle, upload_data, camera_overview
from igsw.adaptive_gaussian_wm.grounded_tracker_masks_v67 import GroundedTrackerMasks
from igsw.adaptive_gaussian_wm.grounded_tracker_roles_v67 import resolve_track_roles
from igsw.adaptive_gaussian_wm.grounded_tracker_sampling_v67 import build_grounded_queries, save_tensor
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json, decode_case, load_tracker, predict
from igsw.adaptive_gaussian_wm.video_file_decoder import VideoDecodeError


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
    parser.add_argument("--review_cases_per_source", type=int, default=80,
                        help="Global visualization quota per source; 0 renders all selected clips.")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--replacement_cases_per_source", type=int, default=0,
                        help="Reserve episodes per source; 0 selects max(32, 25 percent of quota).")
    parser.add_argument("--reuse_source_revision", default="",
                        help="Explicitly reuse unchanged targets from this earlier code revision.")
    parser.add_argument("--recover_from", default="",
                        help="Read completed tracks and original case plans from an older, untouched build.")
    parser.add_argument("--reprocess_only_sources", default="",
                        help="Comma-separated sources whose saved tracks may be reselected but never regenerated.")


def plan_reviews(cases, enabled, quota):
    counts = Counter()
    for case in cases:
        source = case["source"]
        case["render"] = bool(enabled and (quota == 0 or counts[source] < quota))
        counts[source] += int(case["render"])
    return dict(counts)


def main():
    args = parse_args(configure)
    cv2.setNumThreads(1)
    rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    device_index = local // args.workers_per_gpu
    out = Path(args.out) / f"shard_{rank:04d}"
    out.mkdir(parents=True, exist_ok=True)
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    args.wandb_name += f"_shard{rank:04d}"
    configuration = {k: v for k, v in vars(args).items() if not k.startswith("wandb_") and k not in ("stage", "reuse_completed")}
    configuration["worker_count"] = world
    configuration["selection_policy"] = SELECTION_POLICY
    if rank == 0:
        write_json(Path(args.out) / "workers.json", {"count": world, "configuration": configuration})
    if args.stage == "upload":
        manifest = read_json(out / "training_manifest.json")
        upload_data(args, out, manifest["entries"], manifest["configuration"])
        return
    case_plan = out / "cases.json"
    saved_plan = read_json(case_plan) if case_plan.is_file() else None
    if args.reuse_completed and saved_plan and "reserve" in saved_plan and compatible_configuration(saved_plan["configuration"], configuration):
        cases, reserve = saved_plan["cases"], saved_plan["reserve"]
        selection = read_json(out / "selection.json")
    elif args.recover_from:
        cases, reserve, selection = recover_plan(args, rank, world)
    else:
        candidates, selection = select_data_cases(reserve_arguments(args))
        cases, reserve = split_reserve(candidates, 0 if args.case_manifest else args.cases_per_source)
        selection["selected_by_source"] = target_counts(cases)
        selection["reserve_by_source"] = target_counts(reserve)
        selection["requested_episodes_per_source"] = args.cases_per_source
        # Choose the review subset globally, independent of worker count or tracker success.
        selection["review_by_source"] = plan_reviews(cases, args.render, args.review_cases_per_source)
        selection["review_selection"] = "first source quota in deterministic group-round-robin case plan"
        cases = cases[rank::world]
        reserve = reserve[rank::world]
        selection.update(shard_rank=rank, shard_count=world, selected_in_shard=len(cases))
    cases = interleave_sources(cases)
    reserve = [case for case in reserve if not reprocess_only(args, case)]
    selection["reprocess_only_sources"] = args.reprocess_only_sources
    selection["execution_order"] = "per-worker source round-robin; original quotas and review IDs unchanged"
    write_json(out / "selection.json", selection)
    print(f"[motion-data-v68] planned_global_by_source={selection['selected_by_source']} review_global_by_source={selection['review_by_source']} "
          f"shard={rank}/{world} cuda={device_index} workers_per_gpu={args.workers_per_gpu} local_clips={len(cases)}", flush=True)
    def save_plan():
        write_json(case_plan, {"cases": cases, "reserve": reserve, "configuration": configuration})
    save_plan()
    if args.operation == "cameras":
        links = ["<!doctype html><meta charset='utf-8'><h1>Camera catalog</h1>"]
        for case in cases:
            directory = out / case["case_id"]
            directory.mkdir(exist_ok=True)
            camera_overview(case, directory)
            links.append(f"<h2>{case['case_id']}</h2><img style='max-width:100%' src='{case['case_id']}/camera_overview.png'>")
        (out / "index.html").write_text("\n".join(links))
        return
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    segmenter = GroundedTrackerMasks(args, device)
    model, tracker = load_tracker(args, device)
    entries = []
    review_entries = []
    targets = target_counts(cases)
    planned_clips = sum(targets.values())
    planned_reviews = sum(c["render"] for c in cases if "replacement_for" not in c)
    failure_path = out / "decode_failures.json"
    saved_failures = read_json(failure_path) if failure_path.is_file() else None
    failures = (saved_failures["cases"] if args.reuse_completed and saved_failures
                and compatible_configuration(saved_failures["configuration"], configuration) else {})
    bad_paths = {row["path"]: row["error"] for row in failures.values() if row["path_unusable"]}
    replaced = {c["replacement_for"] for c in cases if "replacement_for" in c}
    reprocess_missing = {}

    def save_progress(status, update_gallery=False):
        write_json(out / "progress.json", {"status": status, "completed_clips": len(entries), "planned_clips": planned_clips,
                   "completed_by_source": dict(Counter(e["source"] for e in entries)), "planned_by_source": targets,
                   "recovered_clips": sum("parent_teacher" in e for e in entries),
                   "reprocess_only_missing_clips": len(reprocess_missing),
                   "decode_skipped_clips": len(failures), "attempt_queue_length": len(cases),
                   "decode_skipped_by_source": dict(Counter(e["source"] for e in failures.values())),
                   "rendered_clips": len(review_entries), "planned_review_clips": planned_reviews,
                   "shard_rank": rank, "cuda_device": device_index, "workers_per_gpu": args.workers_per_gpu,
                   "last_case": entries[-1]["case_id"] if entries else None})
        # Full production manifests are written once, not rewritten after each clip.
        if status != "building" or not entries:
            write_json(out / "training_manifest.json", {"contract": CONTRACT, "root": str(out.resolve()), "entries": entries,
                       "configuration": configuration, "tracker": tracker, "status": status,
                       "partition": args.partition, "shard_rank": rank, "shard_count": world,
                       "planned_clips": planned_clips, "decode_skipped_clips": len(failures),
                       "reprocess_only_missing": list(reprocess_missing.values()),
                       "teacher_only": True, "future_used_for_selection": True, "source_revision": args.source_revision})
        if args.render and (update_gallery or status != "building" or not entries):
            write_data_gallery(out, review_entries, selection)

    save_progress("building")
    for ordinal, case in enumerate(cases):
        directory = out / case["case_id"]
        directory.mkdir(exist_ok=True)
        completed = directory / "complete.json"
        if args.reuse_completed and completed.is_file() and compatible_configuration(read_json(completed)["configuration"], configuration):
            entries.append(read_json(completed)["entry"])
            write_json(completed, {"configuration": configuration, "entry": entries[-1]})
            if entries[-1]["rendered"]:
                review_entries.append(entries[-1])
            print(f"[motion-data-v68] shard={rank} reuse={case['case_id']}", flush=True)
            save_progress("building", update_gallery=entries[-1]["rendered"])
            continue
        print(f"[motion-data-v68] shard={rank} case={ordinal+1}/{len(cases)} id={case['case_id']} camera={case['camera']}", flush=True)
        indices = torch.arange(case["first_frame"], case["last_frame"] + 1)
        path = case["record"]["path"]
        inherited = recover_tracks(args, rank, case, directory)
        if reprocess_only(args, case) and inherited is None:
            reprocess_missing[case["case_id"]] = {"case_id": case["case_id"], "source": case["source"],
                                                   "reason": "no_saved_tracks; source_not_authorized_for_generation"}
            write_json(out / "reprocess_only_missing.json", reprocess_missing)
            print(f"[motion-data-v68] reprocess_only_skip={case['case_id']} reason=no_saved_tracks", flush=True)
            save_progress("building")
            continue
        if case["case_id"] in failures:
            row = failures[case["case_id"]]
            rgb = VideoDecodeError(row["error"], path_unusable=row["path_unusable"])
        elif path in bad_paths:
            rgb = VideoDecodeError(bad_paths[path], path_unusable=True)
        elif inherited is not None and not case["render"]:
            rgb = None
        else:
            rgb = decode_case(case, indices, return_error=True)
        if isinstance(rgb, VideoDecodeError):
            replacement_id = None
            if case["case_id"] not in replaced and not reprocess_only(args, case):
                replacement_id = append_replacement(case, cases, reserve)
                if replacement_id:
                    replaced.add(case["case_id"])
                save_plan()
            failures[case["case_id"]] = {"case_id": case["case_id"], "source": case["source"], "path": path,
                "first_frame": case["first_frame"], "last_frame": case["last_frame"], "error": str(rgb),
                "path_unusable": rgb.path_unusable, "review_requested": case["render"]}
            if rgb.path_unusable:
                bad_paths[path] = str(rgb)
            write_json(failure_path, {"configuration": configuration, "cases": failures})
            print(f"[motion-data-v68] decode_skip={case['case_id']} path={path} replacement={replacement_id} error={rgb}", flush=True)
            save_progress("building")
            continue
        if inherited is not None:
            case["height"], case["width"] = inherited["case"]["height"], inherited["case"]["width"]
            background, native, queries = inherited["background_motion"], inherited["native"], inherited["queries"]
            sampling, roles, relay = inherited["sampling"], inherited["role_evidence"], inherited["relay_evidence"]
            print(f"[motion-data-v68] recover_raw_tracks={case['case_id']} parent={inherited['parent_teacher']}", flush=True)
        else:
            case["height"], case["width"] = rgb.shape[-2:]
            print(f"[motion-data-v68] generate_tracks={case['case_id']} source={case['source']}", flush=True)
            background, native, queries, sampling, roles, relay = track_case(
                args, configuration, directory, case, rgb, indices, model, segmenter, device)
        evidence = motion_evidence(native, background, args)
        export_args = deepcopy(args)
        export_args.tracks_source_revision = inherited["source_revision"] if inherited is not None else args.source_revision
        export_args.raw_queries_reused = inherited is not None
        report = export_motion_teacher(directory, case, queries, native, evidence, relay, background, sampling, roles, export_args)
        if case["render"]:
            render_motion_data(directory, rgb, native, queries, background, report, case, args, evidence["valid"] & relay["valid"])
        entry = {"case_id": case["case_id"], "source": case["source"], "camera": case["camera"],
                 "path": str((directory / "teacher.pt").relative_to(out)), "object_targets": report["object_motion_target_count"],
                 "motion_targets": report["selected_point_count"], "selection_policy": SELECTION_POLICY,
                 "tracks_source_revision": export_args.tracks_source_revision,
                 "background_valid_fraction": report["background_usable_frame_fraction"], "rendered": case["render"],
                 "partition": args.partition, "raw_video": case["record"]["path"]}
        if inherited is not None:
            entry["parent_teacher"] = inherited["parent_teacher"]
        if "replacement_for" in case:
            entry["replacement_for"] = case["replacement_for"]
        write_json(completed, {"configuration": configuration, "entry": entry})
        entries.append(entry)
        if entry["rendered"]:
            review_entries.append(entry)
        save_progress("building", update_gallery=entry["rendered"])
        print(f"[motion-data-v68] saved={directory / 'teacher.pt'} targets={entry['object_targets']}", flush=True)
        del rgb, inherited
    status = "completed" if dict(Counter(e["source"] for e in entries)) == targets else "incomplete"
    save_progress(status)
    if args.render:
        write_review_bundle(out, review_entries, [out / name for name in ("index.html", "selection.json", "training_manifest.json", "decode_failures.json")])
    upload_data(args, out, entries, configuration)
    print(f"[motion-data-v68] status={status} shard={rank} clips={len(entries)}/{planned_clips} "
          f"decode_skips={len(failures)} manifest={out / 'training_manifest.json'}", flush=True)


def track_case(args, configuration, directory, case, rgb, indices, model, segmenter, device):
    ref_path = directory / "background.pt"
    if args.reuse_completed and ref_path.is_file() and compatible_configuration(torch.load(ref_path, weights_only=False)["configuration"], configuration):
        background = torch.load(ref_path, weights_only=False)["value"]
    else:
        references = predict(model, rgb, indices, reference_queries(case, args.background_grid_side), device, args.points_per_pass)
        background = fit_background(references, case, args)
        save_tensor(ref_path, {"configuration": configuration, "value": background})
    pilot_args = deepcopy(args)
    pilot_args.point_budget = args.pilot_point_budget
    queries, sampling = build_grounded_queries(rgb, case, pilot_args, segmenter, directory, configuration,
                                              render=case["render"], role_agnostic=True)
    if not len(queries["xy"]):
        queries = reference_queries(case, args.background_grid_side)
        queries["metadata"] = [{"point_id": i, "xy": xy, "frame": case["first_frame"],
                                "region_id": "background_reference", "role": "scene_context",
                                "region_area_px": case["height"] * case["width"],
                                "region_diagonal_px": float((case["height"]**2 + case["width"]**2)**.5),
                                "sam_score": 0.0, "robot_overlap": 0.0} for i, xy in enumerate(queries["xy"].tolist())]
    pilot_path = directory / "pilot_tracks.pt"
    if args.reuse_completed and pilot_path.is_file() and compatible_configuration(torch.load(pilot_path, weights_only=False)["configuration"], configuration):
        pilot = torch.load(pilot_path, weights_only=False)["value"]
    else:
        pilot = predict(model, rgb, indices, queries, device, args.points_per_pass)
        save_tensor(pilot_path, {"configuration": configuration, "value": pilot})
    queries, _, _ = resolve_track_roles(pilot, queries, sampling, directory, args)
    pilot_evidence = motion_evidence(pilot, background, args)
    refinement_path = directory / "refined_query_cache.pt"
    if args.reuse_completed and refinement_path.is_file() and compatible_configuration(torch.load(refinement_path, weights_only=False)["configuration"], configuration):
        cached = torch.load(refinement_path, weights_only=False)
        queries = cached["queries"]
    else:
        queries, _ = refine_and_densify(rgb, case, queries, pilot, pilot_evidence, background, sampling, segmenter, directory, args, render=case["render"])
        save_tensor(refinement_path, {"configuration": configuration, "queries": queries})
    dense_path = directory / "dense_tracks.pt"
    if args.reuse_completed and dense_path.is_file() and compatible_configuration(torch.load(dense_path, weights_only=False)["configuration"], configuration):
        native = torch.load(dense_path, weights_only=False)["value"]
    else:
        native = predict(model, rgb, indices, queries, device, args.points_per_pass)
        save_tensor(dense_path, {"configuration": configuration, "value": native})
    queries, roles, _ = resolve_track_roles(native, queries, sampling, directory, args)
    relay_path = directory / "relay.pt"
    if args.reuse_completed and relay_path.is_file() and compatible_configuration(torch.load(relay_path, weights_only=False)["configuration"], configuration):
        relay = torch.load(relay_path, weights_only=False)["value"]
    else:
        relay = relay_evidence(model, rgb, indices, queries, native, device, args)
        save_tensor(relay_path, {"configuration": configuration, "value": relay})
    return background, native, queries, sampling, roles, relay


if __name__ == "__main__":
    main()
