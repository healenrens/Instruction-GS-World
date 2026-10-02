#!/usr/bin/env python3
"""Held State change reconstruction, compact-state interventions and W&B evidence."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist

from igsw.distributed import init_torchrun
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69
from igsw.adaptive_gaussian_wm.object_video_world_model_v69 import ObjectVideoWorldModelV69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69
from igsw.adaptive_gaussian_wm.object_sequence_annotations_v69 import annotation_template_v69, independent_measurements_v69
from igsw.adaptive_gaussian_wm.object_sequence_evaluation_v69 import independent_binding_v69
from igsw.adaptive_gaussian_wm.state_change_evaluation_v69 import (
    CONDITIONS, METRICS, select_held_episodes_v69, load_state_modules_v69, observe_state_v69,
    reconstruct_interventions_v69, summarize_state_case_v69, aggregate_case_rows_v69, aggregate_paired_rows_v69,
)
from igsw.adaptive_gaussian_wm.state_change_media_v69 import render_state_change_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json
from igsw.adaptive_gaussian_wm.swanlab_tracking_v69 import (
    add_swanlab_arguments, start_swanlab_v69, log_values_v69, log_table_v69, log_video_v69, log_evidence_v69,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="", help="Defaults to the checkpoint run's frozen dataset.json.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder_repository", required=True)
    parser.add_argument("--encoder_frame_batch", type=int, default=2)
    parser.add_argument("--items_per_source", type=int, default=80)
    parser.add_argument("--visualize_per_source", type=int, default=8)
    parser.add_argument("--motion_threshold_px", type=float, default=5.0)
    parser.add_argument("--annotations", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source_revision", default="local-unversioned")
    add_swanlab_arguments(parser, default_name="v69_state_change_held")
    return parser.parse_args()


def append_json(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def read_rank_records(out, world):
    return [json.loads(line) for rank in range(world)
            for line in (out / f"cases_rank{rank:04d}.jsonl").read_text().splitlines()]


def log_rows(run, name, rows, columns):
    log_table_v69(run, name, columns, rows)


def start_tracker(args, config, checkpoint_metadata):
    return start_swanlab_v69(args, "object-video-v69-state-change", "held-state-evaluation",
                            config={**vars(args), **checkpoint_metadata, "config": config.to_dict()})


def publish_results(run, args, report, cases, out):
    rows = [row for case in cases for row in case.get("rows", [])]
    paired = [row for case in cases for row in case.get("paired", [])]
    columns = ["case", "source", "condition", "pool", "scope", "slice"]
    columns += [metric + "_" + statistic for metric in METRICS for statistic in ("count", "mean", "p50", "p90", "p95")]
    log_rows(run, "state_change/cases", rows, columns)
    log_rows(run, "state_change/paired_cases", paired, ["case", "source", "condition", "pool"] +
             [metric + "_penalty_" + statistic for metric in ("position_error_px", "change_error_px")
              for statistic in ("count", "mean", "p50", "p90", "p95")])
    summary_rows = []
    for row in report["summary"]:
        for metric in METRICS:
            for statistic in ("mean", "p50", "p90"):
                summary_rows.append({**{key: row[key] for key in ("source", "condition", "pool", "scope", "slice")},
                                     "metric": metric, "clip_statistic": statistic, **row[metric + "_" + statistic]})
    log_rows(run, "state_change/summary", summary_rows,
             ["source", "condition", "pool", "scope", "slice", "metric", "clip_statistic", "count", "mean", "p50", "p90", "p95"])
    pair_summary = []
    for row in report["paired_summary"]:
        for metric in ("position_error_px", "change_error_px"):
            pair_summary.append({"source": row["source"], "condition": row["condition"], "pool": row["pool"], "metric": metric,
                                 "observed_state_better_fraction": row[metric + "_observed_state_better_fraction"],
                                 **row[metric + "_penalty_mean"]})
    log_rows(run, "state_change/paired_summary", pair_summary,
             ["source", "condition", "pool", "metric", "count", "mean", "p50", "p90", "p95", "observed_state_better_fraction"])
    log_rows(run, "state_change/case_inventory", cases,
             ["case", "source", "episode_index", "status", "native_hw", "valid_query_count", "query_ownership_entropy_median", "unbound_mass_median", "independent_status"])
    independent_rows = [row for case in cases for row in case.get("independent_rows", [])]
    log_rows(run, "state_change/independent_cases", independent_rows, columns)
    log_rows(run, "state_change/independent_case_evidence", [
        {"case": case["case"], "point_identity": case["independent_grouping"]["grouping_labeled_objects"],
         "query_mass": case["independent_grouping"]["independent_query_mass"],
         "unrelated_mass": case["independent_grouping"]["independent_unrelated_mass"],
         "deletion": case["independent_grouping"]["independent_deletion"]}
        for case in cases if "independent_grouping" in case
    ], ["case", "point_identity", "query_mass", "unrelated_mass", "deletion"])
    log_rows(run, "state_change/independent_grouping", [
        {"case": case["case"], "condition": row["condition"], "relation": row["relation"], **row["brier_error"]}
        for case in cases for row in case.get("independent_grouping", {}).get("independent_grouping_controls", [])
    ], ["case", "condition", "relation", "count", "mean", "p50", "p90", "p95"])
    scalar = {"state_change/checkpoint_step": report["checkpoint_step"], "state_change/completed_clips": report["completed_clips"],
              "state_change/failed_decode_clips": report["failed_decode_clips"], "state_change/independent_status": report["independent_status"]}
    for row in report["summary"]:
        if row["source"] == "__all_sources__" and row["pool"] == "all_points" and row["scope"] == "motion" and row["slice"] == "motion_active":
            for metric in ("position_error_px", "change_error_px", "normalized_change_error"):
                stats = row[metric + "_mean"]
                if stats["p50"] is not None:
                    scalar[f"state_change/{row['condition']}/{metric}_clip_p50"] = stats["p50"]
                    scalar[f"state_change/{row['condition']}/{metric}_clip_p90"] = stats["p90"]
    log_values_v69(run, scalar)
    for case in cases:
        for kind, path in case.get("videos", {}).items():
            log_video_v69(run, f"review/{case['case']}/{kind}", path)
    paths = [out / name for name in ("report.json", "metric_definitions.json", "index.html", "annotation_template.jsonl")]
    paths += [out / name for name in ("case_reports", "evidence") if (out / name).is_dir()]
    log_evidence_v69(run, paths, out)
    run.finish()


def main():
    args = arguments()
    context = init_torchrun(backend="gloo")
    device = torch.device(f"cuda:{context.local_rank}")
    torch.cuda.set_device(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "checkpoint_snapshot.pt"
    if context.is_main:
        shutil.copyfile(args.checkpoint, snapshot)
    if context.distributed:
        dist.barrier()
    saved = torch.load(snapshot, map_location="cpu", mmap=True, weights_only=False)
    args.manifest = args.manifest or str(Path(saved["args"]["out"]) / "dataset.json")
    config = ObjectVideoConfigV69(**saved["config"])
    checkpoint_metadata = {"checkpoint_step": saved["step"], "checkpoint_revision": saved["args"]["source_revision"],
                           "checkpoint_training_stage": saved["args"]["stage"]}
    torch.manual_seed(args.seed)
    perception = PretrainedVisualEncoderV69(config.encoder, args.encoder_repository, saved["args"]["encoder_weights"],
                     args.encoder_frame_batch, history_seconds=config.history_seconds, saved_backbone=saved["perception"]).to(device)
    model = ObjectVideoWorldModelV69(config, "state").to(device).eval()
    loaded_modules = load_state_modules_v69(model, saved["model"])
    del saved
    dataset = ObjectVideoSequenceDatasetV69(args.manifest, config, args.seed, "held", args.annotations)
    indices, coverage = select_held_episodes_v69(dataset, args.items_per_source, args.seed)
    visual_count, visualize = Counter(), set()
    for index in indices:
        source = dataset.entries[index]["source"]
        if visual_count[source] < args.visualize_per_source:
            visualize.add(index)
            visual_count[source] += 1
    records = out / f"cases_rank{context.rank:04d}.jsonl"
    records.write_text("", encoding="utf-8")
    (out / "case_reports").mkdir(exist_ok=True)
    (out / "evidence").mkdir(exist_ok=True)
    begin = time.monotonic()
    rank_completed = 0
    run = start_tracker(args, config, checkpoint_metadata) if context.is_main else None
    print(json.dumps({"event": "state_change_eval_start", **checkpoint_metadata, "loaded_modules": loaded_modules,
                      "rank": context.rank, "world": context.world_size, "selected_episodes": len(indices),
                      "cases_by_source": coverage, "out": str(out)}), flush=True)
    with torch.no_grad():
        for number in range(context.rank, len(indices), context.world_size):
            index = indices[number]
            sample = dataset[index]
            if "decode_error" in sample:
                record = {"case": sample["case_id"], "source": sample["source"], "status": "decode_failed", "error": sample["decode_error"]}
                append_json(records, record)
                print(json.dumps(record), flush=True)
                continue
            batch = move_batch_v69(collate_object_video_v69([sample]), device)
            fields = perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = observe_state_v69(model, fields, batch)
                predictions = reconstruct_interventions_v69(model, output, batch, args.seed + index)
            rows, paired = summarize_state_case_v69(predictions, batch, config, args.motion_threshold_px)
            owners = output["reference_ownership"][0].float()
            present = batch["teacher"]["point_present"][0]
            entropy = -(owners * owners.clamp_min(1e-8).log()).sum(-1)
            record = {"case": sample["case_id"], "source": sample["source"], "episode_index": sample["episode_index"],
                      "status": "completed", "native_hw": sample["native_hw"].tolist(), "rows": rows, "paired": paired,
                      "valid_query_count": int(output["queries"].valid.sum()),
                      "query_ownership_entropy_median": float(entropy[present].median()) if bool(present.any()) else None,
                      "unbound_mass_median": float(owners[present, -1].median()) if bool(present.any()) else None,
                      "independent_status": "not_measured_no_independent_labels"}
            annotation = sample["annotation"]
            if (annotation is not None and annotation["queries"] and annotation["tracks"]
                    and annotation["provenance"] in ("human_annotation", "simulator_ground_truth")
                    and annotation["uses_training_tracker"] is False
                    and any(q["frame_index"] in sample["frame_indices"][:config.history_frames].tolist() for q in annotation["queries"])):
                truth_batch, truth_queries, labels = independent_measurements_v69(batch, annotation)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    truth_output = observe_state_v69(model, fields, truth_batch, truth_queries)
                    truth_predictions = reconstruct_interventions_v69(model, truth_output, truth_batch, args.seed + index)
                    record["independent_grouping"] = independent_binding_v69(model, truth_output, truth_batch, labels)
                record["independent_rows"], record["independent_paired"] = summarize_state_case_v69(truth_predictions, truth_batch, config, args.motion_threshold_px)
                record["independent_status"] = "measured_independent_points_with_human_history_queries"
            torch.save({"case": sample["case_id"], "source": sample["source"], "frame_indices": sample["frame_indices"],
                        "native_hw": sample["native_hw"], "times": sample["times"], "teacher": {k: v.cpu() for k, v in batch["teacher"].items()},
                        "predictions": {k: v.float().cpu() for k, v in predictions.items()},
                        "queries": {key: getattr(output["queries"], key).cpu() for key in ("xy", "frame_index", "valid")},
                        "centers": torch.stack([s.centers.float().cpu() for s in output["observed_states"]], 1),
                        "annotation_template": annotation_template_v69(sample)}, out / "evidence" / f"{sample['case_id']}.pt")
            if index in visualize:
                record["videos"] = render_state_change_v69(batch, output, predictions, out / "videos" / sample["case_id"])
            write_json(out / "case_reports" / f"{sample['case_id']}.json", record)
            append_json(records, record)
            rank_completed += 1
            if run is not None:
                run.log({"progress/rank0_completed_cases": rank_completed,
                         "progress/last_case": sample["case_id"], "progress/last_source": sample["source"],
                         "progress/elapsed_seconds": time.monotonic() - begin})
            print(f"[state-change-v69] rank={context.rank} completed={rank_completed} case_index={number+1}/{len(indices)} case={sample['case_id']}", flush=True)
            del output, predictions, fields, batch
    if context.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if context.is_main:
        cases = read_rank_records(out, context.world_size)
        rows = [row for case in cases for row in case.get("rows", [])]
        paired = [row for case in cases for row in case.get("paired", [])]
        completed = [case for case in cases if case["status"] == "completed"]
        independent_rows = [row for case in cases for row in case.get("independent_rows", [])]
        report = {"status": "completed", **checkpoint_metadata, "checkpoint_snapshot": str(snapshot), "requested_checkpoint": args.checkpoint,
                  "evaluation_revision": args.source_revision, "model_config": config.to_dict(), "loaded_modules": loaded_modules,
                  "selected_episodes": coverage, "completed_clips": len(completed), "failed_decode_clips": len(cases)-len(completed),
                  "completed_by_source": dict(Counter(case["source"] for case in completed)),
                  "motion_threshold_px": args.motion_threshold_px, "summary": aggregate_case_rows_v69(rows),
                  "paired_summary": aggregate_paired_rows_v69(paired), "independent_summary": aggregate_case_rows_v69(independent_rows),
                  "independent_status": "measured" if independent_rows else "not_measured_no_independent_labels",
                  "elapsed_seconds": time.monotonic()-begin, "world_size": context.world_size,
                  "measurement_source": "tracker pseudo measurements unless independent annotations explicitly provided",
                  "state_input": "all observed RGB frames; no forecasting claim",
                  "query_ownership_entropy_is_only_diagnostic": True,
                  "metric_scope": "position reconstruction and current-to-observed displacement; compact-state intervention evidence, not proof of object semantics"}
        write_json(out / "report.json", report)
        write_definitions(out, args.motion_threshold_px)
        with (out / "annotation_template.jsonl").open("w", encoding="utf-8") as stream:
            for case in completed:
                evidence = torch.load(out / "evidence" / f"{case['case']}.pt", map_location="cpu", weights_only=False)
                stream.write(json.dumps(evidence["annotation_template"]) + "\n")
        links = ["<!doctype html><meta charset='utf-8'><h1>V69 observed State change evaluation</h1>",
                 f"<p>Checkpoint step {report['checkpoint_step']}; {report['checkpoint_revision']}</p>",
                 "<p>All video frames are observed inputs. Yellow: selected transport target; cyan: context target; red: state readout. Query colors are indices, not object labels.</p>"]
        for case in completed:
            if "videos" in case:
                links.append(f"<h2>{html.escape(case['source'])} / {html.escape(case['case'])}</h2>")
                for kind, path in case["videos"].items():
                    relative = Path(path).relative_to(out).as_posix()
                    links.append(f"<p>{kind}</p><video controls style='max-width:100%' src='{html.escape(relative)}'></video>")
        (out / "index.html").write_text("\n".join(links), encoding="utf-8")
        if run is not None:
            publish_results(run, args, report, cases, out)
        print(f"[state-change-v69] report={out / 'report.json'} review={out / 'index.html'}", flush=True)


def write_definitions(out, threshold):
    write_json(out / "metric_definitions.json", {
        "position_error_px": "Euclidean pixel distance between readout position and measured position, after the history window.",
        "change_error_px": "Euclidean pixel error of readout displacement from t=0; requires a reliable measurement at t=0 and target frame.",
        "normalized_change_error": "Change error / measured displacement, only for displacement >= motion_threshold_px. Zero displacement prediction scores 1.",
        "direction_cosine_error": "1-cos(predicted displacement, measured displacement), only motion-active. Zero predicted displacement scores 1.",
        "motion_threshold_px": threshold, "threshold_role": "reported pixel slice, not a calibrated object label or success threshold",
        "motion_bins": ["under 1", "1 to 5", "5 to 20", "20 to 50", "at least 50 native pixels"],
        "conditions": {"observed_state": "True observed states.", "frozen_state": "Freeze tokens and centers after t=0.",
                       "shuffled_state": "Permute continuation states without fixed frames; preserve recipient timestamp and calibration history.",
                       "frozen_tokens": "Freeze tokens at t=0, retain observed centers.",
                       "frozen_centers": "Freeze centers at t=0, retain observed tokens.", "reference_copy": "Repeat the supplied per-point history reference location."},
        "paired_penalty_px": "Control error minus observed-state error, at the exact same motion-active measurements.",
        "summary": "First compute per-case point/frame statistics, then report distributions across cases, separately by source.",
        "raw_evidence": "evidence/*.pt contains point IDs, targets, masks, all predictions and query centers.",
        "teacher_limits": "Tracker motion/visibility and global top75 selection are pseudo evidence; raw displacement may include camera motion.",
        "semantic_status": "Independent object grouping requires human/simulator annotations. Query centers and entropy are not object validity scores.",
    })


if __name__ == "__main__":
    main()
