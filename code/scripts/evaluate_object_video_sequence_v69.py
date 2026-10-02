#!/usr/bin/env python3
"""Held sequence evaluation, interventions, independent truth, and SwanLab evidence."""

import argparse
from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69
from igsw.adaptive_gaussian_wm.v69_runtime import add_v69_arguments, append_case_records
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.object_video_world_model_v69 import ObjectVideoWorldModelV69
from igsw.adaptive_gaussian_wm.object_sequence_evaluation_v69 import sequence_metrics_v69, independent_binding_v69, history_forecast_ablation_v69
from igsw.adaptive_gaussian_wm.object_sequence_annotations_v69 import annotation_template_v69, independent_measurements_v69
from igsw.adaptive_gaussian_wm.object_sequence_media_v69 import render_sequence_v69
from igsw.adaptive_gaussian_wm.object_association_diagnostics_v69 import independent_association_v69, effect_rate_and_sampling_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json
from igsw.adaptive_gaussian_wm.swanlab_tracking_v69 import (
    start_swanlab_v69, log_values_v69, log_table_v69, log_video_v69, log_evidence_v69,
)


def log_diagnostic_table(run, name, columns, records):
    log_table_v69(run, name, columns, records)


def main():
    p = add_v69_arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--items", type=int, default=400)
    p.add_argument("--visualize", type=int, default=40)
    p.add_argument("--annotations", default="")
    p.add_argument("--effect_samples", type=int, default=4)
    args = p.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ObjectVideoConfigV69(**state["config"])
    stage = state["args"]["stage"]
    dataset = ObjectVideoSequenceDatasetV69(args.manifest, config, args.seed, "held", args.annotations)
    perception = PretrainedVisualEncoderV69(config.encoder, args.encoder_repository, args.encoder_weights,
                     args.encoder_frame_batch, history_seconds=config.history_seconds, saved_backbone=state["perception"]).to(device)
    model = ObjectVideoWorldModelV69(config, stage).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    step = state["step"]
    del state
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    order = order[:args.items] if args.items else order
    rows, ablations, failures, independent, templates, effect_diagnostics = [], [], [], [], [], []
    independent_skips = []
    videos, counts = [], Counter()
    begin = time.monotonic()
    with torch.no_grad():
        for number, index in enumerate(order):
            sample = dataset[index]
            if "decode_error" in sample:
                failures.append(sample)
                print(json.dumps({"event": "decode_skip", **sample}), flush=True)
                continue
            templates.append(annotation_template_v69(sample))
            batch = move_batch_v69(collate_object_video_v69([sample]), device)
            fields = perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(fields, batch, deterministic_effect=True)
                case_rows = sequence_metrics_v69(output, batch, config, stage)
                if stage == "dynamics":
                    ablation = history_forecast_ablation_v69(model, fields, output, batch, perception)
                    ablations.extend(ablation["rows"])
                    effect_diagnostics.append(effect_rate_and_sampling_v69(model, output, batch, args.effect_samples))
                annotation = sample["annotation"]
                historical_frames = set(sample["frame_indices"][:config.history_frames].tolist())
                if (annotation is not None and annotation["queries"] and annotation["tracks"]
                        and annotation["provenance"] in ("human_annotation", "simulator_ground_truth")
                        and annotation["uses_training_tracker"] is False
                        and any(q["frame_index"] in historical_frames for q in annotation["queries"])):
                    truth_batch, queries, labels = independent_measurements_v69(batch, sample["annotation"])
                    truth_batch["independent_truth"] = True
                    truth_output = model(fields, truth_batch, queries, deterministic_effect=True)
                    truth_rows = sequence_metrics_v69(truth_output, truth_batch, config, stage)
                    independent.append({"case": sample["case_id"], "metrics": independent_binding_v69(model, truth_output, truth_batch, labels),
                                        "trajectory_rows": truth_rows,
                                        "association": independent_association_v69(model, truth_output, truth_batch, labels, args.effect_samples)})
                elif annotation is not None:
                    independent_skips.append({"case": sample["case_id"], "reason": "requires independent labels, measurements and an observed-history query"})
            rows.extend(case_rows)
            append_case_records(out / "cases.jsonl", case_rows)
            counts[sample["source"]] += 1
            if len(videos) < args.visualize:
                path = render_sequence_v69(batch, output, out / "videos" / sample["case_id"], config, stage)
                videos.append({"case": sample["case_id"], "path": str(path)})
            print(f"[object-video-v69-eval] {number+1}/{len(order)} case={sample['case_id']} rows={len(case_rows)}", flush=True)
    aggregates, macro = defaultdict(list), defaultdict(list)
    for row in rows:
        key = (row["source"], row["condition"], row["subset"], round(row["seconds"], 1))
        aggregates[key].extend(row["epe_px"])
        if row["mean"] is not None:
            macro[key].append(row["mean"])
    from igsw.adaptive_gaussian_wm.object_sequence_evaluation_v69 import distribution_v69
    summary = [{"source": source, "condition": condition, "subset": subset, "seconds": seconds,
                "aggregation": "point_weighted; macro_case_mean separately reported",
                "macro_case_mean": distribution_v69(torch.tensor(macro[(source, condition, subset, seconds)])),
                **distribution_v69(torch.tensor(values))} for (source, condition, subset, seconds), values in sorted(aggregates.items())]
    report = {"status": "completed", "checkpoint": args.checkpoint, "checkpoint_step": step, "stage": stage,
              "cases_by_source": dict(counts), "failed_decode": failures, "summary": summary, "history_ablations": ablations,
              "independent": independent, "independent_status": "measured" if independent else "not_measured_no_annotations",
              "association_status": "measured" if any(c["association"]["status"] == "measured_independent_annotations" for c in independent)
                                    else "not_measured_no_valid_independent_entity_targets",
              "effect_diagnostics": effect_diagnostics,
              "independent_skips": independent_skips,
              "elapsed_seconds": time.monotonic()-begin, "perception": perception.provenance,
              "future_condition": "actual-future posterior; no deployment effect selector" if stage == "dynamics" else "observed-video reconstruction",
              "held_tracker_results_are_not_object_ground_truth": True, "videos": videos}
    write_json(out / "report.json", report)
    with (out / "annotation_template.jsonl").open("w", encoding="utf-8") as stream:
        for row in templates:
            stream.write(json.dumps(row)+"\n")
    links = ["<!doctype html><meta charset='utf-8'><h1>V69 held sequence evaluation</h1>",
             "<p>Yellow: measured teacher location. Red: model. Future images are targets, not prediction inputs.</p>"]
    for video in videos:
        relative = Path(video["path"]).relative_to(out).as_posix()
        links.append(f"<h2>{html.escape(video['case'])}</h2><video controls style='max-width:100%' src='{html.escape(relative)}'></video>")
    (out / "index.html").write_text("\n".join(links), encoding="utf-8")
    run = start_swanlab_v69(args, "object-video-sequence-v69", "held-evaluation", config={**vars(args), **config.to_dict()})
    if run:
        log_table_v69(run, "evaluation/cases",
                      ["case", "source", "condition", "subset", "seconds", "selection_status", "count", "mean", "p50", "p90", "p95"], rows)
        log_table_v69(run, "evaluation/summary",
                      ["source", "condition", "subset", "seconds", "count", "mean", "p50", "p90", "p95", "macro_case_mean"], summary)
        log_values_v69(run, {"evaluation/independent_cases": len(independent), "evaluation/independent_status": report["independent_status"],
                            "evaluation/source_counts": dict(counts), "evaluation/checkpoint_step": step})
        query_records, consistency_records, intervention_records, grouping_records, query_sample_records = [], [], [], [], []
        for case in independent:
            for row in case["association"]["query_reconstruction"]:
                query_records.append({"case": case["case"], "region": row["measured_region"], **row})
            for row in case["association"]["same_entity_common_location_consistency"]:
                consistency_records.append({"case": case["case"], **row})
            for row in case["association"]["query_effect_interventions"]:
                intervention_records.append({"case": case["case"], **row,
                                             "response_mean_px": row["response_px"]["mean"],
                                             "response_p90_px": row["response_px"]["p90"],
                                             "reconstruction_mean_px": row["reconstruction_px"]["mean"]})
            for row in case["metrics"]["independent_grouping_controls"]:
                grouping_records.append({"case": case["case"], **row, **row["brier_error"]})
            for row in case["association"]["sampling"].get("independent_query_sampling", []):
                query_sample_records.append({"case": case["case"], **row, **row["spread_distribution"]})
        log_diagnostic_table(run, "association/query_reconstruction", ["case", "query", "object", "region", "count", "mean", "p50", "p90", "kl_nats"], query_records)
        log_diagnostic_table(run, "association/common_location_consistency", ["case", "object", "query_a", "query_b", "count", "mean", "p50", "p90"], consistency_records)
        log_diagnostic_table(run, "association/effect_interventions", ["case", "intervention", "recipient_query", "donor_query", "observed_object", "response_mean_px", "response_p90_px", "reconstruction_mean_px"], intervention_records)
        log_diagnostic_table(run, "association/entity_grouping", ["case", "condition", "relation", "count", "mean", "p50", "p90"], grouping_records)
        log_diagnostic_table(run, "effect/independent_query_sampling", ["case", "query", "object", "count", "mean", "p50", "p90"], query_sample_records)
        log_values_v69(run, {"association/status": report["association_status"]})
        rate_records, sample_records = [], []
        for case in effect_diagnostics:
            for row in case.get("rates", []):
                for query, rate in zip(row["query_ids"], row["kl_nats_by_query"]):
                    rate_records.append({"case": row["case"], "query": query, "kl_nats": rate,
                                         "clip_total_kl_nats": row["clip_total_kl_nats"], "valid_queries": row["valid_queries"]})
            sample_records.extend(case.get("sample_reconstruction", []))
        log_diagnostic_table(run, "effect/per_query_rate", ["case", "query", "kl_nats", "clip_total_kl_nats", "valid_queries"], rate_records)
        log_diagnostic_table(run, "effect/sample_reconstruction", ["case", "sample", "count", "mean", "p50", "p90"], sample_records)
        for video in videos:
            log_video_v69(run, f"video/{video['case']}", video["path"])
        log_evidence_v69(run, [out], out)
        run.finish()
    print(f"[object-video-v69-eval] report={out / 'report.json'} review={out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
