#!/usr/bin/env python3
"""Check tracker visibility and relay thresholds against independently supplied pixels."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json
from igsw.adaptive_gaussian_wm.object_video_manifest_v69 import load_object_video_manifest_v69
from igsw.adaptive_gaussian_wm.object_sequence_evaluation_v69 import distribution_v69
from igsw.adaptive_gaussian_wm.swanlab_tracking_v69 import (
    add_swanlab_arguments, start_swanlab_v69, log_table_v69, log_evidence_v69, set_results_v69,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thresholds_px", default="1,2,3,5,8")
    add_swanlab_arguments(p, default_name="object_video_v69_teacher_calibration")
    args = p.parse_args()
    manifest = load_object_video_manifest_v69(args.manifest)
    entries = {row["case_id"]: row for row in manifest["entries"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for line in Path(args.annotations).read_text().splitlines():
        annotation = json.loads(line)
        if annotation["uses_training_tracker"] or annotation["provenance"] not in ("human_annotation", "simulator_ground_truth"):
            missing.append({"case": annotation["case_id"], "reason": "annotation_is_not_independent"})
            continue
        entry = entries[annotation["case_id"]]
        data = torch.load(Path(manifest["root"]) / entry["path"], map_location="cpu", weights_only=False)
        first, second = data["native"], data["relay_evidence"]["prediction"]
        for track in annotation["tracks"]:
            if "tracker_point_id" not in track:
                missing.append({"case": annotation["case_id"], "object": track["object_id"], "reason": "no_explicit_tracker_point_id_for_calibration"})
                continue
            point = track["tracker_point_id"]
            for observation in track["observations"]:
                frame = observation["frame_index"]-data["case"]["first_frame"]
                a, b = first["tracks"][frame, point].float(), second["tracks"][frame, point].float()
                finite_a, finite_b = bool(torch.isfinite(a).all()), bool(torch.isfinite(b).all())
                visible_a = finite_a and bool(first["visibility"][frame, point] & first["in_bounds"][frame, point])
                visible_b = finite_b and bool(second["visibility"][frame, point] & second["in_bounds"][frame, point])
                has_xy = observation["visible"] is True and observation["xy_px"] is not None
                gt = torch.tensor(observation["xy_px"]) if has_xy else None
                rows.append({"case": entry["case_id"], "source": entry["source"], "object": track["object_id"],
                             "point_id": point, "frame": observation["frame_index"], "human_visible": observation["visible"],
                             "first_visible": visible_a, "relay_visible": visible_b,
                             "first_error_px": float((a-gt).norm()) if has_xy and finite_a else None,
                             "relay_error_px": float((b-gt).norm()) if has_xy and finite_b else None,
                             "agreement_error_px": float((a-b).norm()) if finite_a and finite_b else None,
                             "first_finite": finite_a, "relay_finite": finite_b,
                             "supplied_anchor": frame in (int(first["query_local_frames"][point]), int(data["relay_evidence"]["anchor_frames"][point]))})
    sweep = []
    for threshold in (float(x) for x in args.thresholds_px.split(",")):
        accepted = [r for r in rows if not r["supplied_anchor"] and r["first_visible"] and r["relay_visible"] and r["agreement_error_px"] <= threshold]
        errors = [r["first_error_px"] for r in accepted if r["first_error_px"] is not None]
        sweep.append({"relay_threshold_px": threshold, "accepted_observations": len(accepted), "gt_position_error": distribution_v69(torch.tensor(errors)),
                      "human_invisible_but_accepted": sum(r["human_visible"] is False for r in accepted),
                      "accepted_error_over_5px": sum(x > 5 for x in errors), "evaluated_position_count": len(errors)})
    known = [r for r in rows if r["human_visible"] is not None and not r["supplied_anchor"]]
    report = {"status": "calibration_measurements", "rows": rows, "missing": missing, "threshold_sweep": sweep,
              "visibility": {"known_count": len(known), "false_visible": sum(r["first_visible"] and r["human_visible"] is False for r in known),
                             "false_invisible": sum(not r["first_visible"] and r["human_visible"] is True for r in known)},
              "production_threshold_modified": False, "annotation_selection_limits_population_claims": True}
    write_json(out / "report.json", report)
    run = start_swanlab_v69(args, group="object-video-sequence-v69", job_type="teacher-calibration", config=vars(args))
    if run is not None:
        log_table_v69(run, "teacher_calibration/cases",
                      ["case", "source", "object", "point_id", "frame", "human_visible", "first_visible", "relay_visible",
                       "first_error_px", "relay_error_px", "agreement_error_px", "supplied_anchor"], rows)
        set_results_v69(run, {"visibility": report["visibility"], "threshold_sweep": sweep, "production_threshold_modified": False})
        log_evidence_v69(run, [out / "report.json"], base_path=out)
        run.finish()
    print(f"[teacher-calibration-v69] report={out / 'report.json'} observations={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
