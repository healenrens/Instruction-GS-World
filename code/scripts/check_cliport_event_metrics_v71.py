#!/usr/bin/env python3
"""Exercise independent scores with real requested/opposite outcomes, copy and identity swaps."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.data import SequenceConfig, write_json
from igsw.object_event_wm.independent_metrics import score_entities, score_rgb_outcomes, physical_motion_context
from igsw.object_event_wm.paired_event_summary import summarize_paired_events
from igsw.object_event_wm.simulation_data import load_episode, load_truth, nearest_indices, requested_times


def read_case(root, entry, config):
    path = root/entry["episode"]
    meta, rgb, recorded = load_episode(path)
    times = requested_times(config)
    indices = nearest_indices(recorded, times)
    return meta, rgb[indices], load_truth(path, indices), times, bool(np.allclose(recorded[indices], times, atol=1e-8, rtol=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    collection = Path(args.collection)
    source = json.loads(collection.read_text())
    config = SequenceConfig(history_seconds=source["history_seconds"], future_seconds=source["future_seconds"],
                            history_frames=16, future_frames=50)
    root, th = collection.parent, config.history_frames
    groups = defaultdict(list)
    for entry in source["episodes"]:
        groups[entry["scene_seed"]].append(entry)
    cases, excluded = [], []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"cases.jsonl").open("w") as stream:
        for entries in groups.values():
            if len(entries) != 2:
                excluded.extend(entry["case_id"] for entry in entries)
                continue
            for entry, alternative in ((entries[0], entries[1]), (entries[1], entries[0])):
                meta, rgb, truth, times, exact = read_case(root, entry, config)
                other_meta, other_rgb, other_truth, _, other_exact = read_case(root, alternative, config)
                anchor = truth["segmentation"][th-1]
                labels = truth["segmentation"][th:]
                actors = sorted({row["actor_body_id"] for row in meta["program"]})
                swapped = labels.copy()
                swapped[labels == actors[0]], swapped[labels == actors[1]] = actors[1], actors[0]
                controls = {
                    "oracle": (rgb[th:], labels),
                    "opposite_order": (other_rgb[th:], other_truth["segmentation"][th:]),
                    "current_copy": (np.broadcast_to(rgb[th-1], rgb[th:].shape), np.broadcast_to(anchor, labels.shape)),
                    "identity_swap": (rgb[th:], swapped),
                    "two_outcome_rgb_mean": ((rgb[th:].astype(np.float32)+other_rgb[th:].astype(np.float32))/510., None),
                }
                results = {}
                for name, (predicted, predicted_labels) in controls.items():
                    result = {"reference_combination": meta["combination"],
                              "rgb": score_rgb_outcomes(predicted, rgb[th:], anchor, labels, actors,
                                meta["robot_and_tool_body_ids"], times[th:], other_rgb[th:], other_truth["segmentation"][th:])}
                    if predicted_labels is not None:
                        result["objects"] = score_entities(anchor, predicted_labels, anchor, labels, meta["entities"], times[th:])
                    results[name] = result
                oracle = results["oracle"]
                checks = {"fixed_requested_grid_available": exact and other_exact,
                          "both_requested_programs_realized": meta["status"] == other_meta["status"] == "completed",
                          "paired_history_pixels_identical": bool(np.array_equal(rgb[:th], other_rgb[:th])),
                          "oracle_rgb_error_zero": oracle["rgb"]["full_rgb_l1"] == 0,
                          "oracle_visible_instance_iou_one": all(row["mean_visible_iou"] == 1. for row in oracle["objects"]["entities"] if row["scored_visible_frames"]),
                          "oracle_centroid_error_zero": all(row["centroid_rmse_px"] == 0. for row in oracle["objects"]["entities"] if row["scored_visible_frames"]),
                          "correct_outcome_preferred": oracle["rgb"]["order_preference_margin"] > 0,
                          "opposite_outcome_rejected": results["opposite_order"]["rgb"]["order_preference_margin"] < 0,
                          "outcome_mean_has_no_order_preference": abs(results["two_outcome_rgb_mean"]["rgb"]["order_preference_margin"]) < 1e-6,
                          "identity_swap_changes_tracking_not_rgb": results["identity_swap"]["rgb"]["full_rgb_l1"] == 0 and any(
                              row["mean_visible_iou"] < 1. for row in results["identity_swap"]["objects"]["entities"] if row["body_id"] in actors and row["scored_visible_frames"]),
                          "copy_has_nonzero_actor_error": results["current_copy"]["rgb"]["actor_region_rgb_l1"] > 0}
                row = {"case_id": entry["case_id"], "scene_seed": entry["scene_seed"],
                       "combination": meta["combination"],
                       "alternative_case_id": alternative["case_id"], "instruction": meta["instruction"],
                       "alternative_instruction": other_meta["instruction"], "checks": checks, "conditions": results}
                row["gt_physical_motion"] = physical_motion_context(truth["poses_xyz_xyzw"][th-1, :, :3],
                    truth["poses_xyz_xyzw"][th:, :, :3], truth["body_ids"])
                cases.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                print(json.dumps({"case_id": row["case_id"], "checks": checks,
                                  "oracle_order_margin": oracle["rgb"]["order_preference_margin"]}), flush=True)
    paired = summarize_paired_events(cases)
    paired_checks = [{"scene_seed": pair["scene_seed"],
                      "oracle_positive": pair["conditions"]["oracle"]["paired_order_margin"] > 0,
                      "opposite_negative": pair["conditions"]["opposite_order"]["paired_order_margin"] < 0,
                      "copy_bias_cancels": abs(pair["conditions"]["current_copy"]["paired_order_margin"]) < 1e-6,
                      "fused_mean_bias_cancels": abs(pair["conditions"]["two_outcome_rgb_mean"]["paired_order_margin"]) < 1e-6}
                     for pair in paired["scene_pairs"]]
    pairs_passed = bool(paired_checks) and all(all(value for key, value in row.items() if key != "scene_seed")
                                              for row in paired_checks)
    report = {"task": "metric_falsification_on_real_simulator_sequences", "collection": str(collection.resolve()),
              "cases": len(cases), "excluded_unpaired_cases": excluded,
              "status": "passed" if cases and all(all(row["checks"].values()) for row in cases) and pairs_passed else "failed",
              "paired_outcome_preference": paired, "paired_checks": paired_checks,
              "model_evaluated": False, "physical_motion_onset_classifier_used": False,
              "per_case": [{"case_id": row["case_id"], "checks": row["checks"],
                            "oracle_order_margin": row["conditions"]["oracle"]["rgb"]["order_preference_margin"]} for row in cases]}
    write_json(out/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
