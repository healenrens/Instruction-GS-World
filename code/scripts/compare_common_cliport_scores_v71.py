#!/usr/bin/env python3
"""Compare methods on the same cases, reference frames and common-view protocol."""

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.paired_event_summary import summarize_paired_events


def read(directory):
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text())
    cases = {row["case_id"]: row for row in
             (json.loads(line) for line in (directory/"cases.jsonl").read_text().splitlines() if line.strip())}
    return report, cases


def select(case, mode, sample):
    return next((result for result in case["conditions"].values()
                 if result["mode"] == mode and result["sample_index"] in (None, sample)), None)


def measurements(result):
    frames = result["common_image_quality"]
    return {"actor_rgb_l1_native": result["rgb"]["actor_region_rgb_l1"],
            **{f"{key}_common": mean(row[key] for row in frames) if frames else None
               for key in ("rgb_l1", "psnr", "ssim")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", default="correct_language",
                        choices=("correct_language", "no_language", "opposite_order_language", "posterior", "zero_effect", "source_state_copy"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    left, lc = read(args.reference)
    right, rc = read(args.candidate)
    keys = ("contract", "manifest", "sequence", "common_max_edge", "resampling", "ground_truth")
    differences = {key: {"reference": left[key], "candidate": right[key]} for key in keys if left[key] != right[key]}
    common = sorted(lc.keys() & rc.keys())
    rows, excluded, pairs_left, pairs_right = [], [], [], []
    if not differences:
        for case_id in common:
            a, b = select(lc[case_id], args.mode, args.sample_index), select(rc[case_id], args.mode, args.sample_index)
            if a is None or b is None:
                excluded.append({"case_id": case_id, "reason": "requested condition/sample missing"})
                continue
            af = [(row["frame"], row["seconds"]) for row in a["common_image_quality"]]
            bf = [(row["frame"], row["seconds"]) for row in b["common_image_quality"]]
            if a["reference_case_id"] != b["reference_case_id"] or af != bf or a["uses_future_rgb"] != b["uses_future_rgb"]:
                excluded.append({"case_id": case_id, "reason": "reference frames or future-access condition differs"})
                continue
            am, bm = measurements(a), measurements(b)
            rows.append({"case_id": case_id, "scene_seed": lc[case_id]["scene_seed"],
                         "reference_case_id": a["reference_case_id"], "reference": am, "candidate": bm,
                         "model_image_sizes": {"reference": a["model_image_size"], "candidate": b["model_image_size"]},
                         "candidate_minus_reference": {key: bm[key]-am[key] if am[key] is not None and bm[key] is not None else None
                                                       for key in am}})
            pairs_left.append({**lc[case_id], "conditions": {args.mode: a}})
            pairs_right.append({**rc[case_id], "conditions": {args.mode: b}})
    aggregate = {}
    if rows:
        for key in rows[0]["reference"]:
            available = [row for row in rows if row["candidate_minus_reference"][key] is not None]
            values = [row["candidate_minus_reference"][key] for row in available]
            aggregate[key] = {"paired_cases": len(available),
                              "reference_mean": mean(row["reference"][key] for row in available) if available else None,
                              "candidate_mean": mean(row["candidate"][key] for row in available) if available else None,
                              "mean_difference": mean(values) if values else None,
                              "median_difference": median(values) if values else None,
                              "lower_is_better": "l1" in key}
    report = {"reference_directory": str(Path(args.reference).resolve()), "candidate_directory": str(Path(args.candidate).resolve()),
              "mode": args.mode, "sample_index": args.sample_index, "protocol_differences": differences,
              "reference_only_cases": sorted(lc.keys()-rc.keys()), "candidate_only_cases": sorted(rc.keys()-lc.keys()),
              "excluded_cases": excluded, "paired_cases": rows, "metrics": aggregate,
              "paired_order_reference": summarize_paired_events(pairs_left),
              "paired_order_candidate": summarize_paired_events(pairs_right),
              "scope": "matched-case output comparison; not a test of statistical significance",
              "not_controlled_here": ["model input resolution", "pretrained assets", "training compute"]}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(path), "paired_cases": len(rows), "protocol_differences": differences,
                      "excluded_cases": excluded, "metrics": aggregate}), flush=True)


if __name__ == "__main__":
    main()
