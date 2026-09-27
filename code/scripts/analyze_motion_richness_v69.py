#!/usr/bin/env python3
"""Compare individual motion diagnostics with human window rankings; no fitted hidden score."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wandb_project", default="instruct-gs-world")
    p.add_argument("--wandb_entity", default="healenrenss-university-of-chinese-acadmic-and-science")
    p.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb_name", default="object_video_v69_human_motion_comparison")
    args = p.parse_args()
    rows = {row["window_id"]: row for row in read_json(args.report)["rows"]}
    pairs = read_json(args.labels)["pairs"]
    conditions = {"relative_span_p90": lambda row: row["relative_span_image_diagonal"]["p90"],
                  "motion_grid_coverage": lambda row: row["motion_covered_grid_fraction"],
                  "path_length_p90": lambda row: row["observed_path_length_px"]["p90"]}
    comparisons, counts = [], defaultdict(lambda: {"correct": 0, "count": 0})
    for pair in pairs:
        if pair["winner"] not in ("left", "right", "equal"):
            continue
        left, right = rows[pair["left"]], rows[pair["right"]]
        for metric, getter in conditions.items():
            a, b = getter(left), getter(right)
            if a is None or b is None:
                continue
            predicted = "equal" if a == b else "left" if a > b else "right"
            correct = predicted == pair["winner"] if pair["winner"] != "equal" else None
            key = (pair.get("partition", "unspecified"), left["source"], metric)
            if correct is not None:
                counts[key]["count"] += 1
                counts[key]["correct"] += int(correct)
            comparisons.append({**pair, "metric": metric, "left_value": a, "right_value": b, "predicted": predicted,
                                "absolute_difference": abs(a-b), "correct": correct})
    summary = [{"partition": partition, "source": source, "metric": metric, **value, "agreement": value["correct"]/value["count"]}
               for (partition, source, metric), value in sorted(counts.items())]
    write_json(args.output, {"status": "human_rank_comparison", "summary": summary, "comparisons": comparisons,
                            "sampling_modified": False, "automatic_acceptance": False})
    if args.wandb_mode != "disabled":
        import wandb
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         group="object-video-sequence-v69", job_type="motion-rank-analysis", mode=args.wandb_mode, config=vars(args))
        table = wandb.Table(columns=["partition", "source", "metric", "correct", "count", "agreement"])
        for row in summary:
            table.add_data(*[row[name] for name in table.columns])
        run.log({"motion_rank/summary": table, "motion_rank/automatic_acceptance": False})
        artifact = wandb.Artifact(args.wandb_name, type="human-motion-comparison")
        artifact.add_file(args.output)
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps({"output": args.output, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
