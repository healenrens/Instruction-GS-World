#!/usr/bin/env python3
"""Pair actual evaluated windows and report errors before episode-level summaries."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read_cases(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def case_key(row):
    return row["case_id"], tuple(row["frame_indices"]), tuple(row["times"]), row["image_size"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference cases.jsonl.")
    parser.add_argument("--candidate", required=True, help="Candidate cases.jsonl; may be the same file.")
    parser.add_argument("--reference-condition", default="posterior")
    parser.add_argument("--candidate-condition", default="posterior")
    parser.add_argument("--metric", choices=("rgb_l1", "psnr", "ssim", "appearance_change_weighted_l1", "temporal_rgb_delta_l1"), default="rgb_l1")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference, candidate = read_cases(args.reference), read_cases(args.candidate)
    other = {case_key(row): row for row in candidate}
    direction = 1 if args.metric in ("psnr", "ssim") else -1
    pairs, missing = [], []
    for left in reference:
        right = other.get(case_key(left))
        if right is None:
            missing.append({"case_id": left["case_id"], "reason": "no_matching_window"})
            continue
        first = {row["frame"]: row for row in left["conditions"][args.reference_condition]}
        second = {row["frame"]: row for row in right["conditions"][args.candidate_condition]}
        frames = sorted(frame for frame in first.keys() & second.keys()
                        if first[frame].get(args.metric) is not None and second[frame].get(args.metric) is not None)
        if not frames:
            missing.append({"case_id": left["case_id"], "reason": "metric_unavailable"})
            continue
        before = float(np.mean([first[frame][args.metric] for frame in frames]))
        after = float(np.mean([second[frame][args.metric] for frame in frames]))
        terminal = len(left["frame_indices"])-left["history_frames"]-1
        pairs.append({"case_id": left["case_id"], "source": left["source"], "episode_index": left["episode_index"],
                      "group": right.get("group", left.get("group")), "instruction": left["instruction"],
                      "reference_partition": left["partition"], "candidate_partition": right["partition"],
                      "reference": before, "candidate": after, "gain": direction*(after-before),
                      "frames_compared": len(frames),
                      "terminal_reference": first.get(terminal, {}).get(args.metric),
                      "terminal_candidate": second.get(terminal, {}).get(args.metric)})
    episodes = defaultdict(list)
    for row in pairs:
        episodes[(row["source"], row["group"], row["episode_index"])].append(row)
    episode_rows = [{"source": key[0], **{field: float(np.mean([row[field] for row in values]))
                                        for field in ("reference", "candidate", "gain")}}
                    for key, values in episodes.items()]
    by_source = defaultdict(list)
    for row in episode_rows:
        by_source[row["source"]].append(row)
    summaries = {}
    for source, values in by_source.items():
        gains = np.array([row["gain"] for row in values])
        summaries[source] = {"episodes": len(values), "reference": float(np.mean([row["reference"] for row in values])),
                             "candidate": float(np.mean([row["candidate"] for row in values])),
                             "gain_mean": float(gains.mean()), "gain_median": float(np.median(gains)),
                             "gain_q05": float(np.quantile(gains, .05)), "gain_q95": float(np.quantile(gains, .95)),
                             "candidate_better_episodes": int((gains > 0).sum())}
    interval = None
    if episode_rows:
        rng = np.random.default_rng(args.seed)
        draws = np.zeros(args.bootstrap_samples)
        for values in by_source.values():
            gains = np.array([row["gain"] for row in values])
            draws += rng.choice(gains, (args.bootstrap_samples, len(gains)), replace=True).sum(1)/len(episode_rows)
        interval = np.quantile(draws, [.025, .975]).tolist()
    report = {"reference": args.reference, "candidate": args.candidate,
              "reference_condition": args.reference_condition, "candidate_condition": args.candidate_condition,
              "metric": args.metric, "positive_gain_means": "candidate_has_lower_error_or_higher_similarity",
              "case_pairs": len(pairs), "episode_pairs": len(episode_rows), "missing_reference_pairs": missing,
              "unpaired_candidate_cases": len(set(other)-{case_key(row) for row in reference}),
              "episode_gain_mean": float(np.mean([row["gain"] for row in episode_rows])) if episode_rows else None,
              "gain_ci95": interval, "bootstrap_unit": "episode, stratified by source",
              "terminal_policy": "Only the final requested frame; an earlier visible frame is never substituted.",
              "by_source": summaries, "cases": pairs}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if pairs:
        with out.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    print(json.dumps({key: report[key] for key in ("case_pairs", "episode_pairs", "episode_gain_mean", "gain_ci95")}), flush=True)


if __name__ == "__main__":
    main()
