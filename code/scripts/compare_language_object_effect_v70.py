#!/usr/bin/env python3
"""Paired episode-level comparisons; no claims based on training flow loss."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


METRICS = ("ade_px", "fde_px", "relative_ade", "relative_fde", "static_drift_px",
           "error_1s_px", "error_3s_px", "error_5s_px")
CONDITIONS = ("correct_language/sample_0", "no_language/sample_0", "shuffled_language/sample_0",
              "posterior_mean", "persistence")


def paired_comparison(first, second, first_condition, second_condition, seed, bootstrap):
    def cases(report):
        return {(case["window_id"], tuple(case["frame_indices"])): case for case in report["cases"]}

    left, right = cases(first), cases(second)
    common = sorted(left.keys() & right.keys())
    rows = []
    for key in common:
        a, b = left[key]["trajectory_metrics"], right[key]["trajectory_metrics"]
        if first_condition not in a or second_condition not in b:
            continue
        row = {"window_id": key[0], "source": left[key]["source"]}
        for metric in METRICS:
            av, bv = a[first_condition][metric]["mean"], b[second_condition][metric]["mean"]
            row[metric] = {"first": av, "second": bv,
                           "delta": bv-av if av is not None and bv is not None else None}
        rows.append(row)
    summaries = {}
    for source in ["all", *sorted({row["source"] for row in rows})]:
        summaries[source] = {}
        for metric in METRICS:
            values = [row[metric] for row in rows if (source == "all" or row["source"] == source)
                      and row[metric]["delta"] is not None]
            if not values:
                summaries[source][metric] = {"episodes": 0}
                continue
            delta = [v["delta"] for v in values]
            rng = np.random.default_rng(seed)
            draws = rng.integers(len(delta), size=(bootstrap, len(delta)))
            interval = np.quantile(np.asarray(delta)[draws].mean(axis=1), [.025, .975]).tolist()
            baseline = statistics.mean(v["first"] for v in values)
            change = statistics.mean(delta)
            summaries[source][metric] = {
                "episodes": len(delta), "first_mean": baseline,
                "second_mean": statistics.mean(v["second"] for v in values),
                "delta_second_minus_first": change,
                "relative_reduction": -change / baseline if baseline > 0 else None,
                "episode_win_fraction": sum(v < 0 for v in delta)/len(delta),
                "delta_ci95": interval,
            }
    return {"first_condition": first_condition, "second_condition": second_condition,
            "unpaired_first": len(left.keys()-right.keys()), "unpaired_second": len(right.keys()-left.keys()),
            "summaries": summaries, "episodes": rows}


def compare_reports(first, second, seed=17, bootstrap=2000):
    comparisons = {"checkpoint/"+name: paired_comparison(first, second, name, name, seed, bootstrap)
                   for name in CONDITIONS}
    for label, report in (("first", first), ("second", second)):
        for control in ("no_language/sample_0", "shuffled_language/sample_0", "persistence", "posterior_mean"):
            comparisons[f"{label}/correct_vs_{control}"] = paired_comparison(
                report, report, control, "correct_language/sample_0", seed, bootstrap)
    return {"first_step": first["checkpoint"]["step"], "second_step": second["checkpoint"]["step"],
            "primary": "checkpoint/correct_language/sample_0",
            "difference": "second minus first; negative is lower error",
            "aggregation": "mean per trajectory, mean per episode, paired bootstrap over episodes",
            "teacher_is_pseudo_measurement": True,
            "shuffled_language": "condition ablation, not independently labeled conflicting instruction",
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    first, second = [json.loads(Path(path).read_text()) for path in (args.first, args.second)]
    report = compare_reports(first, second, args.seed, args.bootstrap)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(report, indent=2))
    lines = ["# V70 Paired Evaluation", "", f'Steps: {report["first_step"]} -> {report["second_step"]}',
             "", "Primary: fixed-seed single sample. Negative delta means lower error.",
             "Tracker coordinates are pseudo measurements, not independently verified object truth.",
             "", "| Metric | First | Second | Delta | Paired episode 95% CI | Episodes |",
             "|---|---:|---:|---:|---|---:|"]
    main_result = report["comparisons"][report["primary"]]["summaries"]["all"]
    for metric, value in main_result.items():
        if value["episodes"]:
            lines.append(f'| {metric} | {value["first_mean"]:.4f} | {value["second_mean"]:.4f} | '
                         f'{value["delta_second_minus_first"]:.4f} | {value["delta_ci95"]} | {value["episodes"]} |')
    lines.extend(["", f'[Earlier checkpoint videos]({Path(args.first).parent.name}/index.html)',
                  f'[Later checkpoint videos]({Path(args.second).parent.name}/index.html)',
                  "", "All language contrasts, per-source comparisons and individual episodes: comparison.json."])
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"event": "v70_paired_evaluation_complete", "output": str(output),
                      "primary": main_result}), flush=True)


if __name__ == "__main__":
    main()
