"""Summarize held-split Prior gates across a checkpoint sweep."""
from __future__ import annotations

import argparse
import json
import os


METRICS = ("feature_mse", "latent_mse", "rgb_distance")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _comparison(report: dict, name: str) -> dict[str, dict]:
    comparisons = report["metrics"]["comparison"]
    return {
        metric: comparisons[metric][name]
        for metric in METRICS
    }


def _validate(report: dict, path: str, split: str) -> None:
    expected = {
        "status": "ok",
        "split": split,
        "action_anchor": "object_slot",
        "action_dim": 6,
        "action_residual_dim": 0,
        "wrong_instruction_scope": "prior_only",
        "wrong_task_rank": 0,
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid Prior report {path}: {mismatches}")


def _standardized_min(comparisons: dict[str, dict]) -> float:
    return min(
        item["absolute_improvement"]
        / max(item["paired_standard_error"], 1e-12)
        for item in comparisons.values()
    )


def _step_summary(
    run: str,
    step: int,
    splits: tuple[str, ...],
    suffix: str,
) -> dict:
    split_reports = {}
    for split in splits:
        path = os.path.join(
            run,
            f"prior_gate_step{step}_{split}_{suffix}.json",
        )
        report = _load(path)
        _validate(report, path, split)
        prior = _comparison(report, "prior_vs_zero")
        instruction = _comparison(report, "prior_vs_wrong_instruction")
        split_reports[split] = {
            "path": os.path.abspath(path),
            "samples": report["metrics"]["samples"],
            "prior_vs_zero": prior,
            "correct_vs_wrong_instruction": instruction,
            "gate": report["metrics"]["gate"],
        }

    prior_items = [
        item
        for split in splits
        for item in split_reports[split]["prior_vs_zero"].values()
    ]
    instruction_items = [
        item
        for split in splits
        for item in split_reports[split][
            "correct_vs_wrong_instruction"
        ].values()
    ]
    prior_positive = sum(
        item["absolute_improvement"] > 0.0 for item in prior_items
    )
    instruction_positive = sum(
        item["absolute_improvement"] > 0.0
        for item in instruction_items
    )
    prior_2se = sum(item["positive_2se_margin"] for item in prior_items)
    instruction_2se = sum(
        item["positive_2se_margin"] for item in instruction_items
    )
    return {
        "step": step,
        "splits": split_reports,
        "prior_positive_count": prior_positive,
        "instruction_positive_count": instruction_positive,
        "prior_2se_count": prior_2se,
        "instruction_2se_count": instruction_2se,
        "comparison_count": len(prior_items),
        "minimum_prior_standardized_margin": min(
            _standardized_min(
                split_reports[split]["prior_vs_zero"]
            )
            for split in splits
        ),
        "minimum_instruction_standardized_margin": min(
            _standardized_min(
                split_reports[split][
                    "correct_vs_wrong_instruction"
                ]
            )
            for split in splits
        ),
        "mean_gate_pass": (
            prior_positive == len(prior_items)
            and instruction_positive == len(instruction_items)
        ),
        "strict_2se_gate_pass": (
            prior_2se == len(prior_items)
            and instruction_2se == len(instruction_items)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("heldseed", "heldtask"),
    )
    parser.add_argument("--suffix", default="rank0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("checkpoint steps must be unique")
    run = os.path.abspath(args.run)
    splits = tuple(args.splits)
    summaries = [
        _step_summary(run, step, splits, args.suffix)
        for step in sorted(args.steps)
    ]
    ranked = sorted(
        summaries,
        key=lambda item: (
            item["mean_gate_pass"],
            item["instruction_positive_count"],
            item["minimum_instruction_standardized_margin"],
            item["minimum_prior_standardized_margin"],
        ),
        reverse=True,
    )
    mean_passes = [
        item["step"] for item in summaries if item["mean_gate_pass"]
    ]
    strict_passes = [
        item["step"] for item in summaries if item["strict_2se_gate_pass"]
    ]
    report = {
        "status": "ok",
        "run": run,
        "splits": list(splits),
        "metrics": list(METRICS),
        "steps": summaries,
        "mean_gate_pass_steps": mean_passes,
        "strict_2se_gate_pass_steps": strict_passes,
        "best_diagnostic_step": ranked[0]["step"],
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
