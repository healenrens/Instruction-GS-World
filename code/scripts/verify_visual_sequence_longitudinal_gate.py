"""Fail-closed stability gate across long posterior-core checkpoints."""
from __future__ import annotations

import argparse
import json
import os


STEPS = (4000, 8000, 12000)
SPLITS = ("heldseed", "heldtask")
ACTION_SIGNALS = ("change_vs_shuffled", "change_vs_zero")


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def summarize_step(report: dict, expected_step: int) -> dict:
    evaluations = report["evaluation"]
    checkpoints = {
        os.path.abspath(evaluations[split]["checkpoint"])
        for split in SPLITS
    }
    data_roots = {
        os.path.abspath(evaluations[split]["data"])
        for split in SPLITS
    }
    data_hashes = {
        str(evaluations[split]["data_sha256"])
        for split in SPLITS
    }
    checkpoint = next(iter(checkpoints)) if len(checkpoints) == 1 else ""
    return {
        "status": report["status"],
        "expected_step": expected_step,
        "checkpoint": checkpoint,
        "checkpoint_paths": sorted(checkpoints),
        "data_roots": sorted(data_roots),
        "data_sha256": sorted(data_hashes),
        "reported_maximum_step": int(report["thresholds"]["maximum_step"]),
        "training_first_step": int(report["training"]["first_step"]),
        "training_last_step": int(report["training"]["last_step"]),
        "signals": {
            split: {
                name: float(evaluations[split][name])
                for name in ACTION_SIGNALS
            }
            for split in SPLITS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step4000", required=True)
    parser.add_argument("--step8000", required=True)
    parser.add_argument("--step12000", required=True)
    parser.add_argument("--minimum_final_retention", type=float, default=0.70)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 < args.minimum_final_retention <= 1.0:
        raise ValueError("minimum final retention must be in (0,1]")

    paths = {
        4000: os.path.abspath(args.step4000),
        8000: os.path.abspath(args.step8000),
        12000: os.path.abspath(args.step12000),
    }
    summaries = {
        step: summarize_step(load_report(paths[step]), step)
        for step in STEPS
    }
    checks: dict[str, bool] = {}
    for step, summary in summaries.items():
        checks[f"step{step}_gate_passed"] = summary["status"] == "pass"
        checks[f"step{step}_checkpoint_identity"] = (
            len(summary["checkpoint_paths"]) == 1
            and os.path.isfile(summary["checkpoint"])
            and os.path.basename(summary["checkpoint"])
            == f"joint_{step:07d}.pt"
        )
        checks[f"step{step}_training_window"] = (
            summary["reported_maximum_step"] == step
            and summary["training_last_step"] == step
            and summary["training_first_step"] <= step
        )
        checks[f"step{step}_single_data_root"] = (
            len(summary["data_roots"]) == 1
            and os.path.isdir(summary["data_roots"][0])
        )
        checks[f"step{step}_single_data_identity"] = (
            len(summary["data_sha256"]) == 1
            and len(summary["data_sha256"][0]) == 64
        )

    checks["shared_checkpoint_directory"] = len(
        {
            os.path.dirname(summary["checkpoint"])
            for summary in summaries.values()
        }
    ) == 1
    checks["shared_data_root"] = len(
        {
            root
            for summary in summaries.values()
            for root in summary["data_roots"]
        }
    ) == 1
    checks["shared_data_identity"] = len(
        {
            digest
            for summary in summaries.values()
            for digest in summary["data_sha256"]
        }
    ) == 1

    retention = {}
    for split in SPLITS:
        retention[split] = {}
        for signal in ACTION_SIGNALS:
            previous_best = max(
                summaries[step]["signals"][split][signal]
                for step in (4000, 8000)
            )
            final = summaries[12000]["signals"][split][signal]
            ratio = final / max(previous_best, 1e-8)
            retention[split][signal] = {
                "previous_best": previous_best,
                "final": final,
                "retention": ratio,
            }
            checks[f"{split}_{signal}_retained"] = (
                previous_best > 0.0
                and ratio >= args.minimum_final_retention
            )

    passed = all(checks.values())
    report = {
        "status": "pass" if passed else "fail",
        "scope": "posterior_core_stability_steps_4000_8000_12000",
        "minimum_final_retention": args.minimum_final_retention,
        "final_checkpoint": summaries[12000]["checkpoint"],
        "checks": checks,
        "retention": retention,
        "steps": {str(step): summaries[step] for step in STEPS},
        "source_reports": {str(step): paths[step] for step in STEPS},
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise AssertionError(f"longitudinal stability gate failed: {failed}")


if __name__ == "__main__":
    main()
