"""Summarize deployable Prior gates for the selected residual bottleneck."""
from __future__ import annotations

import argparse
import json
import os


SPLITS = ("train", "heldseed", "heldtask")
METRICS = ("feature_mse", "latent_mse", "rgb_distance")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_summary", required=True)
    parser.add_argument(
        "--prior_run_template",
        required=True,
        help="absolute path containing {residual_dim}",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    residual_summary = _load(args.residual_summary)
    selected = residual_summary.get("selected_for_prior")
    if selected is None:
        raise ValueError("residual sweep did not select a Prior configuration")
    if "{residual_dim}" not in args.prior_run_template:
        raise ValueError("prior_run_template must contain {residual_dim}")
    root = os.path.abspath(
        args.prior_run_template.format(residual_dim=selected)
    )
    splits = {}
    for split in SPLITS:
        path = os.path.join(root, f"prior_gate_step100_{split}.json")
        report = _load(path)
        if report.get("status") != "ok":
            raise ValueError(f"invalid Prior report: {path}")
        if report.get("action_residual_dim") != selected:
            raise ValueError(f"Prior residual mismatch: {path}")
        metrics = report["metrics"]
        splits[split] = {
            "samples": metrics["samples"],
            "mean": metrics["mean"],
            "comparison": metrics["comparison"],
            "posterior_gap_closure": metrics["posterior_gap_closure"],
            "diagnostics": metrics["diagnostics"],
            "gate": metrics["gate"],
        }
    held = [splits[name]["gate"] for name in ("heldseed", "heldtask")]
    summary = {
        "status": "ok",
        "selected_residual_dim": selected,
        "prior_root": root,
        "splits": splits,
        "held_prior_beats_zero_all_metrics": all(
            gate["prior_beats_zero_all_metrics"] for gate in held
        ),
        "held_prior_beats_zero_all_metrics_2se": all(
            gate["prior_beats_zero_all_metrics_2se"] for gate in held
        ),
        "held_correct_instruction_beats_wrong_all_metrics": all(
            gate["correct_instruction_beats_wrong_all_metrics"] for gate in held
        ),
        "held_correct_instruction_beats_wrong_all_metrics_2se": all(
            gate["correct_instruction_beats_wrong_all_metrics_2se"]
            for gate in held
        ),
        "metrics": list(METRICS),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
