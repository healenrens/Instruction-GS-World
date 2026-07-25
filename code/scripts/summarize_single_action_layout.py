"""Apply posterior and canonical-primary gates to one action layout."""
from __future__ import annotations

import argparse
import json
import os


METRICS = ("feature_mse", "latent_mse", "rgb_distance")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--residual_dim", type=int, required=True)
    parser.add_argument("--canonical_center_gate", type=float, default=1.0)
    parser.add_argument("--canonical_activity_gate", action="store_true")
    parser.add_argument("--canonical_activity_power", type=float, default=0.5)
    parser.add_argument("--residual_gate", type=float, required=True)
    parser.add_argument("--residual_dropout", type=float, default=0.0)
    parser.add_argument(
        "--semantic_action_basis",
        choices=("fixed", "learned", "rgb"),
        default="fixed",
    )
    parser.add_argument("--checkpoint_step", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.run)
    splits = {}
    for split in ("train", "heldseed", "heldtask"):
        path = os.path.join(
            root,
            f"posterior_gate_step{args.checkpoint_step}_{split}.json",
        )
        report = _load(path)
        if report.get("status") != "ok":
            raise ValueError(f"invalid posterior report: {path}")
        if report.get("action_residual_dim") != args.residual_dim:
            raise ValueError(f"residual dimension mismatch: {path}")
        if report.get("canonical_center_gate", 1.0) != args.canonical_center_gate:
            raise ValueError(f"canonical center gate mismatch: {path}")
        if report.get("canonical_activity_gate", False) != args.canonical_activity_gate:
            raise ValueError(f"canonical activity gate mismatch: {path}")
        if report.get("canonical_activity_power", 0.5) != args.canonical_activity_power:
            raise ValueError(f"canonical activity power mismatch: {path}")
        if report.get("action_residual_gate") != args.residual_gate:
            raise ValueError(f"residual gate mismatch: {path}")
        if report.get("action_residual_dropout") != args.residual_dropout:
            raise ValueError(f"residual dropout mismatch: {path}")
        if report.get("learned_semantic_action_basis") != (
            args.semantic_action_basis == "learned"
        ):
            raise ValueError(f"semantic action basis mismatch: {path}")
        if report.get("rgb_semantic_action", False) != (
            args.semantic_action_basis == "rgb"
        ):
            raise ValueError(f"RGB semantic action mismatch: {path}")
        metrics = report["metrics"]
        zero = {
            metric: metrics["posterior_comparison"][metric]["zero_action"]
            for metric in METRICS
        }
        splits[split] = {
            "samples": metrics["samples"],
            "posterior": {
                metric: metrics["mean"][metric]["posterior"]
                for metric in METRICS
            },
            "zero_action": {
                metric: metrics["mean"][metric]["zero_action"]
                for metric in METRICS
            },
            "posterior_vs_zero": zero,
            "all_mean_improvements_positive": all(
                item["absolute_improvement"] > 0.0 for item in zero.values()
            ),
            "all_margins_exceed_2se": all(
                item["positive_2se_margin"] for item in zero.values()
            ),
        }
    components = {}
    for split in ("heldseed", "heldtask"):
        path = os.path.join(
            root,
            f"action_component_ablation_step{args.checkpoint_step}_{split}.json",
        )
        report = _load(path)
        if report.get("status") != "ok":
            raise ValueError(f"invalid component report: {path}")
        if report.get("canonical_center_gate", 1.0) != args.canonical_center_gate:
            raise ValueError(f"component center gate mismatch: {path}")
        if report.get("canonical_activity_gate", False) != args.canonical_activity_gate:
            raise ValueError(f"component activity gate mismatch: {path}")
        if report.get("canonical_activity_power", 0.5) != args.canonical_activity_power:
            raise ValueError(f"component activity power mismatch: {path}")
        if report.get("learned_semantic_action_basis") != (
            args.semantic_action_basis == "learned"
        ) or report.get("rgb_semantic_action", False) != (
            args.semantic_action_basis == "rgb"
        ):
            raise ValueError(f"component semantic action mismatch: {path}")
        means = report["metrics"]["mean"]
        canonical_primary = {
            metric: (
                means[metric]["posterior_canonical_only"]
                < means[metric]["zero_action"]
                and means[metric]["posterior_canonical_only"]
                <= means[metric]["posterior_residual_only"]
            )
            for metric in METRICS
        }
        components[split] = {
            "samples": report["metrics"]["samples"],
            "mean": {
                metric: {
                    variant: means[metric][variant]
                    for variant in (
                        "posterior",
                        "posterior_canonical_only",
                        "posterior_residual_only",
                        "zero_action",
                    )
                }
                for metric in METRICS
            },
            "canonical_primary": canonical_primary,
            "canonical_primary_all_metrics": all(
                canonical_primary.values()
            ),
        }
    posterior_gate = all(
        splits[split]["all_mean_improvements_positive"]
        for split in ("heldseed", "heldtask")
    )
    canonical_gate = all(
        components[split]["canonical_primary_all_metrics"]
        for split in ("heldseed", "heldtask")
    )
    summary = {
        "status": "ok",
        "run": root,
        "residual_dim": args.residual_dim,
        "canonical_center_gate": args.canonical_center_gate,
        "canonical_activity_gate": args.canonical_activity_gate,
        "canonical_activity_power": args.canonical_activity_power,
        "residual_gate": args.residual_gate,
        "residual_dropout": args.residual_dropout,
        "semantic_action_basis": args.semantic_action_basis,
        "splits": splits,
        "components": components,
        "posterior_held_gate": posterior_gate,
        "canonical_primary_held_gate": canonical_gate,
        "eligible_for_prior": posterior_gate and canonical_gate,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
