"""Fail-closed gate for posterior Dynamics collapse-repair experiments."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_training_tail(
    path: str,
    window: int,
    maximum_step: int,
) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    selected = [
        record
        for record in records
        if "rgb_future_delta_change_energy_ratio" in record
        and (maximum_step == 0 or int(record["global_step"]) <= maximum_step)
    ]
    if len(selected) < window:
        raise ValueError(
            f"training log has {len(selected)} collapse metrics, expected at least {window}"
        )
    return selected[-window:]


def _evaluation_metrics(path: str) -> dict[str, float | int | bool | str]:
    report = _load_json(path)
    evaluation = report["evaluation"]
    headline = evaluation["headline"]
    localization = evaluation["localization"]["mean"]
    change_comparison = evaluation["regions"]["change"]["comparison"][
        "charbonnier"
    ]
    return {
        "status": report["status"],
        "checkpoint": os.path.abspath(report["checkpoint"]),
        "checkpoint_global_step": int(report["checkpoint_global_step"]),
        "checkpoint_phase": report["checkpoint_phase"],
        "checkpoint_version": report["checkpoint_version"],
        "data": os.path.abspath(report["data"]),
        "data_sha256": str(report["data_sha256"]),
        "split": report["split"],
        "amp": report["amp"],
        "requested_max_items": int(report["requested_max_items"]),
        "available_samples": int(report["available_samples"]),
        "anchors": [int(anchor) for anchor in report["anchors"]],
        "samples": int(evaluation["samples"]),
        "clusters": int(evaluation["localization"]["clusters"]),
        "action_source": evaluation["action_source"],
        "deployable_prediction": bool(evaluation["deployable_prediction"]),
        "change_vs_shuffled": float(
            headline["change_posterior_vs_shuffled_relative"]
        ),
        "change_vs_zero": float(headline["change_posterior_vs_zero_relative"]),
        "change_vs_shuffled_significant": bool(
            change_comparison["posterior_vs_shuffled"].get(
                "positive_ci95_lower",
                False,
            )
        ),
        "change_vs_zero_significant": bool(
            change_comparison["posterior_vs_zero"].get(
                "positive_ci95_lower",
                False,
            )
        ),
        "static_vs_zero": float(
            headline["static_posterior_vs_zero_relative"]
        ),
        "change_topk_iou": float(headline["posterior_change_topk_iou"]),
        "change_magnitude": float(
            localization["change_magnitude"]["posterior"]
        ),
        "static_magnitude": float(
            localization["static_magnitude"]["posterior"]
        ),
        "change_static_ratio": float(
            localization["change_static_ratio"]["posterior"]
        ),
    }


def _sample_coverage(
    metrics: dict,
    minimum_samples: int,
) -> dict[str, int | bool]:
    expected_evaluated = min(
        metrics["requested_max_items"],
        metrics["available_samples"],
    )
    complete = (
        len(metrics["anchors"]) > 0
        and len(set(metrics["anchors"])) == len(metrics["anchors"])
        and metrics["requested_max_items"] >= minimum_samples
        and metrics["samples"] == expected_evaluated
        and metrics["samples"] >= minimum_samples
    )
    return {
        "anchor_count": len(metrics["anchors"]),
        "requested_max_items": metrics["requested_max_items"],
        "available_samples": metrics["available_samples"],
        "expected_evaluated_samples": expected_evaluated,
        "observed_evaluated_samples": metrics["samples"],
        "minimum_samples": minimum_samples,
        "complete": complete,
    }


def _training_metrics(records: list[dict]) -> dict[str, float | int]:
    keys = (
        "rgb_future_delta_change_energy_ratio",
        "rgb_future_delta_change_gain_over_copy",
        "rgb_future_delta_predicted_change_rms",
        "rgb_future_delta_target_change_rms",
        "grad_norm",
    )
    metrics: dict[str, float | int] = {
        "records": len(records),
        "first_step": int(records[0]["global_step"]),
        "last_step": int(records[-1]["global_step"]),
    }
    for key in keys:
        values = [float(record[key]) for record in records]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite training metric: {key}")
        metrics[f"median_{key}"] = statistics.median(values)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldseed", required=True)
    parser.add_argument("--heldtask", required=True)
    parser.add_argument("--train_log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--minimum_heldseed_samples",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--minimum_heldtask_samples",
        type=int,
        default=450,
    )
    parser.add_argument("--minimum_clusters", type=int, default=100)
    parser.add_argument("--minimum_change_gain", type=float, default=0.03)
    parser.add_argument("--minimum_change_iou", type=float, default=0.20)
    parser.add_argument("--minimum_change_magnitude", type=float, default=0.005)
    parser.add_argument("--minimum_change_static_ratio", type=float, default=1.5)
    parser.add_argument("--maximum_static_degradation", type=float, default=0.01)
    parser.add_argument("--minimum_energy_ratio", type=float, default=0.05)
    parser.add_argument("--training_window", type=int, default=20)
    parser.add_argument("--maximum_step", type=int, default=0)
    args = parser.parse_args()

    evaluations = {
        "heldseed": _evaluation_metrics(os.path.abspath(args.heldseed)),
        "heldtask": _evaluation_metrics(os.path.abspath(args.heldtask)),
    }
    training = _training_metrics(
        _load_training_tail(
            os.path.abspath(args.train_log),
            args.training_window,
            args.maximum_step,
        )
    )
    checks = {}
    minimum_samples = {
        "heldseed": args.minimum_heldseed_samples,
        "heldtask": args.minimum_heldtask_samples,
    }
    for split, metrics in evaluations.items():
        coverage = _sample_coverage(metrics, minimum_samples[split])
        metrics["sample_coverage"] = coverage
        checks[f"{split}_report_complete"] = (
            metrics["status"] == "ok"
            and metrics["split"] == split
            and coverage["complete"]
            and metrics["clusters"] >= args.minimum_clusters
            and metrics["action_source"] == "future_conditioned_posterior_oracle"
            and metrics["deployable_prediction"] is False
        )
        checks[f"{split}_action_specific"] = (
            metrics["change_vs_shuffled"] >= args.minimum_change_gain
            and metrics["change_vs_shuffled_significant"]
        )
        checks[f"{split}_beats_zero"] = (
            metrics["change_vs_zero"] >= args.minimum_change_gain
            and metrics["change_vs_zero_significant"]
        )
        checks[f"{split}_preserves_static_regions"] = (
            metrics["static_vs_zero"] >= -args.maximum_static_degradation
        )
        checks[f"{split}_localized_change"] = (
            metrics["change_topk_iou"] >= args.minimum_change_iou
            and metrics["change_magnitude"] >= args.minimum_change_magnitude
            and metrics["change_static_ratio"]
            >= args.minimum_change_static_ratio
        )
    checkpoint_paths = {
        metrics["checkpoint"] for metrics in evaluations.values()
    }
    checkpoint_steps = {
        metrics["checkpoint_global_step"] for metrics in evaluations.values()
    }
    checkpoint_phases = {
        metrics["checkpoint_phase"] for metrics in evaluations.values()
    }
    data_roots = {metrics["data"] for metrics in evaluations.values()}
    data_hashes = {metrics["data_sha256"] for metrics in evaluations.values()}
    expected_checkpoint_name = (
        f"joint_{args.maximum_step:07d}.pt" if args.maximum_step > 0 else ""
    )
    checks["shared_checkpoint_identity"] = (
        len(checkpoint_paths) == 1
        and len(checkpoint_steps) == 1
        and checkpoint_phases == {"joint"}
        and all(os.path.isfile(path) for path in checkpoint_paths)
        and (
            args.maximum_step == 0
            or (
                checkpoint_steps == {args.maximum_step}
                and os.path.basename(next(iter(checkpoint_paths)))
                == expected_checkpoint_name
            )
        )
    )
    checks["shared_data_contract"] = (
        len(data_roots) == 1
        and len(data_hashes) == 1
        and next(iter(data_hashes), "") != ""
        and all(os.path.isdir(path) for path in data_roots)
    )
    checks["training_window_matches_checkpoint"] = (
        args.maximum_step == 0
        or training["last_step"] == args.maximum_step
    )
    checks["training_change_energy_alive"] = (
        training["median_rgb_future_delta_change_energy_ratio"]
        >= args.minimum_energy_ratio
        and training["median_rgb_future_delta_predicted_change_rms"] > 1e-4
    )
    checks["training_change_gain_positive"] = (
        training["median_rgb_future_delta_change_gain_over_copy"] > 0.0
    )
    passed = all(checks.values())
    report = {
        "status": "pass" if passed else "fail",
        "thresholds": {
            "minimum_heldseed_samples": args.minimum_heldseed_samples,
            "minimum_heldtask_samples": args.minimum_heldtask_samples,
            "minimum_clusters": args.minimum_clusters,
            "minimum_change_gain": args.minimum_change_gain,
            "minimum_change_iou": args.minimum_change_iou,
            "minimum_change_magnitude": args.minimum_change_magnitude,
            "minimum_change_static_ratio": args.minimum_change_static_ratio,
            "maximum_static_degradation": args.maximum_static_degradation,
            "minimum_energy_ratio": args.minimum_energy_ratio,
            "training_window": args.training_window,
            "maximum_step": args.maximum_step,
        },
        "checks": checks,
        "evaluation": evaluations,
        "training": training,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise AssertionError(f"collapse gate failed: {failed}")


if __name__ == "__main__":
    main()
