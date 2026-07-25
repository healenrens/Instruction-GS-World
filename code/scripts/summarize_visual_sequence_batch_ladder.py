"""Select a visual-sequence microbatch from structured training artifacts."""
from __future__ import annotations

import argparse
import json
import math
import os


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_run(
    label: str,
    path: str,
    expected_effective_batch: int,
    min_headroom: float,
) -> dict:
    train_path = os.path.join(path, "train.jsonl")
    warm_path = os.path.join(path, "warm_start_report.json")
    latest_path = os.path.join(path, "latest.pt")
    records = load_jsonl(train_path) if os.path.isfile(train_path) else []
    warm = load_json(warm_path) if os.path.isfile(warm_path) else None
    last = records[-1] if records else None
    failures = []
    if not records:
        failures.append("missing_training_record")
    if not os.path.isfile(latest_path):
        failures.append("missing_final_checkpoint")
    if last is not None and not finite_numbers(last):
        failures.append("non_finite_metric")
    if last is not None and last.get("effective_batch") != expected_effective_batch:
        failures.append("effective_batch_mismatch")
    reserved_headroom = (
        last.get("memory_reserved_headroom_fraction", -1.0)
        if last is not None
        else -1.0
    )
    if last is not None and reserved_headroom < min_headroom:
        failures.append("insufficient_memory_headroom")
    if warm is None:
        failures.append("missing_warm_start_report")
    elif warm.get("missing") or warm.get("shape_mismatch"):
        failures.append("incompatible_warm_start")
    return {
        "label": label,
        "path": path,
        "passed": not failures,
        "failures": failures,
        "checkpoint": (
            os.path.realpath(latest_path)
            if os.path.isfile(latest_path)
            else None
        ),
        "last_record": last,
        "warm_start": {
            "loaded": warm.get("loaded"),
            "missing": warm.get("missing"),
            "shape_mismatch": warm.get("shape_mismatch"),
            "unexpected_count": len(warm.get("unexpected", ())),
        } if warm is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="repeat LABEL=/absolute/run/path",
    )
    parser.add_argument("--expected_effective_batch", type=int, default=256)
    parser.add_argument("--min_headroom", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_effective_batch < 1:
        raise ValueError("expected effective batch must be positive")
    if not 0.0 <= args.min_headroom < 1.0:
        raise ValueError("minimum memory headroom must be in [0, 1)")
    runs = []
    labels = set()
    for specification in args.run:
        label, path = specification.split("=", 1)
        path = os.path.abspath(path)
        if not label or label in labels or not os.path.isdir(path):
            raise ValueError(f"invalid or duplicate run specification: {specification}")
        labels.add(label)
        runs.append(
            inspect_run(
                label,
                path,
                args.expected_effective_batch,
                args.min_headroom,
            )
        )
    candidates = [run for run in runs if run["passed"]]
    selected = (
        max(
            candidates,
            key=lambda run: (
                run["last_record"]["steps_per_second"],
                run["last_record"]["micro_batch"],
            ),
        )
        if candidates
        else None
    )
    report = {
        "status": "passed" if selected is not None else "failed",
        "selection_rule": (
            "highest optimizer-step throughput with finite metrics, "
            f"effective batch {args.expected_effective_batch}, and "
            f"reserved-memory headroom >= {args.min_headroom:.2f}"
        ),
        "selected": selected["label"] if selected is not None else None,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    if selected is None:
        raise RuntimeError("no batch-ladder run passed the selection gate")


if __name__ == "__main__":
    main()
