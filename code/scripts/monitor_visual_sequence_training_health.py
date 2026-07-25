"""Continuously record fail-loud health signals for a long WM training run."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time


HEALTH_KEYS = (
    "total",
    "grad_norm",
    "action_specificity_sample_std",
    "action_specificity_slot_rms",
    "rgb_future_delta_change_energy_ratio",
    "rgb_future_delta_change_gain_over_copy",
    "rgb_future_delta_predicted_change_rms",
)


def load_complete_records(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def read_pid(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        value = handle.read().strip()
    return int(value) if value else 0


def pid_alive(pid: int) -> bool:
    return pid > 0 and os.path.isdir(f"/proc/{pid}")


def write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def trend_report(selected: list[dict], args) -> dict:
    baseline = [
        record
        for record in selected
        if int(record["global_step"]) >= args.collapse_check_start
    ]
    if len(baseline) < args.window:
        return {}
    keys = (
        "action_specificity_slot_rms",
        "rgb_future_delta_change_energy_ratio",
        "rgb_future_delta_change_gain_over_copy",
    )
    rolling = {
        key: [
            statistics.median(
                float(record[key])
                for record in baseline[end - args.window : end]
            )
            for end in range(args.window, len(baseline) + 1)
        ]
        for key in keys
    }
    current = {key: values[-1] for key, values in rolling.items()}
    historical_best = {key: max(values) for key, values in rolling.items()}
    retention = {
        key: current[key] / max(historical_best[key], 1e-8)
        for key in keys
    }
    return {
        "baseline_first_step": int(baseline[0]["global_step"]),
        "baseline_last_step": int(baseline[-1]["global_step"]),
        "rolling_windows": len(next(iter(rolling.values()))),
        "current_median": current,
        "historical_best_median": historical_best,
        "retention": retention,
    }


def health_report(records: list[dict], args) -> dict:
    selected = [
        record
        for record in records
        if all(key in record for key in HEALTH_KEYS)
    ]
    tail = selected[-args.window :]
    alerts = []
    if records and not selected:
        alerts.append("health_metric_schema_mismatch")
    nonfinite = sorted(
        {
            key
            for record in tail
            for key in HEALTH_KEYS
            if not math.isfinite(float(record[key]))
        }
    )
    if nonfinite:
        alerts.append(f"nonfinite_metrics:{','.join(nonfinite)}")

    medians = {
        key: statistics.median(float(record[key]) for record in tail)
        for key in HEALTH_KEYS
    } if tail else {}
    maxima = {
        "grad_norm": max(float(record["grad_norm"]) for record in tail)
    } if tail else {}
    last_step = int(selected[-1]["global_step"]) if selected else 0
    trend = trend_report(selected, args)
    if len(tail) >= args.minimum_records:
        if maxima["grad_norm"] >= args.maximum_grad_norm:
            alerts.append("gradient_explosion")
        if (
            medians["action_specificity_sample_std"]
            < args.minimum_action_std
        ):
            alerts.append("posterior_action_variance_collapse")
        if (
            medians["action_specificity_slot_rms"]
            < args.minimum_action_slot_rms
        ):
            alerts.append("dynamics_action_response_collapse")
        if last_step >= args.collapse_check_start:
            if (
                medians["rgb_future_delta_change_energy_ratio"]
                < args.minimum_change_energy_ratio
            ):
                alerts.append("observed_change_energy_collapse")
            if (
                medians["rgb_future_delta_predicted_change_rms"]
                < args.minimum_predicted_change_rms
            ):
                alerts.append("observed_change_magnitude_collapse")
            if (
                medians["rgb_future_delta_change_gain_over_copy"]
                <= args.minimum_change_gain
            ):
                alerts.append("no_gain_over_current_copy")
        if last_step >= args.trend_check_start and trend:
            retention = trend["retention"]
            if (
                retention["rgb_future_delta_change_gain_over_copy"]
                < args.minimum_trend_retention
            ):
                alerts.append("change_gain_trend_collapse")
            if (
                retention["rgb_future_delta_change_energy_ratio"]
                < args.minimum_trend_retention
            ):
                alerts.append("change_energy_trend_collapse")
            if (
                retention["action_specificity_slot_rms"]
                < args.minimum_trend_retention
            ):
                alerts.append("action_response_trend_collapse")
    return {
        "status": "alert" if alerts else "healthy",
        "raw_records": len(records),
        "records": len(selected),
        "ignored_records": len(records) - len(selected),
        "window_records": len(tail),
        "first_step": int(tail[0]["global_step"]) if tail else 0,
        "last_step": last_step,
        "medians": medians,
        "maxima": maxima,
        "trend": trend,
        "alerts": alerts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log", required=True)
    parser.add_argument("--active_pid_file", required=True)
    parser.add_argument("--supervisor_pid_file", required=True)
    parser.add_argument("--final_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--minimum_records", type=int, default=5)
    parser.add_argument("--collapse_check_start", type=int, default=500)
    parser.add_argument("--maximum_grad_norm", type=float, default=100.0)
    parser.add_argument("--minimum_action_std", type=float, default=0.05)
    parser.add_argument(
        "--minimum_action_slot_rms",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--minimum_change_energy_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--minimum_predicted_change_rms",
        type=float,
        default=0.005,
    )
    parser.add_argument("--minimum_change_gain", type=float, default=0.03)
    parser.add_argument("--trend_check_start", type=int, default=2000)
    parser.add_argument("--minimum_trend_retention", type=float, default=0.70)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if min(args.poll_seconds, args.window, args.minimum_records) < 1:
        raise ValueError("polling and window values must be positive")
    if args.minimum_records > args.window:
        raise ValueError("minimum_records cannot exceed the health window")
    if args.trend_check_start < args.collapse_check_start:
        raise ValueError("trend checks must start after collapse checks")
    if not 0.0 < args.minimum_trend_retention <= 1.0:
        raise ValueError("trend retention must be in (0,1]")

    paths = (
        args.train_log,
        args.active_pid_file,
        args.supervisor_pid_file,
        args.final_checkpoint,
        args.output,
    )
    if any(not os.path.isabs(path) for path in paths):
        raise ValueError("all health-monitor paths must be absolute")

    while True:
        report = health_report(load_complete_records(args.train_log), args)
        active_pid = read_pid(args.active_pid_file)
        supervisor_pid = read_pid(args.supervisor_pid_file)
        report["active_pid"] = active_pid
        report["active_process_alive"] = pid_alive(active_pid)
        report["supervisor_pid"] = supervisor_pid
        report["supervisor_alive"] = pid_alive(supervisor_pid)
        report["final_checkpoint_present"] = os.path.isfile(
            args.final_checkpoint
        )
        report["updated_at_unix"] = time.time()
        if report["final_checkpoint_present"]:
            report["run_state"] = "complete"
        elif report["supervisor_alive"]:
            report["run_state"] = "running_or_recovering"
        else:
            report["run_state"] = "stopped_before_completion"
            report["status"] = "alert"
            report["alerts"].append("supervisor_stopped_before_completion")
        write_report(os.path.abspath(args.output), report)
        if args.once or report["run_state"] in (
            "complete",
            "stopped_before_completion",
        ):
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
