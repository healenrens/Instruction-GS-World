"""Build a task summary and absolute-path bad-case index for a RoboTwin run."""
import argparse
import csv
import json
import os
from collections import Counter


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_tsv(path, rows, fields):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def result_row(path, result):
    video = os.path.abspath(result["video_path"])
    if not os.path.isfile(video) or os.path.getsize(video) <= 0:
        raise ValueError(f"missing or empty video: {video}")
    return {
        "task": result["task"],
        "trial": int(result["episode_index"]),
        "success": bool(result["success"]),
        "requested_seed": int(result["requested_seed"]),
        "accepted_seed": int(result["seed"]),
        "instruction": result["instruction"],
        "step_count": int(result["step_count"]),
        "replans": int(result["replans"]),
        "elapsed_sec": float(result["elapsed_sec"]),
        "video_frames": int(result["video_frames"]),
        "video_bytes": int(result["video_bytes"]),
        "video_path": video,
        "result_path": os.path.abspath(path),
    }


def markdown_report(report, task_rows, paths):
    lines = [
        "# RoboTwin Evaluation Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Output: `{report['output']}`",
        f"- Coverage: `{report['completed']}/{report['target']}`",
        f"- Success: `{report['success']}`",
        f"- Success rate: `{report['success_rate']:.6f}`" if report["success_rate"] is not None
        else "- Success rate: unavailable",
        f"- Retained videos: `{report['videos']}`",
        "",
        "## Artifacts",
        "",
        f"- All rollouts: `{paths['rollouts_tsv']}`",
        f"- Per-task summary: `{paths['task_summary_tsv']}`",
        f"- Bad cases TSV: `{paths['bad_cases_tsv']}`",
        f"- Bad cases JSON: `{paths['bad_cases_json']}`",
        "",
        "## Per-task Results",
        "",
        "| Task | Completed | Success | Failure | Success rate | Representative bad-case video |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in task_rows:
        rate = "" if row["success_rate"] is None else f"{row['success_rate']:.3f}"
        video = row["representative_bad_case_video"] or ""
        lines.append(
            f"| {row['task']} | {row['completed']} | {row['success']} | {row['failure']} | {rate} | "
            f"`{video}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="evaluation output root")
    parser.add_argument("--report-dir", default="", help="default: <out>/final_report")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    report_dir = os.path.abspath(args.report_dir or os.path.join(out, "final_report"))
    config = read_json(os.path.join(out, "parallel_run_config.json"))
    tasks = config["tasks"]
    episodes = int(config["episodes"])
    rows = []
    missing = []
    for task in tasks:
        for trial in range(episodes):
            path = os.path.join(out, task, f"trial_{trial:02d}", "result.json")
            if not os.path.isfile(path):
                missing.append({"task": task, "trial": trial, "result_path": os.path.abspath(path)})
                continue
            rows.append(result_row(path, read_json(path)))
    if args.require_complete and missing:
        raise ValueError(f"evaluation is incomplete: missing {len(missing)} result files")

    rows.sort(key=lambda row: (row["task"], row["trial"]))
    bad_cases = [row for row in rows if not row["success"]]
    task_rows = []
    success_distribution = Counter()
    for task in tasks:
        task_results = [row for row in rows if row["task"] == task]
        successes = sum(row["success"] for row in task_results)
        failures = len(task_results) - successes
        representative = next((row["video_path"] for row in task_results if not row["success"]), "")
        task_rows.append({
            "task": task,
            "completed": len(task_results),
            "target": episodes,
            "success": successes,
            "failure": failures,
            "success_rate": successes / len(task_results) if task_results else None,
            "representative_bad_case_video": representative,
        })
        success_distribution[successes] += 1

    os.makedirs(report_dir, exist_ok=True)
    paths = {
        "rollouts_tsv": os.path.join(report_dir, "rollouts.tsv"),
        "task_summary_tsv": os.path.join(report_dir, "task_summary.tsv"),
        "bad_cases_tsv": os.path.join(report_dir, "bad_cases.tsv"),
        "bad_cases_json": os.path.join(report_dir, "bad_cases.json"),
        "report_json": os.path.join(report_dir, "report.json"),
        "report_md": os.path.join(report_dir, "REPORT.md"),
    }
    rollout_fields = [
        "task", "trial", "success", "requested_seed", "accepted_seed", "instruction",
        "step_count", "replans", "elapsed_sec", "video_frames", "video_bytes", "video_path", "result_path",
    ]
    task_fields = [
        "task", "completed", "target", "success", "failure", "success_rate",
        "representative_bad_case_video",
    ]
    write_tsv(paths["rollouts_tsv"], rows, rollout_fields)
    write_tsv(paths["task_summary_tsv"], task_rows, task_fields)
    write_tsv(paths["bad_cases_tsv"], bad_cases, rollout_fields)
    write_json(paths["bad_cases_json"], {"count": len(bad_cases), "records": bad_cases})

    success = sum(row["success"] for row in rows)
    target = len(tasks) * episodes
    report = {
        "status": "complete" if not missing and len(rows) == target else "incomplete",
        "output": out,
        "report_dir": report_dir,
        "checkpoint": os.path.abspath(config["checkpoint"]),
        "tasks": len(tasks),
        "episodes_per_task": episodes,
        "completed": len(rows),
        "target": target,
        "success": success,
        "failure": len(rows) - success,
        "success_rate": success / len(rows) if rows else None,
        "videos": len(rows),
        "missing": missing,
        "task_success_count_distribution": {
            str(value): success_distribution[value] for value in sorted(success_distribution)
        },
        "artifacts": paths,
    }
    write_json(paths["report_json"], report)
    with open(paths["report_md"] + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(markdown_report(report, task_rows, paths))
    os.replace(paths["report_md"] + ".tmp", paths["report_md"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
