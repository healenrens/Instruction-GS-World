"""Build one report for paired demo_clean and demo_randomized RoboTwin evaluations."""
import argparse
import json
import os
from collections import Counter

import rt2_eval_report as single


CONFIGS = ("demo_clean", "demo_randomized")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rate(success, completed):
    return success / completed if completed else None


def markdown_report(report, task_rows):
    lines = [
        "# RoboTwin Dual-Configuration Evaluation",
        "",
        f"- Scope: `{report['scope']}`",
        f"- Status: `{report['status']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Output: `{report['output']}`",
        f"- Coverage: `{report['completed']}/{report['target']}`",
        f"- Overall success: `{report['success']}/{report['completed']}`",
        f"- Overall success rate: `{report['success_rate']:.6f}`"
        if report["success_rate"] is not None else "- Overall success rate: unavailable",
        "",
        "## Configuration Summary",
        "",
        "| Configuration | Completed | Success | Failure | Success rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for cfg in CONFIGS:
        row = report["by_config"][cfg]
        cfg_rate = "" if row["success_rate"] is None else f"{row['success_rate']:.6f}"
        lines.append(
            f"| {cfg} | {row['completed']} | {row['success']} | {row['failure']} | {cfg_rate} |"
        )
    lines.extend([
        "",
        "## Per-Task Summary",
        "",
        "| Task | Clean | Randomized | Overall |",
        "|---|---:|---:|---:|",
    ])
    for row in task_rows:
        clean = f"{row['demo_clean_success']}/{row['demo_clean_completed']}"
        randomized = f"{row['demo_randomized_success']}/{row['demo_randomized_completed']}"
        overall = f"{row['success']}/{row['completed']}"
        lines.append(f"| {row['task']} | {clean} | {randomized} | {overall} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    dual_config = read_json(os.path.join(out, "dual_run_config.json"))
    checkpoint = os.path.abspath(dual_config["checkpoint"])
    episodes = int(dual_config["episodes"])
    expected_tasks = None
    rows = []
    missing = []
    violations = []
    child_artifacts = {}

    for cfg in CONFIGS:
        child_out = os.path.join(out, cfg)
        child_config = read_json(os.path.join(child_out, "parallel_run_config.json"))
        verification = read_json(os.path.join(child_out, "verification.json"))
        child_report = read_json(os.path.join(child_out, "final_report", "report.json"))
        child_artifacts[cfg] = {
            "output": child_out,
            "verification": os.path.join(child_out, "verification.json"),
            "report": os.path.join(child_out, "final_report", "report.json"),
        }
        if child_config.get("cfg") != cfg:
            violations.append(f"{cfg}: child cfg is {child_config.get('cfg')!r}")
        if os.path.abspath(child_config.get("checkpoint", "")) != checkpoint:
            violations.append(f"{cfg}: checkpoint mismatch")
        if int(child_config.get("episodes", -1)) != episodes:
            violations.append(f"{cfg}: episode count mismatch")
        if verification.get("status") != "complete" or verification.get("violations"):
            violations.append(f"{cfg}: verification is not clean")
        if child_report.get("status") != "complete":
            violations.append(f"{cfg}: final report is not complete")

        tasks = child_config["tasks"]
        if expected_tasks is None:
            expected_tasks = tasks
        elif tasks != expected_tasks:
            violations.append(f"{cfg}: task list differs from {CONFIGS[0]}")
        for task in tasks:
            for trial in range(episodes):
                path = os.path.join(child_out, task, f"trial_{trial:02d}", "result.json")
                if not os.path.isfile(path):
                    missing.append({
                        "config": cfg, "task": task, "trial": trial,
                        "result_path": os.path.abspath(path),
                    })
                    continue
                row = single.result_row(path, read_json(path))
                row["config"] = cfg
                rows.append(row)

    tasks = expected_tasks or []
    rows.sort(key=lambda row: (row["task"], row["config"], row["trial"]))
    bad_cases = [row for row in rows if not row["success"]]
    by_config = {}
    for cfg in CONFIGS:
        cfg_rows = [row for row in rows if row["config"] == cfg]
        success = sum(row["success"] for row in cfg_rows)
        by_config[cfg] = {
            "completed": len(cfg_rows),
            "target": len(tasks) * episodes,
            "success": success,
            "failure": len(cfg_rows) - success,
            "success_rate": rate(success, len(cfg_rows)),
        }

    task_rows = []
    success_distribution = Counter()
    for task in tasks:
        task_result = {"task": task}
        task_all = []
        for cfg in CONFIGS:
            cfg_rows = [row for row in rows if row["task"] == task and row["config"] == cfg]
            cfg_success = sum(row["success"] for row in cfg_rows)
            task_result[f"{cfg}_completed"] = len(cfg_rows)
            task_result[f"{cfg}_success"] = cfg_success
            task_result[f"{cfg}_success_rate"] = rate(cfg_success, len(cfg_rows))
            task_all.extend(cfg_rows)
        total_success = sum(row["success"] for row in task_all)
        task_result.update({
            "completed": len(task_all),
            "target": episodes * len(CONFIGS),
            "success": total_success,
            "failure": len(task_all) - total_success,
            "success_rate": rate(total_success, len(task_all)),
        })
        success_distribution[total_success] += 1
        task_rows.append(task_result)

    report_dir = os.path.join(out, "final_report")
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
        "config", "task", "trial", "success", "requested_seed", "accepted_seed",
        "instruction", "step_count", "replans", "elapsed_sec", "video_frames",
        "video_bytes", "video_path", "result_path",
    ]
    task_fields = [
        "task", "demo_clean_completed", "demo_clean_success", "demo_clean_success_rate",
        "demo_randomized_completed", "demo_randomized_success", "demo_randomized_success_rate",
        "completed", "target", "success", "failure", "success_rate",
    ]
    single.write_tsv(paths["rollouts_tsv"], rows, rollout_fields)
    single.write_tsv(paths["task_summary_tsv"], task_rows, task_fields)
    single.write_tsv(paths["bad_cases_tsv"], bad_cases, rollout_fields)
    single.write_json(paths["bad_cases_json"], {"count": len(bad_cases), "records": bad_cases})

    target = len(tasks) * episodes * len(CONFIGS)
    success = sum(row["success"] for row in rows)
    status = "invalid" if violations else "incomplete" if missing or len(rows) != target else "complete"
    report = {
        "scope": "quick" if episodes == 1 else "formal" if episodes == 5 else "custom",
        "status": status,
        "output": out,
        "checkpoint": checkpoint,
        "configs": list(CONFIGS),
        "tasks": len(tasks),
        "episodes_per_task_per_config": episodes,
        "completed": len(rows),
        "target": target,
        "success": success,
        "failure": len(rows) - success,
        "success_rate": rate(success, len(rows)),
        "by_config": by_config,
        "missing": missing,
        "violations": violations,
        "task_success_count_distribution": {
            str(value): success_distribution[value] for value in sorted(success_distribution)
        },
        "child_artifacts": child_artifacts,
        "artifacts": paths,
    }
    single.write_json(paths["report_json"], report)
    with open(paths["report_md"] + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(markdown_report(report, task_rows))
    os.replace(paths["report_md"] + ".tmp", paths["report_md"])
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_complete and status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
