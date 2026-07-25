"""Verify RoboTwin evaluation coverage, runtime identity, results, and retained videos."""
import argparse
import glob
import json
import os
import subprocess

import rt2_eval_batch as batch


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def result_violations(result, path, task, cfg, trial, runtime, server_urls, ffprobe):
    expected = {
        "task": task, "cfg": cfg, "episode_index": trial,
        "exec_horizon": 50, "smooth": True, "wrist": True,
        "policy_seed_mode": "environment_seed_replan_v1",
        "instruction_seed_mode": batch.INSTRUCTION_SEED_MODE, **runtime,
    }
    violations = [f"{path}: {key}={result.get(key)!r}, expected {value!r}"
                  for key, value in expected.items() if result.get(key) != value]
    if not result.get("instruction"):
        violations.append(f"{path}: empty instruction")
    if result.get("server") not in server_urls:
        violations.append(f"{path}: unknown server {result.get('server')!r}")
    video = result.get("video_path", "")
    expected_video = os.path.join(os.path.dirname(path), "rollout.mp4")
    if video != expected_video:
        violations.append(f"{path}: video_path={video!r}, expected {expected_video!r}")
        return violations, None
    if not os.path.isfile(video):
        violations.append(f"{path}: missing video {video}")
        return violations, None
    actual_bytes = os.path.getsize(video)
    if actual_bytes != result.get("video_bytes"):
        violations.append(f"{path}: video bytes={actual_bytes}, metadata={result.get('video_bytes')}")
    if result.get("video_frames", 0) <= 0:
        violations.append(f"{path}: video_frames must be positive")
    policy_seeds = result.get("policy_seeds")
    if not isinstance(policy_seeds, list) or len(policy_seeds) != int(result.get("replans", -1)):
        violations.append(f"{path}: policy seed trace does not match replans")
    elif len(policy_seeds) != len(set(policy_seeds)):
        violations.append(f"{path}: duplicate policy seeds")
    if isinstance(result.get("seed"), int):
        expected_instruction_seed = batch.instruction_seed(
            task, cfg, result["seed"], trial, runtime["instruction_type"],
            runtime["description_count"])
        if result.get("instruction_seed") != expected_instruction_seed:
            violations.append(
                f"{path}: instruction_seed={result.get('instruction_seed')!r}, "
                f"expected {expected_instruction_seed!r}"
            )
    else:
        violations.append(f"{path}: invalid accepted environment seed {result.get('seed')!r}")
    planner = result.get("planner_runtime")
    if not isinstance(planner, dict) or planner.get("backend") != "curobo" \
            or planner.get("curobo_active") is not True:
        violations.append(f"{path}: CuRobo planner is not active: {planner}")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        violations.append(f"{path}: ffprobe failed rc={probe.returncode}: {probe.stderr.strip()}")
        return violations, None
    duration = float(probe.stdout.strip())
    if duration <= 0:
        violations.append(f"{path}: non-positive video duration {duration}")
    return violations, duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-tasks", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    checkpoint = os.path.abspath(a.checkpoint)
    config_path = os.path.join(out, "parallel_run_config.json")
    config = read_json(config_path)
    tasks = config.get("tasks", [])
    violations = []
    if len(tasks) != a.expected_tasks or len(tasks) != len(set(tasks)):
        violations.append(f"parallel config has {len(tasks)} tasks, expected {a.expected_tasks} unique tasks")
    if config.get("checkpoint") != checkpoint:
        violations.append(f"parallel config checkpoint={config.get('checkpoint')!r}, expected {checkpoint!r}")
    expected_task_config = batch.validate_task_config(config.get("robotwin_root", ""), config.get("cfg", ""))
    if config.get("task_config") != expected_task_config:
        violations.append(
            f"parallel config task_config={config.get('task_config')!r}, "
            f"expected {expected_task_config!r}"
        )
    expected_step = batch.checkpoint_step(checkpoint)
    if config.get("checkpoint_step") != expected_step:
        violations.append(
            f"parallel config checkpoint_step={config.get('checkpoint_step')!r}, expected {expected_step}"
        )
    if config.get("environment") != batch.EXPECTED_ENVIRONMENT:
        violations.append(f"wrong environment: {config.get('environment')!r}")

    servers = config.get("servers", [])
    expected_health = batch.expected_server_health(checkpoint)
    expected_server_count = len(config.get("gpus", []))
    if len(servers) != expected_server_count:
        violations.append(
            f"parallel config has {len(servers)} servers, expected {expected_server_count}"
        )
    for row in servers:
        mismatch = {key: {"actual": row.get("health", {}).get(key), "expected": value}
                    for key, value in expected_health.items()
                    if row.get("health", {}).get(key) != value}
        if mismatch:
            violations.append(f"server gpu={row.get('gpu')} mismatch: {mismatch}")
    server_urls = {f"http://127.0.0.1:{row['port']}" for row in servers}

    workers = config.get("workers", [])
    assigned = [task for worker in workers for task in worker.get("tasks", [])]
    if sorted(assigned) != sorted(tasks) or len(assigned) != len(set(assigned)):
        violations.append("worker task assignments do not cover every task exactly once")
    expected_worker_count = expected_server_count * int(config.get("workers_per_gpu", 0))
    if len(workers) != expected_worker_count:
        violations.append(
            f"parallel config has {len(workers)} workers, expected {expected_worker_count}"
        )

    runtime = {
        "robotwin_root": config.get("robotwin_root"),
        "planner_backend": config.get("planner_backend"),
        "rollout_python": config.get("rollout_python"),
        "instruction_type": config.get("instruction_type"),
        "description_count": config.get("description_count"),
    }
    results = {}
    durations = []
    for path in sorted(glob.glob(os.path.join(out, "*", "trial_*", "result.json"))):
        result = read_json(path)
        key = (result.get("task"), result.get("episode_index"))
        if key in results:
            violations.append(f"duplicate result for {key}: {path}")
            continue
        results[key] = result
        row_violations, duration = result_violations(
            result, path, key[0], config.get("cfg"), key[1], runtime, server_urls, a.ffprobe,
        )
        violations.extend(row_violations)
        if duration is not None:
            durations.append(duration)

    expected_pairs = {(task, trial) for task in tasks for trial in range(a.episodes)}
    unexpected = sorted(set(results) - expected_pairs)
    missing = sorted(expected_pairs - set(results))
    if unexpected:
        violations.append(f"unexpected task/trial results: {unexpected}")
    if a.require_complete and missing:
        violations.append(f"missing {len(missing)} task/trial results")

    for task in tasks:
        task_results = [results[(task, trial)] for trial in range(a.episodes)
                        if (task, trial) in results]
        for index, result in enumerate(task_results):
            expected_seed = 100000 if index == 0 else int(task_results[index - 1]["seed"]) + 1
            if result.get("requested_seed") != expected_seed:
                violations.append(
                    f"{task} trial {index}: requested_seed={result.get('requested_seed')}, "
                    f"expected {expected_seed}"
                )
            if int(result.get("seed", -1)) < int(result.get("requested_seed", 0)):
                violations.append(f"{task} trial {index}: accepted seed precedes requested seed")

    unresolved_errors = [path for path in sorted(glob.glob(os.path.join(out, "*", "trial_*", "error.json")))
                         if not os.path.isfile(os.path.join(os.path.dirname(path), "result.json"))]
    if unresolved_errors:
        violations.append(f"unresolved error files: {unresolved_errors}")

    success = sum(bool(result.get("success")) for result in results.values())
    summary_path = os.path.join(out, "summary.json")
    summary = read_json(summary_path)
    computed = {"completed": len(results), "target": len(expected_pairs), "success": success}
    summary_mismatch = {key: {"summary": summary.get(key), "computed": value}
                        for key, value in computed.items() if summary.get(key) != value}
    if a.require_complete and summary_mismatch:
        violations.append(f"global summary mismatch: {summary_mismatch}")

    report = {
        "status": "complete" if not missing and not violations else "incomplete" if not violations else "invalid",
        "output": out, "checkpoint": checkpoint, "expected_tasks": a.expected_tasks,
        "episodes_per_task": a.episodes, "completed": len(results), "target": len(expected_pairs),
        "success": success, "success_rate": success / len(results) if results else None,
        "missing": len(missing), "unexpected": unexpected, "unresolved_errors": unresolved_errors,
        "validated_videos": len(durations), "video_duration_sec": round(sum(durations), 3),
        "summary_mismatch": summary_mismatch, "violations": violations,
    }
    if a.report:
        write_json(os.path.abspath(a.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
