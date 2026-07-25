"""Resumable RoboTwin evaluation for the Instruct-GS-World VLA policy.

Runs one policy server against every task in the Instruct-GS-World RoboTwin instruction overlay. Each
episode is a separate rollout-client process, keeps its video, and writes result.json. Seeds follow the
official evaluator: start at 100000, skip unstable or expert-unsolvable scenes, then advance from the
actual accepted seed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request


DEFAULT_CLIENT = "/mnt/pfs/public/xuhaoming/instruct_gs_world/code/scripts/rt2_rollout_client.py"
DEFAULT_PYTHON = "/mnt/pfs/xuhaoming/xr-2/.venv/bin/python"
DEFAULT_ROBOTWIN_ROOT = "/mnt/pfs/xuhaoming/xr-2/RoboTwin"
TASK_ROOT = "/mnt/pfs/public/xuhaoming/Cosmos-3-Finetune/data/robotwin2_instructions_overlay"
EXPECTED_ENVIRONMENT = {
    "numpy": "1.26.4",
    "scipy": "1.10.1",
    "sapien": "3.0.0b1",
    "mplib": "0.2.1",
    "torch": "2.7.1+cu126",
    "pytorch3d": "0.7.9",
    "warp-lang": "1.12.1",
    "curobo_available": True,
}
SERVER_CONTRACT = {
    "status": "ok",
    "wrist": 1,
    "placement": "entropy",
    "L": 512,
    "action_steps": 50,
    "obs_preprocess": "vggt_t1_grid48",
    "causal_geometry_version": "vggt_t1_grid48_v1",
    "policy_seed_mode": "request_v1",
}
INSTRUCTION_SEED_MODE = "environment_seed_instruction_v1"


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def server_health(server):
    with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=10) as response:
        return json.loads(response.read())


def checkpoint_step(checkpoint):
    name = os.path.basename(os.path.abspath(checkpoint))
    match = re.fullmatch(r"vla_(\d+)\.pt", name)
    if match is None:
        raise ValueError(f"checkpoint must be named vla_<step>.pt, got {name!r}")
    return int(match.group(1))


def expected_server_health(checkpoint):
    return {
        **SERVER_CONTRACT,
        "checkpoint": os.path.abspath(checkpoint),
        "step": checkpoint_step(checkpoint),
    }


def instruction_seed(task, cfg, environment_seed, episode_index, instruction_type, description_count):
    identity = (f"{INSTRUCTION_SEED_MODE}|{task}|{cfg}|{int(environment_seed)}|"
                f"{int(episode_index)}|{instruction_type}|{int(description_count)}").encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def validate_server_health(health, checkpoint):
    expected = expected_server_health(checkpoint)
    mismatch = {key: {"server": health.get(key), "expected": value}
                for key, value in expected.items() if health.get(key) != value}
    if mismatch:
        raise ValueError(f"wrong policy server: {mismatch}")
    return expected


def discover_tasks(task_csv, robotwin_root):
    if task_csv:
        tasks = [task.strip() for task in task_csv.split(",") if task.strip()]
    else:
        tasks = sorted(name for name in os.listdir(TASK_ROOT)
                       if os.path.isdir(os.path.join(TASK_ROOT, name)))
    missing = [task for task in tasks
               if not os.path.isfile(os.path.join(robotwin_root, "envs", task + ".py"))]
    if missing:
        raise ValueError(f"tasks missing from RoboTwin envs: {missing}")
    return tasks


def validate_task_config(robotwin_root, cfg):
    path = os.path.abspath(os.path.join(robotwin_root, "task_config", cfg + ".yml"))
    if not os.path.isfile(path):
        raise ValueError(f"RoboTwin task config does not exist: {path}")
    return path


def probe_environment(python, robotwin_root, planner_backend):
    probe = """
import importlib.metadata
import json
import mplib
import numpy
import pytorch3d
import sapien
import scipy
import torch
from envs.robot import planner

payload = {
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "sapien": sapien.__version__,
    "mplib": mplib.__version__,
    "torch": torch.__version__,
    "pytorch3d": pytorch3d.__version__,
    "warp-lang": importlib.metadata.version("warp-lang"),
    "curobo_available": planner.load_curobo_planner(),
}
print(json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    env["ROBOTWIN_ROOT"] = robotwin_root
    env["ROBOTWIN_PLANNER_BACKEND"] = planner_backend
    proc = subprocess.run([python, "-c", probe], cwd=robotwin_root, env=env,
                          text=True, stdout=subprocess.PIPE, check=True)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    mismatch = {key: {"environment": result.get(key), "expected": value}
                for key, value in EXPECTED_ENVIRONMENT.items() if result.get(key) != value}
    if mismatch:
        raise ValueError(f"wrong RoboTwin environment: {mismatch}")
    return result


def valid_result(path, task, cfg, trial, runtime):
    if not os.path.isfile(path):
        return None
    result = read_json(path)
    expected = {
        "task": task,
        "cfg": cfg,
        "episode_index": trial,
        "wrist": True,
        "exec_horizon": 50,
        "smooth": True,
        "policy_seed_mode": "environment_seed_replan_v1",
        "instruction_seed_mode": INSTRUCTION_SEED_MODE,
        **runtime,
    }
    mismatches = {key: {"result": result.get(key), "expected": value}
                  for key, value in expected.items() if result.get(key) != value}
    if mismatches:
        raise ValueError(f"invalid existing result {path}: {mismatches}")
    video = result.get("video_path", "")
    if not os.path.isfile(video) or os.path.getsize(video) == 0:
        raise ValueError(f"missing video for existing result {path}: {video}")
    policy_seeds = result.get("policy_seeds")
    if not isinstance(policy_seeds, list) or len(policy_seeds) != int(result.get("replans", -1)):
        raise ValueError(f"invalid policy seed trace in existing result {path}")
    if len(policy_seeds) != len(set(policy_seeds)):
        raise ValueError(f"duplicate policy seeds in existing result {path}")
    expected_instruction_seed = instruction_seed(
        task, cfg, result["seed"], trial, runtime["instruction_type"], runtime["description_count"])
    if result.get("instruction_seed") != expected_instruction_seed:
        raise ValueError(
            f"invalid instruction seed in existing result {path}: "
            f"{result.get('instruction_seed')} != {expected_instruction_seed}"
        )
    planner = result.get("planner_runtime")
    if not isinstance(planner, dict) or planner.get("backend") != "curobo" \
            or planner.get("curobo_active") is not True:
        raise ValueError(f"CuRobo was not active in existing result {path}: {planner}")
    return result


def collect_results(out, tasks, cfg, episodes, runtime):
    results = []
    for task in tasks:
        for trial in range(episodes):
            path = os.path.join(out, task, f"trial_{trial:02d}", "result.json")
            result = valid_result(path, task, cfg, trial, runtime)
            if result is not None:
                results.append(result)
    return results


def metadata_path(out, stem, extension, metadata_tag):
    suffix = f".{metadata_tag}" if metadata_tag else ""
    return os.path.join(out, f"{stem}{suffix}.{extension}")


def write_summary(out, tasks, cfg, episodes, errors, runtime, metadata_tag=""):
    results = collect_results(out, tasks, cfg, episodes, runtime)
    by_task = {}
    for task in tasks:
        rows = [row for row in results if row["task"] == task]
        success = sum(bool(row["success"]) for row in rows)
        by_task[task] = {
            "completed": len(rows), "target": episodes, "success": success,
            "success_rate": success / len(rows) if rows else None,
        }
    total_success = sum(bool(row["success"]) for row in results)
    summary = {
        "cfg": cfg, "tasks": len(tasks), "episodes_per_task": episodes,
        "completed": len(results), "target": len(tasks) * episodes, "success": total_success,
        "success_rate": total_success / len(results) if results else None,
        "errors": errors, "by_task": by_task,
    }
    write_json(metadata_path(out, "summary", "json", metadata_tag), summary)
    tsv = metadata_path(out, "summary", "tsv", metadata_tag)
    with open(tsv + ".tmp", "w") as f:
        f.write("task\tcompleted\ttarget\tsuccess\tsuccess_rate\n")
        for task in tasks:
            row = by_task[task]
            rate = "" if row["success_rate"] is None else f"{row['success_rate']:.6f}"
            f.write(f"{task}\t{row['completed']}\t{row['target']}\t{row['success']}\t{rate}\n")
    os.replace(tsv + ".tmp", tsv)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:19010")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cfg", default="demo_clean")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--tasks", default="", help="comma list; empty discovers all 50 overlay tasks")
    ap.add_argument("--expected_tasks", type=int, default=50)
    ap.add_argument("--start_seed", type=int, default=100000)
    ap.add_argument("--seed_tries", type=int, default=100)
    ap.add_argument("--instruction_type", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--client", default=DEFAULT_CLIENT)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--robotwin-root", default=DEFAULT_ROBOTWIN_ROOT)
    ap.add_argument("--planner-backend", choices=["curobo"], default="curobo")
    ap.add_argument("--metadata-tag", default="",
                    help="suffix for run_config/summary files when workers share one result root")
    a = ap.parse_args()

    if a.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if a.metadata_tag and not all(c.isalnum() or c in "_-" for c in a.metadata_tag):
        raise ValueError("--metadata-tag must contain only letters, digits, underscore, or hyphen")
    a.robotwin_root = os.path.abspath(a.robotwin_root)
    task_config = validate_task_config(a.robotwin_root, a.cfg)
    tasks = discover_tasks(a.tasks, a.robotwin_root)
    if not a.tasks and len(tasks) != a.expected_tasks:
        raise ValueError(f"discovered {len(tasks)} tasks, expected {a.expected_tasks}")
    environment = probe_environment(a.python, a.robotwin_root, a.planner_backend)
    runtime = {
        "robotwin_root": a.robotwin_root,
        "planner_backend": a.planner_backend,
        "rollout_python": os.path.abspath(a.python),
        "instruction_type": a.instruction_type,
        "description_count": a.episodes,
    }
    health = server_health(a.server)
    validate_server_health(health, a.checkpoint)

    os.makedirs(a.out, exist_ok=True)
    run_config = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "output": os.path.abspath(a.out),
        "server": a.server, "server_health": health, "checkpoint": os.path.abspath(a.checkpoint),
        "cfg": a.cfg, "episodes": a.episodes, "tasks": tasks, "start_seed": a.start_seed,
        "seed_tries": a.seed_tries, "instruction_type": a.instruction_type,
        "task_config": task_config,
        "client": os.path.abspath(a.client), "python": a.python,
        "robotwin_root": a.robotwin_root, "planner_backend": a.planner_backend,
        "environment": environment, "checkpoint_step": checkpoint_step(a.checkpoint),
    }
    run_config["metadata_tag"] = a.metadata_tag
    write_json(metadata_path(a.out, "run_config", "json", a.metadata_tag), run_config)

    errors = []
    write_summary(a.out, tasks, a.cfg, a.episodes, errors, runtime, a.metadata_tag)
    for task_index, task in enumerate(tasks):
        completed = []
        for trial in range(a.episodes):
            path = os.path.join(a.out, task, f"trial_{trial:02d}", "result.json")
            result = valid_result(path, task, a.cfg, trial, runtime)
            if result is None:
                break
            completed.append(result)
        if len(completed) == a.episodes:
            print(f"[batch] skip complete task {task} ({len(completed)}/{a.episodes})", flush=True)
            continue
        if any(os.path.isfile(os.path.join(a.out, task, f"trial_{trial:02d}", "result.json"))
               for trial in range(len(completed) + 1, a.episodes)):
            raise ValueError(f"non-contiguous completed trials for {task}")

        seed = int(completed[-1]["seed"]) + 1 if completed else a.start_seed
        for trial in range(len(completed), a.episodes):
            trial_out = os.path.join(a.out, task, f"trial_{trial:02d}")
            os.makedirs(trial_out, exist_ok=True)
            cmd = [
                a.python, a.client, "--task", task, "--cfg", a.cfg, "--seed", str(seed),
                "--seed_tries", str(a.seed_tries), "--episode_index", str(trial),
                "--instruction_type", a.instruction_type, "--description_count", str(a.episodes),
                "--server", a.server, "--out", trial_out,
                "--exec_horizon", "50", "--smooth", "1", "--wrist", "1",
            ]
            rollout_env = os.environ.copy()
            rollout_env["ROBOTWIN_ROOT"] = a.robotwin_root
            rollout_env["ROBOTWIN_PLANNER_BACKEND"] = a.planner_backend
            print(f"[batch] task={task_index + 1}/{len(tasks)} {task} trial={trial + 1}/{a.episodes} "
                  f"seed_start={seed}", flush=True)
            log_path = os.path.join(trial_out, "rollout.log")
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, cwd=a.robotwin_root, env=rollout_env, stdout=log,
                                      stderr=subprocess.STDOUT, text=True, check=False)
            result_path = os.path.join(trial_out, "result.json")
            if proc.returncode != 0 or not os.path.isfile(result_path):
                error = {"task": task, "trial": trial, "seed_start": seed,
                         "returncode": proc.returncode, "log": os.path.abspath(log_path)}
                errors.append(error)
                write_json(os.path.join(trial_out, "error.json"), error)
                print(f"[batch] ERROR {error}", file=sys.stderr, flush=True)
                break
            result = valid_result(result_path, task, a.cfg, trial, runtime)
            seed = int(result["seed"]) + 1
            summary = write_summary(a.out, tasks, a.cfg, a.episodes, errors, runtime, a.metadata_tag)
            print(f"[batch] done success={result['success']} seed={result['seed']} "
                  f"total={summary['completed']}/{summary['target']}", flush=True)

    summary = write_summary(a.out, tasks, a.cfg, a.episodes, errors, runtime, a.metadata_tag)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["completed"] != summary["target"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
