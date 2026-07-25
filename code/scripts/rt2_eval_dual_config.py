"""Run demo_clean and demo_randomized RoboTwin evaluations on disjoint GPU sets."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
import rt2_eval_parallel as parallel

SINGLE_LAUNCHER = os.path.join(SCRIPT_DIR, "rt2_eval_causal_4gpu.sh")
DUAL_REPORT = os.path.join(SCRIPT_DIR, "rt2_eval_dual_report.py")
CONFIGS = ("demo_clean", "demo_randomized")


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def gpu_list(value):
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU list must contain unique indices: {value!r}")
    return gpus


def terminate_groups(processes):
    for process in processes:
        if process.poll() is None:
            subprocess.run(["/bin/kill", "-TERM", "--", f"-{process.pid}"], check=False)
    deadline = time.time() + 30
    while time.time() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.5)
    for process in processes:
        if process.poll() is None:
            subprocess.run(["/bin/kill", "-KILL", "--", f"-{process.pid}"], check=False)
    for process in processes:
        process.wait()


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def wait_for_gpus(gpus, max_used_mib, wait_seconds, poll_seconds, status_path):
    deadline = time.time() + wait_seconds
    while True:
        memory = parallel.gpu_memory_used()
        missing = [gpu for gpu in gpus if gpu not in memory]
        if missing:
            raise ValueError(f"GPU indices are not visible to nvidia-smi: {missing}")
        busy = {gpu: memory[gpu] for gpu in gpus if memory[gpu] > max_used_mib}
        status = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "state": "waiting_for_gpus" if busy else "gpus_ready",
            "gpu_memory_mib": {str(gpu): memory[gpu] for gpu in gpus},
            "max_initial_gpu_memory_mib": max_used_mib,
        }
        write_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if not busy:
            return memory
        if wait_seconds <= 0 or time.time() >= deadline:
            raise RuntimeError(f"GPUs did not become available within {wait_seconds}s: {busy}")
        time.sleep(poll_seconds)


def progress_row(spec):
    summary_path = os.path.join(spec["output"], "summary.json")
    if not os.path.isfile(summary_path):
        return {"config": spec["config"], "completed": 0, "target": None, "success": 0}
    summary = read_json(summary_path)
    return {
        "config": spec["config"],
        "completed": summary.get("completed", 0),
        "target": summary.get("target"),
        "success": summary.get("success", 0),
        "errors": len(summary.get("errors", [])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--seed-tries", type=int, default=100)
    parser.add_argument("--instruction-type", choices=["seen", "unseen"], default="seen")
    parser.add_argument("--clean-gpus", default="0,1")
    parser.add_argument("--randomized-gpus", default="2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--clean-base-port", type=int, default=19010)
    parser.add_argument("--randomized-base-port", type=int, default=19020)
    parser.add_argument("--robotwin-root", default="/mnt/pfs/xuhaoming/xr-2/RoboTwin")
    parser.add_argument("--robotwin-python", default="/mnt/pfs/xuhaoming/xr-2/.venv/bin/python")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-initial-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--wait-for-gpus-seconds", type=int, default=0)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    if args.episodes <= 0 or args.workers_per_gpu <= 0:
        raise ValueError("episodes and workers-per-gpu must be positive")
    if args.max_initial_gpu_memory_mib < 0 or args.wait_for_gpus_seconds < 0 \
            or args.gpu_poll_seconds <= 0:
        raise ValueError("GPU memory limit and wait settings must be non-negative")
    clean_gpus = gpu_list(args.clean_gpus)
    randomized_gpus = gpu_list(args.randomized_gpus)
    overlap = sorted(set(clean_gpus) & set(randomized_gpus))
    if overlap:
        raise ValueError(f"clean and randomized GPU sets overlap: {overlap}")
    clean_ports = set(range(args.clean_base_port, args.clean_base_port + len(clean_gpus)))
    randomized_ports = set(
        range(args.randomized_base_port, args.randomized_base_port + len(randomized_gpus)))
    if clean_ports & randomized_ports:
        raise ValueError(f"clean and randomized policy-server ports overlap: {clean_ports & randomized_ports}")
    for cfg in CONFIGS:
        path = os.path.join(args.robotwin_root, "task_config", cfg + ".yml")
        if not os.path.isfile(path):
            raise ValueError(f"RoboTwin task config does not exist: {os.path.abspath(path)}")

    expected_tasks = len([task for task in args.tasks.split(",") if task.strip()]) \
        if args.tasks else args.expected_tasks
    specs = [
        {
            "config": "demo_clean",
            "gpus": clean_gpus,
            "base_port": args.clean_base_port,
            "output": os.path.join(out, "demo_clean"),
        },
        {
            "config": "demo_randomized",
            "gpus": randomized_gpus,
            "base_port": args.randomized_base_port,
            "output": os.path.join(out, "demo_randomized"),
        },
    ]
    os.makedirs(out, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    processes = []
    run_config = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output": out,
        "checkpoint": checkpoint,
        "episodes": args.episodes,
        "scope": "quick" if args.episodes == 1 else "formal" if args.episodes == 5 else "custom",
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "expected_tasks": expected_tasks,
        "instruction_type": args.instruction_type,
        "workers_per_gpu": args.workers_per_gpu,
        "max_initial_gpu_memory_mib": args.max_initial_gpu_memory_mib,
        "wait_for_gpus_seconds": args.wait_for_gpus_seconds,
        "robotwin_root": os.path.abspath(args.robotwin_root),
        "robotwin_python": os.path.abspath(args.robotwin_python),
        "preflight_only": args.preflight_only,
        "configurations": specs,
    }
    write_json(os.path.join(out, "dual_run_config.json"), run_config)
    wait_for_gpus(
        clean_gpus + randomized_gpus,
        args.max_initial_gpu_memory_mib,
        args.wait_for_gpus_seconds,
        args.gpu_poll_seconds,
        os.path.join(out, "gpu_wait_status.json"),
    )

    try:
        for spec in specs:
            os.makedirs(spec["output"], exist_ok=True)
            log_path = os.path.join(out, spec["config"] + ".launcher.log")
            env = os.environ.copy()
            env.update({
                "CKPT": checkpoint,
                "OUT": spec["output"],
                "CFG": spec["config"],
                "EPISODES": str(args.episodes),
                "TASKS": args.tasks,
                "EXPECTED_TASKS": str(expected_tasks),
                "START_SEED": str(args.start_seed),
                "SEED_TRIES": str(args.seed_tries),
                "INSTRUCTION_TYPE": args.instruction_type,
                "ROBOTWIN_PYTHON": os.path.abspath(args.robotwin_python),
                "ROBOTWIN_ROOT": os.path.abspath(args.robotwin_root),
                "GPUS": ",".join(map(str, spec["gpus"])),
                "WORKERS_PER_GPU": str(args.workers_per_gpu),
                "BASE_PORT": str(spec["base_port"]),
                "MAX_INITIAL_GPU_MEMORY_MIB": str(args.max_initial_gpu_memory_mib),
                "PREFLIGHT_ONLY": "1" if args.preflight_only else "0",
            })
            log = open(log_path, "a", encoding="utf-8")
            process = subprocess.Popen(
                [SINGLE_LAUNCHER], cwd=PROJECT_ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            log.close()
            processes.append(process)
            spec["pid"] = process.pid
            spec["log"] = log_path
        write_json(os.path.join(out, "dual_run_config.json"), run_config)

        while any(process.poll() is None for process in processes):
            failed = [
                {"config": spec["config"], "returncode": process.returncode}
                for spec, process in zip(specs, processes)
                if process.poll() is not None and process.returncode != 0
            ]
            status = {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "runs": [progress_row(spec) for spec in specs],
                "failed": failed,
            }
            write_json(os.path.join(out, "dual_status.json"), status)
            print(json.dumps(status, sort_keys=True), flush=True)
            if failed:
                terminate_groups(processes)
                raise SystemExit(1)
            time.sleep(args.poll_seconds)

        returncodes = {
            spec["config"]: process.returncode for spec, process in zip(specs, processes)
        }
        if any(code != 0 for code in returncodes.values()):
            raise SystemExit(f"dual evaluation failed: {returncodes}")
        if args.preflight_only:
            payload = {
                "status": "ok",
                "output": out,
                "checkpoint": checkpoint,
                "runs": {
                    spec["config"]: read_json(os.path.join(spec["output"], "preflight.json"))
                    for spec in specs
                },
            }
            write_json(os.path.join(out, "dual_preflight.json"), payload)
            print(json.dumps(payload, sort_keys=True))
            return
        subprocess.run(
            [sys.executable, DUAL_REPORT, "--out", out, "--require-complete"],
            cwd=PROJECT_ROOT, check=True,
        )
    finally:
        terminate_groups(processes)


if __name__ == "__main__":
    main()
