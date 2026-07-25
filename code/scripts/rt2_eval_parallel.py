"""Run RoboTwin evaluation as two task workers per GPU against one policy server per GPU."""
import argparse
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
import rt2_eval_batch as batch


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def port_is_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def gpu_memory_used():
    command = [
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, check=True)
    rows = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        index, used = line.split(",", 1)
        rows[int(index.strip())] = int(used.strip())
    return rows


def wait_for_server(process, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server on port {port} exited with {process.returncode}")
        if port_is_open(port):
            return batch.server_health(f"http://127.0.0.1:{port}")
        time.sleep(2)
    raise RuntimeError(f"policy server on port {port} was not healthy after {timeout}s")


def validate_health(health, checkpoint):
    batch.validate_server_health(health, checkpoint)


def pending_errors(out):
    errors = []
    for path in sorted(glob.glob(os.path.join(out, "*", "trial_*", "error.json"))):
        if not os.path.isfile(os.path.join(os.path.dirname(path), "result.json")):
            errors.append(batch.read_json(path))
    return errors


def write_global_summary(out, tasks, cfg, episodes, runtime):
    return batch.write_summary(out, tasks, cfg, episodes, pending_errors(out), runtime)


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 20
    while time.time() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.5)
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait()


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cfg", default="demo_clean")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--expected-tasks", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=100000)
    ap.add_argument("--seed-tries", type=int, default=100)
    ap.add_argument("--instruction-type", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--python", default=batch.DEFAULT_PYTHON)
    ap.add_argument("--client", default=batch.DEFAULT_CLIENT)
    ap.add_argument("--robotwin-root", default=batch.DEFAULT_ROBOTWIN_ROOT)
    ap.add_argument("--planner-backend", choices=["curobo"], default="curobo")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--base-port", type=int, default=19010)
    ap.add_argument("--server-timeout", type=int, default=600)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--max-initial-gpu-memory-mib", type=int, default=1024)
    ap.add_argument("--preflight-only", action="store_true",
                    help="validate checkpoint, causal policy servers, task set, and RoboTwin environment, "
                         "then stop before launching rollout workers")
    a = ap.parse_args()

    if not os.path.isfile(a.checkpoint):
        raise ValueError(f"checkpoint does not exist: {os.path.abspath(a.checkpoint)}")
    checkpoint_step = batch.checkpoint_step(a.checkpoint)
    gpus = [int(value.strip()) for value in a.gpus.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain unique GPU indices")
    if a.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    initial_gpu_memory = gpu_memory_used()
    missing_gpus = [gpu for gpu in gpus if gpu not in initial_gpu_memory]
    if missing_gpus:
        raise ValueError(f"GPU indices are not visible to nvidia-smi: {missing_gpus}")
    busy_gpus = {
        gpu: initial_gpu_memory[gpu] for gpu in gpus
        if initial_gpu_memory[gpu] > a.max_initial_gpu_memory_mib
    }
    if busy_gpus:
        raise ValueError(
            f"evaluation GPUs exceed initial memory limit "
            f"{a.max_initial_gpu_memory_mib} MiB: {busy_gpus}"
        )
    a.robotwin_root = os.path.abspath(a.robotwin_root)
    task_config = batch.validate_task_config(a.robotwin_root, a.cfg)
    tasks = batch.discover_tasks(a.tasks, a.robotwin_root)
    if not a.tasks and len(tasks) != a.expected_tasks:
        raise ValueError(f"discovered {len(tasks)} tasks, expected {a.expected_tasks}")
    worker_count = len(gpus) * a.workers_per_gpu
    if worker_count > len(tasks):
        raise ValueError(f"{worker_count} workers exceed {len(tasks)} tasks")
    partitions = [tasks[index::worker_count] for index in range(worker_count)]
    ports = [a.base_port + index for index in range(len(gpus))]
    busy_ports = [port for port in ports if port_is_open(port)]
    if busy_ports:
        raise ValueError(f"policy server ports already in use: {busy_ports}")

    os.makedirs(os.path.join(a.out, "workers"), exist_ok=True)
    environment = batch.probe_environment(a.python, a.robotwin_root, a.planner_backend)
    runtime = {
        "robotwin_root": a.robotwin_root,
        "planner_backend": a.planner_backend,
        "rollout_python": os.path.abspath(a.python),
        "instruction_type": a.instruction_type,
        "description_count": a.episodes,
    }
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    servers = []
    workers = []
    health_rows = []
    try:
        for gpu, port in zip(gpus, ports):
            log_path = os.path.join(a.out, f"server_gpu{gpu}.log")
            log = open(log_path, "a")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable, os.path.join(SCRIPT_DIR, "rt2_policy_server.py"),
                "--ckpt", a.checkpoint, "--port", str(port), "--wrist", "1",
                "--placement", "entropy", "--obs_preprocess", "vggt_t1_grid48",
            ]
            servers.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT))
            log.close()

        for process, gpu, port in zip(servers, gpus, ports):
            health = wait_for_server(process, port, a.server_timeout)
            validate_health(health, a.checkpoint)
            health_rows.append({"gpu": gpu, "port": port, "pid": process.pid, "health": health})

        preflight = {
            "status": "ok",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "output": os.path.abspath(a.out),
            "checkpoint": os.path.abspath(a.checkpoint),
            "checkpoint_step": checkpoint_step,
            "tasks": tasks,
            "cfg": a.cfg,
            "task_config": task_config,
            "environment": environment,
            "robotwin_root": a.robotwin_root,
            "planner_backend": a.planner_backend,
            "rollout_python": os.path.abspath(a.python),
            "gpus": gpus,
            "initial_gpu_memory_mib": {str(gpu): initial_gpu_memory[gpu] for gpu in gpus},
            "max_initial_gpu_memory_mib": a.max_initial_gpu_memory_mib,
            "workers_per_gpu": a.workers_per_gpu,
            "servers": health_rows,
        }
        write_json(os.path.join(a.out, "preflight.json"), preflight)
        if a.preflight_only:
            print(json.dumps(preflight, sort_keys=True), flush=True)
            return

        worker_rows = []
        for worker_index, worker_tasks in enumerate(partitions):
            gpu_slot = worker_index // a.workers_per_gpu
            gpu, port = gpus[gpu_slot], ports[gpu_slot]
            tag = f"worker_{worker_index:02d}"
            log_path = os.path.join(a.out, "workers", tag + ".log")
            command = [
                sys.executable, os.path.join(SCRIPT_DIR, "rt2_eval_batch.py"),
                "--out", a.out, "--server", f"http://127.0.0.1:{port}",
                "--checkpoint", a.checkpoint, "--cfg", a.cfg, "--episodes", str(a.episodes),
                "--tasks", ",".join(worker_tasks), "--start_seed", str(a.start_seed),
                "--seed_tries", str(a.seed_tries), "--instruction_type", a.instruction_type,
                "--python", a.python, "--client", a.client, "--robotwin-root", a.robotwin_root,
                "--planner-backend", a.planner_backend, "--metadata-tag", tag,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(log_path, "a")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            log.close()
            workers.append(process)
            worker_rows.append({"worker": tag, "gpu": gpu, "port": port,
                                "pid": process.pid, "tasks": worker_tasks, "log": log_path})

        config = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "output": os.path.abspath(a.out), "checkpoint": os.path.abspath(a.checkpoint),
            "checkpoint_step": checkpoint_step, "cfg": a.cfg, "episodes": a.episodes,
            "tasks": tasks, "instruction_type": a.instruction_type,
            "description_count": a.episodes, "task_config": task_config,
            "environment": environment,
            "robotwin_root": a.robotwin_root, "planner_backend": a.planner_backend,
            "rollout_python": os.path.abspath(a.python),
            "gpus": gpus, "workers_per_gpu": a.workers_per_gpu,
            "initial_gpu_memory_mib": {str(gpu): initial_gpu_memory[gpu] for gpu in gpus},
            "max_initial_gpu_memory_mib": a.max_initial_gpu_memory_mib,
            "servers": health_rows, "workers": worker_rows,
        }
        write_json(os.path.join(a.out, "parallel_run_config.json"), config)
        summary = write_global_summary(a.out, tasks, a.cfg, a.episodes, runtime)
        while any(process.poll() is None for process in workers):
            dead_servers = [row for row, process in zip(health_rows, servers)
                            if process.poll() is not None]
            if dead_servers:
                raise RuntimeError(f"policy servers exited during evaluation: {dead_servers}")
            summary = write_global_summary(a.out, tasks, a.cfg, a.episodes, runtime)
            print(f"[parallel] completed={summary['completed']}/{summary['target']} "
                  f"success={summary['success']} errors={len(summary['errors'])}", flush=True)
            time.sleep(a.poll_seconds)

        summary = write_global_summary(a.out, tasks, a.cfg, a.episodes, runtime)
        returncodes = {f"worker_{index:02d}": process.returncode
                       for index, process in enumerate(workers)}
        print(json.dumps({"summary": summary, "returncodes": returncodes}, sort_keys=True), flush=True)
        if any(code != 0 for code in returncodes.values()) or summary["completed"] != summary["target"]:
            raise SystemExit(1)
    finally:
        stop_processes(workers)
        stop_processes(servers)


if __name__ == "__main__":
    main()
