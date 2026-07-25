"""Run paired RoboTwin rollouts against two policy servers using identical saved cases."""
import argparse
import json
import os
import subprocess
import urllib.request


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def health(server):
    with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=10) as response:
        return json.loads(response.read())


def parse_cases(case_csv):
    cases = []
    for item in case_csv.split(","):
        task, sep, trial = item.strip().partition(":")
        if not task or sep != ":" or not trial.isdigit():
            raise ValueError(f"invalid case {item!r}; expected task:trial")
        cases.append((task, int(trial)))
    if not cases:
        raise ValueError("at least one case is required")
    return cases


def source_case(source_run, task, trial):
    path = os.path.join(source_run, task, f"trial_{trial:02d}", "result.json")
    row = read_json(path)
    expected = {"task": task, "episode_index": trial}
    mismatch = {key: {"result": row.get(key), "expected": value}
                for key, value in expected.items() if row.get(key) != value}
    if mismatch:
        raise ValueError(f"invalid source result {path}: {mismatch}")
    return path, row


def validate_result(path, source, label, server):
    row = read_json(path)
    expected = {
        "task": source["task"], "cfg": source["cfg"],
        "episode_index": source["episode_index"], "requested_seed": source["seed"],
        "seed": source["seed"], "instruction": source["instruction"],
        "server": server, "wrist": True, "smooth": True, "exec_horizon": 50,
    }
    mismatch = {key: {"result": row.get(key), "expected": value}
                for key, value in expected.items() if row.get(key) != value}
    if mismatch:
        raise ValueError(f"invalid {label} result {path}: {mismatch}")
    video = row.get("video_path", "")
    if not os.path.isfile(video) or os.path.getsize(video) == 0:
        raise ValueError(f"missing {label} video: {video}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--cases", required=True, help="comma list such as adjust_bottle:0,click_bell:8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-a", required=True)
    ap.add_argument("--server-b", required=True)
    ap.add_argument("--label-a", default="raw")
    ap.add_argument("--label-b", default="grid48")
    ap.add_argument("--mode-a", default="raw_random6000")
    ap.add_argument("--mode-b", default="train_grid48_simdepth")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--python", default="/mnt/pfs/xuhaoming/xr-2/.venv/bin/python")
    ap.add_argument("--client", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/code/scripts/rt2_rollout_client.py")
    ap.add_argument("--robotwin-root", default="/mnt/pfs/xuhaoming/xr-2/RoboTwin")
    ap.add_argument("--planner-backend", choices=["curobo"], default="curobo")
    a = ap.parse_args()

    labels = [a.label_a, a.label_b]
    servers = [a.server_a.rstrip("/"), a.server_b.rstrip("/")]
    modes = [a.mode_a, a.mode_b]
    if len(set(labels)) != 2:
        raise ValueError("A/B labels must be distinct")
    server_health = [health(server) for server in servers]
    for label, mode, row in zip(labels, modes, server_health):
        expected = {"status": "ok", "checkpoint": os.path.abspath(a.checkpoint),
                    "step": 50000, "wrist": 1, "placement": "entropy", "L": 512,
                    "action_steps": 50, "obs_preprocess": mode}
        mismatch = {key: {"server": row.get(key), "expected": value}
                    for key, value in expected.items() if row.get(key) != value}
        if mismatch:
            raise ValueError(f"wrong {label} server: {mismatch}")

    os.makedirs(a.out, exist_ok=True)
    env = os.environ.copy()
    env["ROBOTWIN_ROOT"] = os.path.abspath(a.robotwin_root)
    env["ROBOTWIN_PLANNER_BACKEND"] = a.planner_backend
    records = []
    for task, trial in parse_cases(a.cases):
        source_path, source = source_case(a.source_run, task, trial)
        procs = []
        logs = []
        result_paths = []
        for label, server in zip(labels, servers):
            case_out = os.path.join(a.out, task, f"trial_{trial:02d}", label)
            os.makedirs(case_out, exist_ok=True)
            log_path = os.path.join(case_out, "rollout.log")
            log = open(log_path, "w")
            cmd = [
                a.python, a.client, "--task", task, "--cfg", source["cfg"],
                "--seed", str(source["seed"]), "--seed_tries", "1",
                "--episode_index", str(trial), "--instr", source["instruction"],
                "--server", server, "--out", case_out, "--exec_horizon", "50",
                "--smooth", "1", "--wrist", "1",
            ]
            procs.append(subprocess.Popen(cmd, cwd=a.robotwin_root, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, text=True))
            logs.append((log, log_path))
            result_paths.append(os.path.join(case_out, "result.json"))
        returncodes = [proc.wait() for proc in procs]
        for log, _ in logs:
            log.close()
        if any(code != 0 for code in returncodes):
            raise RuntimeError({labels[i]: {"returncode": returncodes[i], "log": logs[i][1]}
                                for i in range(2)})
        results = [validate_result(result_paths[i], source, labels[i], servers[i]) for i in range(2)]
        record = {
            "task": task, "trial": trial, "source_result": source_path,
            "source_success": bool(source["success"]), "seed": int(source["seed"]),
            "instruction": source["instruction"],
            labels[0]: results[0], labels[1]: results[1],
        }
        records.append(record)
        write_json(os.path.join(a.out, "summary.partial.json"), {"records": records})
        print(f"[ab] {task}:{trial} source={source['success']} "
              f"{labels[0]}={results[0]['success']} {labels[1]}={results[1]['success']}", flush=True)

    summary = {
        "source_run": os.path.abspath(a.source_run), "output": os.path.abspath(a.out),
        "checkpoint": os.path.abspath(a.checkpoint), "labels": labels, "modes": modes,
        "servers": servers, "server_health": server_health, "cases": len(records),
        "success": {label: sum(bool(row[label]["success"]) for row in records) for label in labels},
        "records": records,
    }
    write_json(os.path.join(a.out, "summary.json"), summary)
    print(json.dumps(summary["success"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
