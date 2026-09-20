#!/usr/bin/env python3
"""Explicit foreground resume using the recorded data settings and worker topology."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_revision", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    record = json.loads((out / "workers.json").read_text())
    saved = record["configuration"]
    previous = saved.get("reuse_source_revision") or saved["source_revision"]
    options = {k: v for k, v in saved.items() if k not in ("worker_count", "queries_json")}
    options.update(out=str(out), source_revision=args.source_revision, reuse_source_revision=previous,
                   stage="run", operation="build", reuse_completed=1)
    rt = os.environ.get("RUNTIME_ROOT", "/mnt/pfs/public/xuhaoming/instruct_gs_world")
    options.update(wandb_mode=os.environ.get("WANDB_MODE", "online"),
                   wandb_project=os.environ.get("WANDB_PROJECT", "instruct-gs-world"),
                   wandb_entity=os.environ.get("WANDB_ENTITY", "healenrenss-university-of-chinese-acadmic-and-science"),
                   wandb_name=out.name, wandb_dir=os.environ.get("WANDB_DIR", f"{rt}/wandb"))
    script = Path(__file__).with_name("build_grounded_motion_data_v68.py")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(record["count"]), str(script)]
    for name, value in options.items():
        value = ",".join(map(str, value)) if isinstance(value, list) else str(value)
        command.extend((f"--{name}", value))
    print(f"[motion-data-v68-resume] out={out} workers={record['count']} workers_per_gpu={saved.get('workers_per_gpu', 1)} "
          f"code={args.source_revision} reuse_previous={previous}", flush=True)
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(script.with_name("merge_grounded_motion_data_v68.py")), "--out", str(out)], check=True)


if __name__ == "__main__":
    main()
