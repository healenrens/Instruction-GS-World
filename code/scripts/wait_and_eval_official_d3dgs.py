"""Wait for an official D3DGS params file, then run render evaluation."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_repo", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_script", default="code/scripts/eval_official_d3dgs_render.py")
    ap.add_argument("--frames", default="default")
    ap.add_argument("--cameras", default="all")
    ap.add_argument("--poll_sec", type=int, default=300)
    ap.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout")
    args = ap.parse_args()

    if args.poll_sec <= 0:
        raise ValueError("--poll_sec must be positive")
    if args.timeout_sec < 0:
        raise ValueError("--timeout_sec must be non-negative")

    official_repo = Path(args.official_repo).expanduser().resolve()
    params_path = official_repo / "output" / args.exp / args.sequence / "params.npz"
    eval_script = Path(args.eval_script).expanduser().resolve()
    if not eval_script.exists():
        raise FileNotFoundError(eval_script)

    start = time.time()
    while not params_path.exists():
        elapsed = int(time.time() - start)
        print(f"waiting elapsed={elapsed}s params={params_path}", flush=True)
        if args.timeout_sec and elapsed >= args.timeout_sec:
            raise TimeoutError(f"timed out waiting for {params_path}")
        time.sleep(args.poll_sec)

    if params_path.stat().st_size <= 0:
        raise ValueError(f"{params_path} is empty")
    print(f"params_ready path={params_path} size={params_path.stat().st_size}", flush=True)

    cmd = [
        sys.executable,
        str(eval_script),
        "--official_repo",
        str(official_repo),
        "--sequence",
        args.sequence,
        "--exp",
        args.exp,
        "--frames",
        args.frames,
        "--cameras",
        args.cameras,
        "--out_dir",
        args.out_dir,
    ]
    print("running " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
