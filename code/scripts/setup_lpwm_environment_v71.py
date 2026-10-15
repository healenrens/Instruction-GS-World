#!/usr/bin/env python3
"""Install the approved LPWM runtime in an independent PFS uv environment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="/mnt/pfs/public/xuhaoming/instruct_gs_world")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = Path(args.work_root).resolve()
    uv = root/"tools/benchmark-v71/uv-0.9.9/uv"
    environment = root/"environments/lpwm-v71"
    python = environment/"bin/python"
    temporary = root/"tmp/lpwm-v71"
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=str(root/"cache/lpwm-uv"), UV_PYTHON_DOWNLOADS="never",
               TMPDIR=str(temporary), XDG_CACHE_HOME=str(root/"cache/lpwm"),
               PYTHONPYCACHEPREFIX=str(root/"cache/lpwm-pycache"))
    if not python.is_file():
        subprocess.run([str(uv), "venv", "--python", args.python, str(environment)], check=True, env=env)
    subprocess.run([str(uv), "pip", "install", "--python", str(python), "torch==2.8.0", "torchvision==0.23.0",
                    "--index-url", "https://download.pytorch.org/whl/cu126"], check=True, env=env)
    subprocess.run([str(uv), "pip", "install", "--python", str(python), "--index-url", "https://pypi.org/simple",
                    "-r", str(Path(__file__).resolve().parents[1]/"requirements-lpwm-v71.txt")], check=True, env=env)
    packages = subprocess.check_output([str(uv), "pip", "freeze", "--python", str(python)], text=True, env=env)
    report = {"python": str(python), "environment": str(environment), "packages": packages.splitlines(),
              "existing_training_environment_modified": False, "tensorflow_installed_by_this_script": False}
    path = root/"metadata/lpwm_environment_v71.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
