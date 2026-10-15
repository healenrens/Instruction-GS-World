#!/usr/bin/env python3
"""Prepare a separate CPU simulation environment on server PFS using uv."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlretrieve


CLIPORT_REVISION = "2be5c47b5bb9bb7040ad90693288b87b1e18e7ad"
UV_VERSION = "0.9.9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="/mnt/pfs/public/xuhaoming/instruct_gs_world")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = Path(args.work_root).resolve()
    source = Path(__file__).resolve().parents[2]
    temporary, tools = root/"tmp/benchmark-v71", root/"tools/benchmark-v71"
    for path in (temporary, tools, root/"environments", root/"third_party", root/"metadata"):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=str(root/"cache/benchmark-uv"),
               UV_PYTHON_INSTALL_DIR=str(tools/"python"), UV_PYTHON_DOWNLOADS="never",
               TMPDIR=str(temporary), XDG_CACHE_HOME=str(root/"cache/benchmark"),
               PYTHONPYCACHEPREFIX=str(root/"cache/benchmark-pycache"))
    uv = tools/f"uv-{UV_VERSION}"/"uv"
    if not uv.is_file():
        bundle = temporary/f"uv-{UV_VERSION}-linux-x86_64.tar.gz"
        urlretrieve(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz", bundle)
        uv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(bundle), "--strip-components=1", "-C", str(uv.parent)], check=True, env=env)
    environment = root/"environments/cliport-v71"
    python = environment/"bin/python"
    if not python.is_file():
        subprocess.run([str(uv), "venv", "--python", args.python, str(environment)], check=True, env=env)
    subprocess.run([str(uv), "pip", "install", "--python", str(python),
                    "torch==2.7.0+cpu", "torchvision==0.22.0+cpu",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True, env=env)
    subprocess.run([str(uv), "pip", "install", "--python", str(python),
                    "--index-url", "https://pypi.org/simple", "-r",
                    str(source/"code/requirements-object-event-benchmark-v71.txt")], check=True, env=env)
    repository = root/"third_party"/f"cliport-{CLIPORT_REVISION[:7]}"
    if not repository.is_dir():
        archive = temporary/f"cliport-{CLIPORT_REVISION}.tar.gz"
        urlretrieve(f"https://codeload.github.com/cliport/cliport/tar.gz/{CLIPORT_REVISION}", archive)
        unpacked = temporary/f"cliport-{CLIPORT_REVISION}"
        unpacked.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(unpacked)], check=True, env=env)
        unpacked.rename(repository)
    packages = subprocess.check_output([str(uv), "pip", "freeze", "--python", str(python)], env=env, text=True)
    report = {"python": str(python), "environment": str(environment), "uv_version": UV_VERSION,
              "cliport_source": str(repository), "cliport_revision": CLIPORT_REVISION,
              "packages": packages.splitlines(), "torch_variant": "cpu",
              "existing_training_environment_modified": False, "simulation_executed": False}
    output = root/"metadata/object_event_benchmark_environment_v71.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
