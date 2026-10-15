#!/usr/bin/env bash
# Preparation only. Training uses local assets and never calls this script.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/pfs/xuhaoming/wm-align}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-/usr/bin/python3}"
RAE_REVISION=a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041
RAE_REPOSITORY="${WORK_ROOT}/third_party/rae-${RAE_REVISION:0:7}"

mkdir -p "${WORK_ROOT}/tools" "${WORK_ROOT}/tmp" "${WORK_ROOT}/metadata" \
  "${WORK_ROOT}/models" "${WORK_ROOT}/logs" "${WORK_ROOT}/third_party" \
  "${WORK_ROOT}/cache/uv" "${WORK_ROOT}/cache/huggingface" "${WORK_ROOT}/cache/torch"
export UV_INSTALL_DIR="${WORK_ROOT}/tools" UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR="${WORK_ROOT}/cache/uv" UV_PYTHON_INSTALL_DIR="${WORK_ROOT}/tools/python"
export UV_TOOL_DIR="${WORK_ROOT}/tools/uv-tools" UV_TOOL_BIN_DIR="${WORK_ROOT}/tools/bin"
export TMPDIR="${WORK_ROOT}/tmp" HF_HOME="${WORK_ROOT}/cache/huggingface"
export TORCH_HOME="${WORK_ROOT}/cache/torch" XDG_CACHE_HOME="${WORK_ROOT}/cache"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/pycache" PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [ ! -x "${WORK_ROOT}/tools/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh -o "${WORK_ROOT}/tmp/install-uv.sh"
  sh "${WORK_ROOT}/tmp/install-uv.sh"
fi
UV="${WORK_ROOT}/tools/uv"
PYTHON="${WORK_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  "${UV}" venv --python "${BASE_PYTHON}" "${WORK_ROOT}/.venv"
fi
"${UV}" pip install --python "${PYTHON}" torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
"${UV}" pip install --python "${PYTHON}" -r "${SOURCE_ROOT}/code/requirements-object-event-v71.txt"

if [ ! -d "${RAE_REPOSITORY}" ]; then
  git clone --no-checkout https://github.com/bytetriper/RAE.git "${RAE_REPOSITORY}"
fi
git -C "${RAE_REPOSITORY}" checkout --detach "${RAE_REVISION}"
"${PYTHON}" "${SOURCE_ROOT}/code/scripts/prepare_object_event_assets_v71.py" \
  --work-root "${WORK_ROOT}" --rae-repository "${RAE_REPOSITORY}" --image-size 512
"${PYTHON}" - "${WORK_ROOT}/metadata/environment_v71.json" <<'PY'
from importlib.metadata import version
import json
from pathlib import Path
import sys

packages = ("torch", "torchvision", "transformers", "huggingface-hub", "numpy", "Pillow",
            "av", "PyYAML", "einops", "lpips", "swanlab")
report = {"python": sys.executable, "python_version": sys.version,
          "packages": {name: version(name) for name in packages}}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
PY
printf '[object-event-v71] prepared python=%s assets=%s\n' \
  "${PYTHON}" "${WORK_ROOT}/metadata/rae_assets_512.json"
