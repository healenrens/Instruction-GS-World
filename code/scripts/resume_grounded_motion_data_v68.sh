#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
SOURCE_REVISION="${SOURCE_REVISION:-$(<"${ROOT}/SOURCE_REVISION")}"
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "${OUT}"
"${VENV_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/resume_grounded_motion_data_v68.py" \
  --out "${OUT}" --input "${RECOVER_FROM:-}" --source_revision "${SOURCE_REVISION}" 2>&1 | tee -a "${OUT}/build.log"
