#!/usr/bin/env bash
# Single-GPU foreground test. Source/weight downloads are separate preparation.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${WORK_ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-object_event_v71_test_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${WORK_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs/${RUN_NAME}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="${WORK_ROOT}/cache/huggingface" TORCH_HOME="${WORK_ROOT}/cache/torch"
export XDG_CACHE_HOME="${WORK_ROOT}/cache" TMPDIR="${WORK_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/pycache"
mkdir -p "${TMPDIR}" "${LOG_ROOT}"

printf '[object-event-test] out=%s log=%s\n' "${OUT}" "${LOG_ROOT}/test.log"
"${PYTHON}" "${SOURCE_ROOT}/code/scripts/test_object_event_world_model_v71.py" \
  --manifest "${MANIFEST:-${WORK_ROOT}/metadata/object_event_v71.json}" \
  --assets "${ASSETS:-${WORK_ROOT}/metadata/rae_assets_512.json}" \
  --out "${OUT}" --precision "${PRECISION:-bf16}" \
  --source-revision "${SOURCE_REVISION:-}" "$@" 2>&1 | tee "${LOG_ROOT}/test.log"
