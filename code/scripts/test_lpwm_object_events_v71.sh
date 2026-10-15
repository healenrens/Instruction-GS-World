#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${WORK_ROOT}/environments/lpwm-v71/bin/python}"
RUN_NAME="${RUN_NAME:-lpwm_object_events_v71_test_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${WORK_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs/${RUN_NAME}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="${WORK_ROOT}/cache/lpwm/huggingface" TORCH_HOME="${WORK_ROOT}/cache/lpwm/torch"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/lpwm" TMPDIR="${WORK_ROOT}/tmp/lpwm-v71"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/lpwm-pycache"
mkdir -p "${TMPDIR}" "${LOG_ROOT}"
printf 'OUT=%s\nLOG=%s/test.log\n' "${OUT}" "${LOG_ROOT}"
"${PYTHON}" "${ROOT}/code/scripts/test_lpwm_object_events_v71.py" \
  --assets "${ASSETS:-${WORK_ROOT}/metadata/lpwm_assets_v71.json}" \
  --manifest "${MANIFEST:-${WORK_ROOT}/metadata/cliport_event_v71_engineering_c682ec9.json}" \
  --text-cache "${TEXT_CACHE:-${WORK_ROOT}/metadata/lpwm_cliport_text_v71}" \
  --out "${OUT}" --source-revision "${SOURCE_REVISION:-}" "$@" 2>&1 | tee "${LOG_ROOT}/test.log"
