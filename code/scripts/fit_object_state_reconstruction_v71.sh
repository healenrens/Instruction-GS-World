#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
PYTHON="${PYTHON:-${WORK_ROOT}/.venv/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

ARGS=(
  --checkpoint "${CHECKPOINT}" --out "${OUT}"
  --manifest "${MANIFEST:-${WORK_ROOT}/metadata/cliport_event_v71.json}"
  --assets "${ASSETS:-${WORK_ROOT}/metadata/rae_assets_512.json}"
  --initialization "${INITIALIZATION:-checkpoint}" --steps "${STEPS:-1000}" --batch "${BATCH:-4}"
  --scenes "${SCENES:-4}" --frame-seconds "${FRAME_SECONDS:-0,2,5,10}"
  --precision "${PRECISION:-bf16}" --frame-batch "${FRAME_BATCH:-4}" --decode-batch "${DECODE_BATCH:-2}"
  --log-every "${LOG_EVERY:-10}" --save-every "${SAVE_EVERY:-100}"
)
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "${RESUME}"); fi
if [[ -n "${SWANLAB_PROJECT:-}" ]]; then ARGS+=(--swanlab-project "${SWANLAB_PROJECT}"); fi
if [[ -n "${SWANLAB_WORKSPACE:-}" ]]; then ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}"); fi
exec "${PYTHON}" "${ROOT}/code/scripts/fit_object_state_reconstruction_v71.py" "${ARGS[@]}" "$@"
