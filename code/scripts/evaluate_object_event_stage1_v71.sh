#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
PYTHON="${PYTHON:-${WORK_ROOT}/.venv/bin/python}"
MANIFEST="${MANIFEST:-${WORK_ROOT}/metadata/cliport_event_v71.json}"
ASSETS="${ASSETS:-${WORK_ROOT}/metadata/rae_assets_512.json}"
OUT="${OUT:-${WORK_ROOT}/outputs/v71_stage1_held/$(basename "$(dirname "${CHECKPOINT}")")_$(basename "${CHECKPOINT}" .pt)}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
EXTRA=()
if [[ "${RESUME_EVAL:-0}" == 1 ]]; then
  EXTRA+=(--resume)
fi

printf '[stage1-held-v71] checkpoint=%s\n[stage1-held-v71] output=%s\n' "${CHECKPOINT}" "${OUT}"
exec "${PYTHON}" "${ROOT}/code/scripts/evaluate_object_event_stage1_v71.py" \
  --checkpoint "${CHECKPOINT}" --manifest "${MANIFEST}" --assets "${ASSETS}" --out "${OUT}" \
  --partition "${PARTITION:-development}" --items "${ITEMS:-200}" --video-cases "${VIDEO_CASES:-12}" \
  --frame-batch "${FRAME_BATCH:-4}" --decode-batch "${DECODE_BATCH:-2}" --precision "${PRECISION:-bf16}" \
  "${EXTRA[@]}"
