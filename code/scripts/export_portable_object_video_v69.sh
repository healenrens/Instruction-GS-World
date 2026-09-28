#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PORTABLE_OUT="${PORTABLE_OUT:-${RUNTIME_ROOT}/exports/object_video_v69_portable}"
OPTIONS=()
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
  OPTIONS+=(--plan_only)
fi
if [[ "${CREATE_ARCHIVE:-1}" == 0 ]]; then
  OPTIONS+=(--no_archive)
fi
mkdir -p "${PORTABLE_OUT}"
echo "[portable-v69] out=${PORTABLE_OUT} log=${PORTABLE_OUT}/export.log"
"${VENV_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/export_portable_object_video_v69.py" \
  --manifest "${MANIFEST:-${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json}" \
  --out "${PORTABLE_OUT}" --workers "${COPY_WORKERS:-4}" --items "${EXPORT_ITEMS:-0}" \
  --archive "${ARCHIVE_PATH:-}" "${OPTIONS[@]}" 2>&1 | tee -a "${PORTABLE_OUT}/export.log"
