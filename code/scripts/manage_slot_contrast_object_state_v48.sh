#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
PY="${VENV_ROOT}/.venv/bin/python"
SOURCE_REVISION="${SOURCE_REVISION:-}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
RUN_NAME="${RUN_NAME:-slot_contrast_object_state_v48_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v48_gates/${SOURCE_REVISION}_startup.json}"
PID_FILE="${LOG_ROOT}/launcher.pid"
LAUNCH_LOG="${LOG_ROOT}/launcher.log"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

require_artifacts() {
  if [ -z "${SOURCE_REVISION}" ]; then
    echo "[slot-contrast-v48-manager] SOURCE_REVISION is missing"
    return 2
  fi
  if [ ! -f "${DINO_CHECKPOINT}" ]; then
    echo "[slot-contrast-v48-manager] local DINO checkpoint is missing: ${DINO_CHECKPOINT}"
    return 2
  fi
}

verify_run() {
  require_artifacts || return $?
  mkdir -p "$(dirname "${GATE_REPORT}")"
  rm -f "${GATE_REPORT}"
  echo "[slot-contrast-v48-manager] verifier=${GATE_REPORT}"
  "${PY}" "${ROOT}/code/scripts/verify_slot_contrast_object_state_v48.py" \
    --data "${DATA}" --output "${GATE_REPORT}" \
    --source_revision "${SOURCE_REVISION}" --dino_checkpoint "${DINO_CHECKPOINT}" \
    --dino_frame_batch "${VERIFY_DINO_FRAME_BATCH:-16}" --amp "${AMP:-bf16}"
}

foreground() {
  require_artifacts || return $?
  export ROOT RUNTIME_ROOT VENV_ROOT DATA RUN_NAME OUT LOG_ROOT GATE_REPORT
  export SOURCE_REVISION DINO_CHECKPOINT
  bash "${ROOT}/code/scripts/train_slot_contrast_object_state_v48.sh"
}

background() {
  require_artifacts || return $?
  if [ ! -f "${GATE_REPORT}" ]; then
    echo "[slot-contrast-v48-manager] gate report is missing: ${GATE_REPORT}"
    return 2
  fi
  mkdir -p "${LOG_ROOT}"
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[slot-contrast-v48-manager] already running pid=$(cat "${PID_FILE}")"
    return 2
  fi
  export ROOT RUNTIME_ROOT VENV_ROOT DATA RUN_NAME OUT LOG_ROOT GATE_REPORT
  export SOURCE_REVISION DINO_CHECKPOINT
  nohup bash "${ROOT}/code/scripts/train_slot_contrast_object_state_v48.sh" \
    >"${LAUNCH_LOG}" 2>&1 &
  PID=$!
  printf '%s\n' "${PID}" >"${PID_FILE}"
  echo "[slot-contrast-v48-manager] started pid=${PID} log=${LAUNCH_LOG}"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[slot-contrast-v48-manager] state=running pid=$(cat "${PID_FILE}")"
  else
    echo "[slot-contrast-v48-manager] state=stopped"
  fi
  echo "[slot-contrast-v48-manager] out=${OUT} gate=${GATE_REPORT}"
  if [ -f "${LAUNCH_LOG}" ]; then tail -n "${STATUS_LINES:-80}" "${LAUNCH_LOG}"; fi
  if [ -f "${OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-80}" "${OUT}/train.jsonl"; fi
}

case "${COMMAND}" in
  verify) verify_run ;;
  foreground) foreground ;;
  start) background ;;
  resume) RESUME="${RESUME:-${OUT}/latest.pt}"; export RESUME; background ;;
  resume-foreground) RESUME="${RESUME:-${OUT}/latest.pt}"; export RESUME; foreground ;;
  status) status ;;
  stop)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      echo "[slot-contrast-v48-manager] stop requested pid=$(cat "${PID_FILE}")"
    else
      echo "[slot-contrast-v48-manager] no running launcher"
    fi ;;
  *) echo "usage: $0 {verify|foreground|start|resume|resume-foreground|status|stop}"; exit 2 ;;
esac
