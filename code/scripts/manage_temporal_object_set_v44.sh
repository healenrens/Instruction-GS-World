#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
PY="${VENV_ROOT}/.venv/bin/python"
COMMIT="$(cd "${ROOT}" && git rev-parse HEAD 2>/dev/null)"
RUN_NAME="${RUN_NAME:-temporal_object_set_v44_seed17_${COMMIT:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v44_gates/${COMMIT}_startup.json}"
PID_FILE="${LOG_ROOT}/launcher.pid"
LAUNCH_LOG="${LOG_ROOT}/launcher.log"

verify_run() {
  if [ ! -x "${PY}" ]; then
    echo "[temporal-object-v44-manager] python is missing: ${PY}"
    return 2
  fi
  if [ ! -f "${DATA}/episode_manifest.json" ]; then
    echo "[temporal-object-v44-manager] data manifest is missing: ${DATA}"
    return 2
  fi
  if [ ! -f "${DATA}/episode_manifest.verified.sha256" ]; then
    echo "[temporal-object-v44-manager] verified manifest checksum is missing"
    return 2
  fi
  (cd "${DATA}" && sha256sum -c episode_manifest.verified.sha256) || return 2
  mkdir -p "$(dirname "${GATE_REPORT}")"
  rm -f "${GATE_REPORT}"
  echo "[temporal-object-v44-manager] verifier=${GATE_REPORT}"
  "${PY}" "${ROOT}/code/scripts/verify_temporal_object_set_v44.py" \
    --data "${DATA}" \
    --output "${GATE_REPORT}" \
    --dino_frame_batch "${DINO_FRAME_BATCH:-16}" \
    --amp "${AMP:-bf16}"
}

launch_foreground() {
  export ROOT RUNTIME_ROOT VENV_ROOT DATA RUN_NAME OUT LOG_ROOT GATE_REPORT
  bash "${ROOT}/code/scripts/train_temporal_object_set_v44.sh"
}

start_background() {
  if [ ! -f "${GATE_REPORT}" ]; then
    echo "[temporal-object-v44-manager] gate report is missing: ${GATE_REPORT}"
    return 2
  fi
  mkdir -p "${LOG_ROOT}"
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[temporal-object-v44-manager] already running pid=$(cat "${PID_FILE}")"
    return 2
  fi
  export ROOT RUNTIME_ROOT VENV_ROOT DATA RUN_NAME OUT LOG_ROOT GATE_REPORT
  nohup bash "${ROOT}/code/scripts/train_temporal_object_set_v44.sh" \
    >"${LAUNCH_LOG}" 2>&1 &
  PID=$!
  printf '%s\n' "${PID}" >"${PID_FILE}"
  echo "[temporal-object-v44-manager] started pid=${PID} log=${LAUNCH_LOG}"
}

show_status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[temporal-object-v44-manager] state=running pid=$(cat "${PID_FILE}")"
  else
    echo "[temporal-object-v44-manager] state=stopped"
  fi
  echo "[temporal-object-v44-manager] out=${OUT}"
  echo "[temporal-object-v44-manager] gate=${GATE_REPORT}"
  if [ -f "${LAUNCH_LOG}" ]; then
    tail -n "${STATUS_LINES:-80}" "${LAUNCH_LOG}"
  fi
  if [ -f "${OUT}/train.jsonl" ]; then
    tail -n "${STATUS_LINES:-80}" "${OUT}/train.jsonl"
  fi
}

case "${COMMAND}" in
  verify)
    verify_run
    ;;
  foreground)
    launch_foreground
    ;;
  start)
    start_background
    ;;
  resume)
    RESUME="${RESUME:-${OUT}/latest.pt}"
    export RESUME
    start_background
    ;;
  resume-foreground)
    RESUME="${RESUME:-${OUT}/latest.pt}"
    export RESUME
    launch_foreground
    ;;
  status)
    show_status
    ;;
  stop)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      echo "[temporal-object-v44-manager] stop requested pid=$(cat "${PID_FILE}")"
    else
      echo "[temporal-object-v44-manager] no running launcher"
    fi
    ;;
  *)
    echo "usage: $0 {verify|foreground|start|resume|resume-foreground|status|stop}"
    exit 2
    ;;
esac
