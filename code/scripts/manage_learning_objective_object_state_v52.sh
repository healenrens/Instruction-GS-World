#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
RUN_NAME="${RUN_NAME:-learning_objective_object_state_v52_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v52_gates/${SOURCE_REVISION}_startup.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
PY="${VENV_ROOT}/.venv/bin/python"
PID_FILE="${LOG_ROOT}/launcher.pid"
LAUNCH_LOG="${LOG_ROOT}/launcher.log"
export ROOT RUNTIME_ROOT VENV_ROOT DATA SOURCE_REVISION RUN_NAME OUT LOG_ROOT
export GATE_REPORT DINO_CHECKPOINT TRACKER_CHECKPOINT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

verify_run() {
  if [ -z "${SOURCE_REVISION}" ]; then
    echo "[object-state-v52-manager] SOURCE_REVISION is missing"
    return 2
  fi
  mkdir -p "$(dirname "${GATE_REPORT}")"
  rm -f "${GATE_REPORT}"
  CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-0}" \
    "${PY}" "${ROOT}/code/scripts/verify_learning_objective_object_state_v52.py" \
      --data "${DATA}" --output "${GATE_REPORT}" \
      --source_revision "${SOURCE_REVISION}" \
      --dino_checkpoint "${DINO_CHECKPOINT}" \
      --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
      --dino_frame_batch "${VERIFY_DINO_FRAME_BATCH:-16}" \
      --chunk_length "${VERIFY_CHUNK_LENGTH:-8}" --amp "${AMP:-bf16}"
}

foreground() {
  bash "${ROOT}/code/scripts/train_learning_objective_object_state_v52.sh"
}

background() {
  mkdir -p "${LOG_ROOT}"
  nohup bash "${ROOT}/code/scripts/train_learning_objective_object_state_v52.sh" \
    >"${LAUNCH_LOG}" 2>&1 &
  PID=$!
  printf '%s\n' "${PID}" >"${PID_FILE}"
  echo "[object-state-v52-manager] started pid=${PID} log=${LAUNCH_LOG}"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[object-state-v52-manager] state=running pid=$(cat "${PID_FILE}")"
  else
    echo "[object-state-v52-manager] state=stopped"
  fi
  echo "[object-state-v52-manager] out=${OUT} gate=${GATE_REPORT}"
  if [ -f "${LAUNCH_LOG}" ]; then tail -n "${STATUS_LINES:-80}" "${LAUNCH_LOG}"; fi
  if [ -f "${OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-80}" "${OUT}/train.jsonl"; fi
}

case "${COMMAND}" in
  verify) verify_run ;;
  foreground) foreground ;;
  start) background ;;
  resume)
    RESUME="${RESUME:-${OUT}/latest.pt}"
    export RESUME
    background ;;
  resume-foreground)
    RESUME="${RESUME:-${OUT}/latest.pt}"
    export RESUME
    foreground ;;
  status) status ;;
  stop)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      echo "[object-state-v52-manager] stop requested pid=$(cat "${PID_FILE}")"
    else
      echo "[object-state-v52-manager] no running launcher"
    fi ;;
  *)
    echo "usage: $0 {verify|foreground|start|resume|resume-foreground|status|stop}"
    exit 2 ;;
esac
