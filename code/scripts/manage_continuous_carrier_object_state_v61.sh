#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
VARIANT="${VARIANT:-siglip_dino_object}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP_CHECKPOINT="${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v61_gates/${SOURCE_REVISION}_${VARIANT}.json}"
RUN_NAME="${RUN_NAME:-continuous_carrier_v61_${VARIANT}_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
PID_FILE="${LOG_ROOT}/launcher.pid"
LAUNCH_LOG="${LOG_ROOT}/launcher.log"

export ROOT RUNTIME_ROOT VENV_ROOT VARIANT SOURCE_REVISION DATA_INDEX
export DINO_CHECKPOINT SIGLIP_CHECKPOINT TRACKER_CHECKPOINT GATE_REPORT RUN_NAME OUT
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

verify() {
  mkdir -p "$(dirname "${GATE_REPORT}")"
  CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-0}" \
    "${PY}" "${ROOT}/code/scripts/verify_continuous_carrier_object_state_v61.py" \
      --variant "${VARIANT}" \
      --data_index "${DATA_INDEX}" \
      --dino_checkpoint "${DINO_CHECKPOINT}" \
      --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
      --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
      --source_revision "${SOURCE_REVISION}" \
      --output "${GATE_REPORT}" \
      --chunk_length "${VERIFY_CHUNK_LENGTH:-4}" \
      --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
      --student_frame_batch "${VERIFY_STUDENT_FRAME_BATCH:-8}" \
      --dino_frame_batch "${VERIFY_DINO_FRAME_BATCH:-8}" \
      --siglip_teacher_batch "${VERIFY_SIGLIP_BATCH:-8}" \
      --amp "${AMP:-bf16}"
}

foreground() {
  bash "${ROOT}/code/scripts/train_continuous_carrier_object_state_v61.sh"
}

background() {
  mkdir -p "${LOG_ROOT}"
  nohup bash "${ROOT}/code/scripts/train_continuous_carrier_object_state_v61.sh" \
    >"${LAUNCH_LOG}" 2>&1 &
  printf '%s\n' "$!" >"${PID_FILE}"
  echo "[continuous-carrier-v61-manager] started pid=$! log=${LAUNCH_LOG}"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[continuous-carrier-v61-manager] state=running pid=$(cat "${PID_FILE}")"
  else
    echo "[continuous-carrier-v61-manager] state=stopped"
  fi
  echo "[continuous-carrier-v61-manager] variant=${VARIANT} gate=${GATE_REPORT} out=${OUT}"
  if [ -f "${LAUNCH_LOG}" ]; then tail -n "${STATUS_LINES:-80}" "${LAUNCH_LOG}"; fi
  if [ -f "${OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-80}" "${OUT}/train.jsonl"; fi
}

case "${COMMAND}" in
  verify) verify ;;
  foreground) foreground ;;
  start) background ;;
  resume)
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    background
    ;;
  resume-foreground)
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    foreground
    ;;
  status) status ;;
  *) echo "usage: $0 {verify|foreground|start|resume|resume-foreground|status}" ;;
esac
