#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DECODE_REPORT="${DECODE_REPORT:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/decode_frontier.json}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v59_gates/${SOURCE_REVISION}_startup.json}"
INIT_FROM="${INIT_FROM:-${RUNTIME_ROOT}/outputs/query_persistent_object_state_v58_seed17_3379685/latest.pt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
RUN_NAME="${RUN_NAME:-object_transition_objective_v59_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
PID_FILE="${LOG_ROOT}/launcher.pid"
LAUNCH_LOG="${LOG_ROOT}/launcher.log"

export ROOT RUNTIME_ROOT VENV_ROOT DATA_INDEX DECODE_REPORT SOURCE_REVISION
export GATE_REPORT INIT_FROM DINO_CHECKPOINT TRACKER_CHECKPOINT RUN_NAME OUT LOG_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

verify_run() {
  mkdir -p "$(dirname "${GATE_REPORT}")"
  CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-2}" \
    "${PY}" "${ROOT}/code/scripts/verify_object_transition_objective_v59.py" \
      --data_index "${DATA_INDEX}" --decode_report "${DECODE_REPORT}" \
      --dino_checkpoint "${DINO_CHECKPOINT}" \
      --tracker_checkpoint "${TRACKER_CHECKPOINT}" --init_from "${INIT_FROM}" \
      --output "${GATE_REPORT}" --source_revision "${SOURCE_REVISION}" \
      --history_lengths "${HISTORY_LENGTHS:-1,2,3,4}" \
      --teacher_future_frames "${TEACHER_FUTURE_FRAMES:-4}" \
      --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400,800}" \
      --dino_frame_batch "${VERIFY_DINO_FRAME_BATCH:-32}" --amp "${AMP:-bf16}"
}

foreground() {
  bash "${ROOT}/code/scripts/train_object_transition_objective_v59.sh"
}

background() {
  mkdir -p "${LOG_ROOT}"
  nohup bash "${ROOT}/code/scripts/train_object_transition_objective_v59.sh" \
    >"${LAUNCH_LOG}" 2>&1 &
  PID=$!
  printf '%s\n' "${PID}" >"${PID_FILE}"
  echo "[object-transition-v59-manager] started pid=${PID} log=${LAUNCH_LOG}"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[object-transition-v59-manager] state=running pid=$(cat "${PID_FILE}")"
  else
    echo "[object-transition-v59-manager] state=stopped"
  fi
  echo "[object-transition-v59-manager] gate=${GATE_REPORT} out=${OUT}"
  if [ -f "${LAUNCH_LOG}" ]; then tail -n "${STATUS_LINES:-80}" "${LAUNCH_LOG}"; fi
  if [ -f "${OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-80}" "${OUT}/train.jsonl"; fi
}

evaluate_run() {
  CHECKPOINT="${EVAL_CHECKPOINT:-${OUT}/latest.pt}" \
  OUTPUT="${EVAL_OUTPUT:-${OUT}/unseen_objective_evaluation.json}" \
  EVALUATOR_REVISION="${SOURCE_REVISION}" \
  WANDB_NAME="${EVAL_WANDB_NAME:-${RUN_NAME}_unseen_eval}" \
  WANDB_GROUP="${EVAL_WANDB_GROUP:-object-transition-objective-v59-eval}" \
    bash "${ROOT}/code/scripts/evaluate_object_transition_objective_v59_suite.sh"
}

case "${COMMAND}" in
  verify) verify_run ;;
  evaluate) evaluate_run ;;
  foreground) foreground ;;
  start) background ;;
  resume)
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    background ;;
  resume-foreground)
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    foreground ;;
  status) status ;;
  stop)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      echo "[object-transition-v59-manager] stop requested pid=$(cat "${PID_FILE}")"
    else
      echo "[object-transition-v59-manager] no running launcher"
    fi ;;
  *)
    echo "usage: $0 {verify|foreground|start|resume|resume-foreground|evaluate|status|stop}"
    exit 2 ;;
esac
