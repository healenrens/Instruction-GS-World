#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
STAGE="${STAGE:-tokenizer}"
SPEC="${MULTISOURCE_SPEC:-${ROOT}/configs/multisource_real_robot_video_v53.json}"
INDEX_ROOT="${MULTISOURCE_INDEX_ROOT:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53}"
DATA_INDEX="${DATA_INDEX:-${INDEX_ROOT}/index.json}"
DATA_REPORT="${DATA_REPORT:-${INDEX_ROOT}/verification.json}"
DECODE_REPORT="${DECODE_REPORT:-${INDEX_ROOT}/decode_frontier.json}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
RUN_NAME="${RUN_NAME:-semantic_object_world_model_v53_${STAGE}_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v53_gates/${SOURCE_REVISION}_${STAGE}.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
PID_FILE="${LOG_ROOT}/launcher.pid"
LAUNCH_LOG="${LOG_ROOT}/launcher.log"

export ROOT RUNTIME_ROOT VENV_ROOT STAGE DATA_INDEX DECODE_REPORT SOURCE_REVISION RUN_NAME OUT LOG_ROOT
export GATE_REPORT DINO_CHECKPOINT
export MULTISOURCE_SPEC="${SPEC}" MULTISOURCE_INDEX_ROOT="${INDEX_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"

build_index() {
  mkdir -p "${INDEX_ROOT}"
  "${PY}" "${ROOT}/code/scripts/build_multisource_video_index_v53.py" \
    --spec "${SPEC}" --output "${DATA_INDEX}"
}

verify_data() {
  "${PY}" "${ROOT}/code/scripts/verify_multisource_video_index_v53.py" \
    --data_index "${DATA_INDEX}" --output "${DATA_REPORT}" || return $?
  local nproc="${NPROC_PER_NODE:-auto}"
  if [ "${nproc}" = auto ]; then
    nproc="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
  fi
  local batch="${BATCH_PER_GPU:-auto}"
  if [ "${batch}" = auto ]; then
    local memory
    memory="$("${PY}" -c 'import torch; print(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) // 2**20)')"
    if [ "${memory}" -ge 76000 ]; then batch=32
    elif [ "${memory}" -ge 45000 ]; then batch=16
    elif [ "${memory}" -ge 22000 ]; then batch=8
    else batch=2
    fi
  fi
  local workers="${WORKERS_PER_RANK:-auto}"
  if [ "${workers}" = auto ]; then
    workers="$("${PY}" -c "import os; print(max(2, min(8, (os.cpu_count() or 8) // (2 * ${nproc}))))")"
  fi
  "${PY}" "${ROOT}/code/scripts/verify_multisource_decode_frontier_v53.py" \
    --data_index "${DATA_INDEX}" --output "${DECODE_REPORT}" \
    --world_size "${nproc}" --batch_size "${batch}" \
    --workers_per_rank "${workers}" --prefetch_factor "${PREFETCH_FACTOR:-2}" \
    --chunk_lengths "${CHUNK_LENGTHS:-3,4,6,8}" \
    --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400}" \
    --seed "${SEED:-17}"
}

verify_model() {
  if [ -z "${SOURCE_REVISION}" ]; then
    echo "[semantic-object-v53-manager] SOURCE_REVISION is missing"
    return 2
  fi
  mkdir -p "$(dirname "${GATE_REPORT}")"
  VERIFY_ARGS=(
    --stage "${STAGE}" --data_index "${DATA_INDEX}"
    --dino_checkpoint "${DINO_CHECKPOINT}"
    --source_revision "${SOURCE_REVISION}" --output "${GATE_REPORT}"
    --chunk_length "${VERIFY_CHUNK_LENGTH:-3}"
    --training_chunk_lengths "${CHUNK_LENGTHS:-3,4,6,8}"
    --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400}"
    --dino_frame_batch "${VERIFY_DINO_FRAME_BATCH:-8}" --amp "${AMP:-bf16}"
    --stability_steps_per_source "${VERIFY_STEPS_PER_SOURCE:-2}"
    --stability_batch_per_source "${VERIFY_BATCH_PER_SOURCE:-2}"
    --stability_training_batch "${VERIFY_TRAINING_BATCH:-32}"
  )
  if [ -n "${INIT_FROM:-}" ]; then VERIFY_ARGS+=(--init_from "${INIT_FROM}"); fi
  CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-0}" \
    "${PY}" "${ROOT}/code/scripts/verify_semantic_object_world_model_v53.py" \
      "${VERIFY_ARGS[@]}"
}

foreground() {
  bash "${ROOT}/code/scripts/train_semantic_object_world_model_v53.sh"
}

background() {
  mkdir -p "${LOG_ROOT}"
  nohup bash "${ROOT}/code/scripts/train_semantic_object_world_model_v53.sh" \
    >"${LAUNCH_LOG}" 2>&1 &
  PID=$!
  printf '%s\n' "${PID}" >"${PID_FILE}"
  echo "[semantic-object-v53-manager] started pid=${PID} log=${LAUNCH_LOG}"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[semantic-object-v53-manager] state=running pid=$(cat "${PID_FILE}")"
  else
    echo "[semantic-object-v53-manager] state=stopped"
  fi
  echo "[semantic-object-v53-manager] stage=${STAGE} out=${OUT} gate=${GATE_REPORT}"
  echo "[semantic-object-v53-manager] decode_report=${DECODE_REPORT}"
  if [ -f "${LAUNCH_LOG}" ]; then tail -n "${STATUS_LINES:-80}" "${LAUNCH_LOG}"; fi
  if [ -f "${OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-80}" "${OUT}/train.jsonl"; fi
}

case "${COMMAND}" in
  build-index) build_index ;;
  verify-data) verify_data ;;
  prepare-data) build_index && verify_data ;;
  verify) verify_model ;;
  foreground) foreground ;;
  start) background ;;
  resume)
    unset INIT_FROM
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    background ;;
  resume-foreground)
    unset INIT_FROM
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    foreground ;;
  status) status ;;
  stop)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      echo "[semantic-object-v53-manager] stop requested pid=$(cat "${PID_FILE}")"
    else
      echo "[semantic-object-v53-manager] no running launcher"
    fi ;;
  *)
    echo "usage: $0 {build-index|verify-data|prepare-data|verify|foreground|start|resume|resume-foreground|status|stop}"
    exit 2 ;;
esac
