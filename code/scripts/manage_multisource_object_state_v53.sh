#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
SPEC="${MULTISOURCE_SPEC:-${ROOT}/configs/multisource_real_robot_video_v53.json}"
INDEX_ROOT="${MULTISOURCE_INDEX_ROOT:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53}"
DATA_INDEX="${DATA_INDEX:-${INDEX_ROOT}/index.json}"
DATA_REPORT="${DATA_REPORT:-${INDEX_ROOT}/verification.json}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v53_gates/${SOURCE_REVISION}_startup.json}"
RUN_NAME="${RUN_NAME:-multisource_object_state_world_model_v53_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"

export ROOT RUNTIME_ROOT VENV_ROOT DATA DATA_INDEX SOURCE_REVISION GATE_REPORT
export RUN_NAME OUT LOG_ROOT
export MULTISOURCE_SPEC="${SPEC}" MULTISOURCE_INDEX_ROOT="${INDEX_ROOT}"
export DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
export TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-auto}"
export GRAD_ACCUM="${GRAD_ACCUM:-auto}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
export WORKERS_PER_RANK="${WORKERS_PER_RANK:-auto}"
export DINO_FRAME_BATCH="${DINO_FRAME_BATCH:-auto}"
export TEMPORAL_STEP_MS="${TEMPORAL_STEP_MS:-33,67,100,133}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
export WANDB_GROUP="${WANDB_GROUP:-multisource-object-state-world-model-v53}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export WANDB_TAGS="${WANDB_TAGS:-v53,multisource,task-diverse,pure-video,world-model,object-state}"

build_index() {
  mkdir -p "${INDEX_ROOT}"
  "${PY}" "${ROOT}/code/scripts/build_multisource_video_index_v53.py" \
    --spec "${SPEC}" --output "${DATA_INDEX}"
}

verify_data() {
  "${PY}" "${ROOT}/code/scripts/verify_multisource_video_index_v53.py" \
    --data_index "${DATA_INDEX}" --output "${DATA_REPORT}"
}

verify_model() {
  bash "${ROOT}/code/scripts/manage_learning_objective_object_state_v52.sh" verify
}

case "${COMMAND}" in
  build-index) build_index ;;
  verify-data) verify_data ;;
  verify-model) verify_model ;;
  prepare)
    build_index && verify_data && verify_model
    ;;
  foreground)
    bash "${ROOT}/code/scripts/manage_learning_objective_object_state_v52.sh" foreground
    ;;
  start)
    bash "${ROOT}/code/scripts/manage_learning_objective_object_state_v52.sh" start
    ;;
  resume-foreground)
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    bash "${ROOT}/code/scripts/manage_learning_objective_object_state_v52.sh" resume-foreground
    ;;
  resume)
    export RESUME="${RESUME:-${OUT}/latest.pt}"
    bash "${ROOT}/code/scripts/manage_learning_objective_object_state_v52.sh" resume
    ;;
  status)
    bash "${ROOT}/code/scripts/manage_learning_objective_object_state_v52.sh" status
    ;;
  *)
    echo "usage: $0 {build-index|verify-data|verify-model|prepare|foreground|start|resume-foreground|resume|status}"
    exit 2
    ;;
esac
