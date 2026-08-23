#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL_NAME="${EVAL_NAME:-query_persistent_object_state_v58_held_eval}"
EVAL_REPORT="${EVAL_REPORT:-${RUNTIME_ROOT}/outputs/v58_evaluations/${EVAL_NAME}.json}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "[query-object-v58-eval] CHECKPOINT is missing: ${CHECKPOINT}"
  exit 2
fi
mkdir -p "$(dirname "${EVAL_REPORT}")" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
cd "${ROOT}" || exit 2
exec "${PY}" "${ROOT}/code/scripts/evaluate_query_persistent_object_state_v58.py" \
  --data_index "${DATA_INDEX}" --checkpoint "${CHECKPOINT}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" --output "${EVAL_REPORT}" \
  --history_lengths "${HISTORY_LENGTHS:-1,2,3,4}" \
  --teacher_future_frames "${TEACHER_FUTURE_FRAMES:-4}" \
  --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400,800}" \
  --samples_per_condition "${EVAL_SAMPLES_PER_CONDITION:-64}" \
  --batch "${EVAL_BATCH:-8}" --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-96}" \
  --amp "${AMP:-bf16}" --seed "${EVAL_SEED:-117}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-}" --wandb_name "${WANDB_NAME:-${EVAL_NAME}}" \
  --wandb_group "${WANDB_GROUP:-query-persistent-object-state-v58-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
