#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
VARIANT="${VARIANT:-siglip2_dino_object}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
RUN_NAME="${RUN_NAME:-continuous_carrier_v61_${VARIANT}_seed17_${SOURCE_REVISION:0:7}}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}/latest.pt}"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP2_CHECKPOINT="${SIGLIP2_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
EVAL_NAME="${EVAL_NAME:-${RUN_NAME}_held_object_state_eval}"
OUTPUT="${OUTPUT:-${RUNTIME_ROOT}/outputs/v61_evaluations/${EVAL_NAME}.json}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$(dirname "${OUTPUT}")" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"
exec "${PY}" "${ROOT}/code/scripts/evaluate_continuous_carrier_object_state_v61.py" \
  --checkpoint "${CHECKPOINT}" \
  --data_index "${DATA_INDEX}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --siglip2_checkpoint "${SIGLIP2_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --chunk_lengths "${EVAL_CHUNK_LENGTHS:-4,6,8}" \
  --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400}" \
  --student_frame_batch "${EVAL_STUDENT_FRAME_BATCH:-16}" \
  --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-16}" \
  --siglip2_teacher_batch "${EVAL_SIGLIP2_BATCH:-16}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_name "${EVAL_NAME}" \
  --wandb_group "${WANDB_GROUP:-continuous-carrier-object-state-v61-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
