#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
RUN_NAME="${RUN_NAME:-learning_objective_object_state_v52_seed17_${SOURCE_REVISION:0:7}}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}/v52_object_state_0022500.pt}"
EVAL_NAME="${EVAL_NAME:-learning_objective_object_state_v52_step22500}"
EVAL_ROOT="${EVAL_ROOT:-${RUNTIME_ROOT}/outputs/v52_evaluations/${EVAL_NAME}}"
EVAL_REPORT="${EVAL_REPORT:-${EVAL_ROOT}/report.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
PY="${VENV_ROOT}/.venv/bin/python"

mkdir -p "${EVAL_ROOT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
cd "${ROOT}" || exit 2

exec "${PY}" "${ROOT}/code/scripts/evaluate_learning_objective_object_state_v52.py" \
  --data "${DATA}" --checkpoint "${CHECKPOINT}" --output "${EVAL_REPORT}" \
  --source_revision "${SOURCE_REVISION}" --evaluator_revision "${EVALUATOR_REVISION:-${SOURCE_REVISION}}" \
  --expected_step 22500 --dino_checkpoint "${DINO_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --splits "${EVAL_SPLITS:-heldseed,heldtask}" \
  --chunk_lengths "${EVAL_CHUNK_LENGTHS:-8,16,24,32}" \
  --temporal_stride "${EVAL_TEMPORAL_STRIDE:-1}" --items "${EVAL_ITEMS:-128}" \
  --batch "${EVAL_BATCH:-1}" --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-64}" \
  --amp "${AMP:-bf16}" --seed "${SEED:-17}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-}" --wandb_name "${WANDB_NAME:-${EVAL_NAME}}" \
  --wandb_group "${WANDB_GROUP:-learning-objective-object-state-v52-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
