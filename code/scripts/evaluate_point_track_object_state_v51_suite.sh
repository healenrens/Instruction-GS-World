#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
EVALUATOR_REVISION="${EVALUATOR_REVISION:-${SOURCE_REVISION}}"
EXPECTED_STEP="${EXPECTED_STEP:-22500}"
TRAIN_RUN="${TRAIN_RUN:-point_track_object_state_v51_object_state_seed17_${SOURCE_REVISION:0:7}}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/${TRAIN_RUN}/v51_object_state_0022500.pt}"
EVAL_NAME="${EVAL_NAME:-point_track_object_state_v51_step22500_comprehensive}"
EVAL_ROOT="${EVAL_ROOT:-${RUNTIME_ROOT}/outputs/v51_evaluations/${EVAL_NAME}}"
EVAL_REPORT="${EVAL_REPORT:-${EVAL_ROOT}/report.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
WANDB_DIR="${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
PY="${VENV_ROOT}/.venv/bin/python"

if [ -z "${SOURCE_REVISION}" ]; then
  echo "[point-track-v51-eval] SOURCE_REVISION is missing"
  exit 2
fi

mkdir -p "${EVAL_ROOT}" "${WANDB_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

cd "${ROOT}" || exit 2

exec "${PY}" "${ROOT}/code/scripts/evaluate_point_track_object_state_v51_suite.py" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${EVAL_REPORT}" \
  --source_revision "${SOURCE_REVISION}" \
  --evaluator_revision "${EVALUATOR_REVISION}" \
  --expected_step "${EXPECTED_STEP}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --splits "${EVAL_SPLITS:-heldseed,heldtask}" \
  --chunk_lengths "${EVAL_CHUNK_LENGTHS:-8,16,24,32}" \
  --temporal_stride "${EVAL_TEMPORAL_STRIDE:-1}" \
  --items "${EVAL_ITEMS:-128}" \
  --batch "${EVAL_BATCH:-1}" \
  --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-64}" \
  --tracker_sequence_batch "${EVAL_TRACKER_SEQUENCE_BATCH:-1}" \
  --causal_items "${EVAL_CAUSAL_ITEMS:-8}" \
  --amp "${AMP:-bf16}" \
  --seed "${SEED:-17}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${EVAL_NAME}}" \
  --wandb_group "${WANDB_GROUP:-point-track-object-state-v51-eval}" \
  --wandb_dir "${WANDB_DIR}"
