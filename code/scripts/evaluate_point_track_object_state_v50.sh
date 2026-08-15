#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
STAGE="${STAGE:-object_state}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL_REPORT="${EVAL_REPORT:-${RUNTIME_ROOT}/outputs/v50_evaluations/${STAGE}_${SOURCE_REVISION:0:7}.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
PY="${VENV_ROOT}/.venv/bin/python"

if [ -z "${SOURCE_REVISION}" ]; then
  echo "[point-track-v50-eval] SOURCE_REVISION is missing"
  exit 2
fi
if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "[point-track-v50-eval] CHECKPOINT is missing: ${CHECKPOINT}"
  exit 2
fi
if [ ! -f "${DINO_CHECKPOINT}" ]; then
  echo "[point-track-v50-eval] DINO checkpoint is missing: ${DINO_CHECKPOINT}"
  exit 2
fi
if [ "${STAGE}" = object_state ] && [ ! -f "${TRACKER_CHECKPOINT}" ]; then
  echo "[point-track-v50-eval] tracker checkpoint is missing: ${TRACKER_CHECKPOINT}"
  exit 2
fi

mkdir -p "$(dirname "${EVAL_REPORT}")" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
cd "${ROOT}" || exit 2
exec "${PY}" "${ROOT}/code/scripts/evaluate_point_track_object_state_v50.py" \
  --stage "${STAGE}" --data "${DATA}" --checkpoint "${CHECKPOINT}" \
  --output "${EVAL_REPORT}" --source_revision "${SOURCE_REVISION}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --split "${EVAL_SPLIT:-heldseed}" --items "${EVAL_ITEMS:-128}" \
  --chunk_length "${EVAL_CHUNK_LENGTH:-16}" --temporal_stride "${EVAL_TEMPORAL_STRIDE:-2}" \
  --batch "${EVAL_BATCH:-1}" --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-32}" \
  --amp "${AMP:-bf16}" --seed "${SEED:-17}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-}" --wandb_name "${WANDB_NAME:-point_track_v50_${STAGE}_held_eval}" \
  --wandb_group "${WANDB_GROUP:-point-track-object-state-v50-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
