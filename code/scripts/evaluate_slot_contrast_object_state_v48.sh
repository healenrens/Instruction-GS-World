#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/slot_contrast_object_state_v48_stable_recurrence_seed17_058ae4a/v48_0050000.pt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs/v48_held_object_state}"
PY="${VENV_ROOT}/.venv/bin/python"

if [ ! -x "${PY}" ]; then
  echo "[v48-held] Python runtime is missing: ${PY}"
  exit 2
fi
if [ ! -f "${DATA}/episode_manifest.json" ]; then
  echo "[v48-held] RGB episode manifest is missing under ${DATA}"
  exit 2
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "[v48-held] checkpoint is missing: ${CHECKPOINT}"
  exit 2
fi
if [ ! -f "${DINO_CHECKPOINT}" ]; then
  echo "[v48-held] DINO checkpoint is missing: ${DINO_CHECKPOINT}"
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}"
IFS=',' read -r -a REQUESTED_SPLITS <<< "${SPLITS:-heldseed,heldtask}"
for SPLIT in "${REQUESTED_SPLITS[@]}"; do
  REPORT="${OUTPUT_ROOT}/${SPLIT}.json"
  VISUALS="${OUTPUT_ROOT}/${SPLIT}_visualizations"
  echo "[v48-held] split=${SPLIT} checkpoint=${CHECKPOINT} report=${REPORT}"
  "${PY}" "${ROOT}/code/scripts/evaluate_slot_contrast_object_state_v48.py" \
    --data "${DATA}" \
    --checkpoint "${CHECKPOINT}" \
    --dino_checkpoint "${DINO_CHECKPOINT}" \
    --split "${SPLIT}" \
    --output "${REPORT}" \
    --visualization_dir "${VISUALS}" \
    --max_items "${MAX_ITEMS:-512}" \
    --batch "${BATCH:-8}" \
    --workers "${WORKERS:-7}" \
    --dino_frame_batch "${DINO_FRAME_BATCH:-128}" \
    --amp "${AMP:-bf16}" \
    --seed "${SEED:-17}" \
    --chunk_lengths "${CHUNK_LENGTHS:-8,16,24,32}" \
    --temporal_strides "${TEMPORAL_STRIDES:-1,2,3,4}" \
    --cross_episode_pairs "${CROSS_EPISODE_PAIRS:-2000}" \
    --qualitative_items "${QUALITATIVE_ITEMS:-8}"
  EVALUATION_RC=$?
  if [ "${EVALUATION_RC}" -ne 0 ]; then
    echo "[v48-held] split=${SPLIT} failed rc=${EVALUATION_RC}"
    exit "${EVALUATION_RC}"
  fi
done

echo "[v48-held] completed output_root=${OUTPUT_ROOT}"
