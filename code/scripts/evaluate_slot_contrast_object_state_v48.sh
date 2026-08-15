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
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}"
WANDB_GROUP="${WANDB_GROUP:-v48-held-object-state-evaluation}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-v48-held-object-state}"
WANDB_TAGS="${WANDB_TAGS:-v48,held-evaluation,object-state}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
WANDB_SOURCE_RUN="${WANDB_SOURCE_RUN:-healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world/l1eu5wr3}"

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
mkdir -p "${WANDB_DIR}"
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
    --qualitative_items "${QUALITATIVE_ITEMS:-8}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_name "${WANDB_NAME_PREFIX}-${SPLIT}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_tags "${WANDB_TAGS}" \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_source_run "${WANDB_SOURCE_RUN}"
  EVALUATION_RC=$?
  if [ "${EVALUATION_RC}" -ne 0 ]; then
    echo "[v48-held] split=${SPLIT} failed rc=${EVALUATION_RC}"
    exit "${EVALUATION_RC}"
  fi
done

echo "[v48-held] completed output_root=${OUTPUT_ROOT}"
