#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
RUN_ID="${RUN_ID:-object_motion_field_g0_v66_seed17_${SOURCE_REVISION:0:7}}"
OUT_ROOT="${OUT_ROOT:-${RUNTIME_ROOT}/outputs/object_motion_field_g0_v66/${RUN_ID}}"
REPORT="${REPORT:-${OUT_ROOT}/object_motion_field_g0_audit.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

echo "[object-motion-field-v66] run_id=${RUN_ID} world=4 samples=1536 report=${REPORT}"
cd "${ROOT}"
exec "${VENV_ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node 4 \
  "${ROOT}/code/scripts/audit_object_motion_field_g0_v66.py" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --dino_checkpoint "${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${REPORT}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --items_per_source "${ITEMS_PER_SOURCE:-256}" \
  --chunk_length "${CHUNK_LENGTH:-10}" \
  --batch "${BATCH:-2}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-64}" \
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-64}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_ID}}" \
  --wandb_group "${WANDB_GROUP:-object-motion-field-g0-v66}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
