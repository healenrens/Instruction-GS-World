#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
if [ -z "${SOURCE_REVISION:-}" ]; then
  if [ -f "${ROOT}/SOURCE_REVISION" ]; then
    read -r SOURCE_REVISION < "${ROOT}/SOURCE_REVISION"
  else
    SOURCE_REVISION="$(git -C "${ROOT}" rev-parse HEAD)"
  fi
fi
RUN_ID="${RUN_ID:-v64_object_bound_teacher_$(date +%Y%m%d_%H%M%S)_$$}"
OUT_ROOT="${OUT_ROOT:-${RUNTIME_ROOT}/outputs/v64_object_bound_teacher/${RUN_ID}}"
REPORT="${REPORT:-${OUT_ROOT}/object_bound_transition_audit.json}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

echo "[v64-object-bound-teacher] run_id=${RUN_ID} world=8 report=${REPORT}"
cd "${ROOT}"
exec "${TORCHRUN}" --standalone --nproc_per_node 8 \
  "${ROOT}/code/scripts/audit_object_bound_transition_teacher_v64.py" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --dino_checkpoint "${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${REPORT}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --items_per_source "${ITEMS_PER_SOURCE:-32}" \
  --chunk_length "${CHUNK_LENGTH:-10}" \
  --batch "${BATCH:-8}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-64}" \
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-64}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_ID}}" \
  --wandb_group "${WANDB_GROUP:-object-transition-v64-object-bound-teacher}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
