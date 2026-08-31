#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
RUN_ID="${RUN_ID:-v62c_codec_structure_$(date +%Y%m%d_%H%M%S)_$$}"
OUT_ROOT="${OUT_ROOT:-${RUNTIME_ROOT}/outputs/v62_parallel/c_codec_structure/${RUN_ID}}"
REPORT="${REPORT:-${OUT_ROOT}/object_codec_structural_eval.json}"
E0_CHECKPOINT="${E0_CHECKPOINT:-${RUNTIME_ROOT}/outputs/object_transition_v62_e0_seed17_f00082d/latest.pt}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

echo "[v62c] run_id=${RUN_ID} checkpoint=${E0_CHECKPOINT} report=${REPORT}"
cd "${ROOT}"
exec "${PY}" "${ROOT}/code/scripts/evaluate_object_codec_structure_v62c.py" \
  --checkpoint "${E0_CHECKPOINT}" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --dino_checkpoint "${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${REPORT}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --items_per_source "${ITEMS_PER_SOURCE:-32}" \
  --batch "${BATCH:-8}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-64}" \
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-64}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_ID}}" \
  --wandb_group "${WANDB_GROUP:-object-transition-v62c-codec-structure}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
