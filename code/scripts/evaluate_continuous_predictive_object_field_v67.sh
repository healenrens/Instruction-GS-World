#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
STAGE="${STAGE:-predictive_state}"
if [ -f "${ROOT}/SOURCE_REVISION" ]; then
  read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"
else
  DEPLOYED_REVISION=local-unversioned
fi
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP_CHECKPOINT="${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT must point to a completed v67 stage checkpoint}"
RUN_NAME="${RUN_NAME:-continuous_predictive_object_field_v67_${STAGE}_held_eval_${SOURCE_REVISION:0:7}}"
OUTPUT="${OUTPUT:-${RUNTIME_ROOT}/outputs/v67_evaluations/${RUN_NAME}.json}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$(dirname "${OUTPUT}")" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"

EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-8}"

exec "${VENV_ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node "${EVAL_NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/evaluate_continuous_predictive_object_field_v67.py" \
  --stage "${STAGE}" \
  --checkpoint "${CHECKPOINT}" \
  --data_index "${DATA_INDEX}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${OUTPUT}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --items_per_source "${ITEMS_PER_SOURCE:-128}" \
  --batch "${EVAL_BATCH_PER_GPU:-4}" \
  --expected_world_size "${EVAL_NPROC_PER_NODE}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-96}" \
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-96}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" \
  --wandb_group "${WANDB_GROUP:-continuous-predictive-object-field-v67-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
