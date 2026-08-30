#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"

DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP_CHECKPOINT="${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"

SIGLIP_DINO_CHECKPOINT="${SIGLIP_DINO_CHECKPOINT:-${RUNTIME_ROOT}/outputs/continuous_carrier_v61_siglip_dino_probe_seed17_82bbd8d_20260829_182236/latest.pt}"
SIGLIP_DINO_OBJECT_CHECKPOINT="${SIGLIP_DINO_OBJECT_CHECKPOINT:-${RUNTIME_ROOT}/outputs/continuous_carrier_v61_siglip_dino_object_probe_seed17_82bbd8d_20260829_182236/latest.pt}"

EVALUATOR_REVISION="${EVALUATOR_REVISION:-}"
EVAL_NAME="${EVAL_NAME:-v61_siglip_dino_vs_object_representation_sufficiency_${EVALUATOR_REVISION:0:7}}"
OUTPUT="${OUTPUT:-${RUNTIME_ROOT}/outputs/v61_representation_sufficiency/${EVAL_NAME}.json}"

export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUTPUT}")" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"

exec "${PY}" "${ROOT}/code/scripts/evaluate_representation_sufficiency_v61.py" \
  --checkpoint "siglip_dino=${SIGLIP_DINO_CHECKPOINT}" \
  --checkpoint "siglip_dino_object=${SIGLIP_DINO_OBJECT_CHECKPOINT}" \
  --data_index "${DATA_INDEX}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --chunk_lengths "${EVAL_CHUNK_LENGTHS:-4,6,8}" \
  --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --conditions_per_source "${CONDITIONS_PER_SOURCE:-8}" \
  --student_frame_batch "${EVAL_STUDENT_FRAME_BATCH:-32}" \
  --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-32}" \
  --siglip_teacher_batch "${EVAL_SIGLIP_BATCH:-32}" \
  --probe_sample_limit "${PROBE_SAMPLE_LIMIT:-16384}" \
  --probe_mlp_steps "${PROBE_MLP_STEPS:-100}" \
  --active_variance_floor "${ACTIVE_VARIANCE_FLOOR:-1e-4}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${EVAL_NAME}" \
  --wandb_group "${WANDB_GROUP:-v61-representation-sufficiency-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
  --evaluator_revision "${EVALUATOR_REVISION}"
