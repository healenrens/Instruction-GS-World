#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
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
CHECKPOINT="${CHECKPOINT:?CHECKPOINT must point to the trained v67 predictive-state checkpoint}"
RUN_NAME="${RUN_NAME:-continuous_predictive_object_field_v67_contract_audit_${SOURCE_REVISION:0:7}}"
OUTPUT="${OUTPUT:-${RUNTIME_ROOT}/outputs/v67_contract_audits/${RUN_NAME}.json}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${RUNTIME_ROOT}/outputs/v67_contract_audits/${RUN_NAME}_evidence}"
AUDIT_NPROC_PER_NODE="${AUDIT_NPROC_PER_NODE:-8}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$(dirname "${OUTPUT}")" "${ARTIFACT_DIR}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"

exec "${VENV_ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node "${AUDIT_NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/audit_continuous_predictive_contracts_v67.py" \
  --checkpoint "${CHECKPOINT}" \
  --data_index "${DATA_INDEX}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${OUTPUT}" \
  --artifact_dir "${ARTIFACT_DIR}" \
  --annotation_jsonl "${ANNOTATION_JSONL:-}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --items_per_source "${ITEMS_PER_SOURCE:-256}" \
  --review_cases_per_source "${REVIEW_CASES_PER_SOURCE:-12}" \
  --batch "${AUDIT_BATCH_PER_GPU:-2}" \
  --expected_world_size "${AUDIT_NPROC_PER_NODE}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-96}" \
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-96}" \
  --amp "${AMP:-bf16}" \
  --motion_sigmas "${MOTION_SIGMAS:-0.02,0.04,0.08,0.16,0.32}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" \
  --wandb_group "${WANDB_GROUP:-continuous-predictive-object-field-v67-contract-audit}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
