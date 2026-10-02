#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
SOURCE_REVISION="$(<"${ROOT}/SOURCE_REVISION")"
STATE_RUN="${RUNTIME_ROOT}/outputs/object_video_v69_state_seed17_40e4586_20260929_003931"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_single_gpu_${SOURCE_REVISION:0:7}_$(date +%Y%m%d_%H%M%S)}"
STATE_CHECKPOINT="${STATE_CHECKPOINT:-${RUNTIME_ROOT}/outputs/object_video_v69_state_change_held_aed14bb_20261001_235841/checkpoint_snapshot.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
export CUDA_VISIBLE_DEVICES="${TEST_GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT}"
echo "[object-video-stage2-v69-test] full_model=true stage=dynamics preset=large device=cuda:0 state=${STATE_CHECKPOINT} log=${OUT}/test.log"
"${PY}" "${ROOT}/code/scripts/test_object_video_stage2_v69.py" \
  --manifest "${MANIFEST:-${STATE_RUN}/dataset.json}" --out "${OUT}" \
  --encoder "${ENCODER:-dinov3_vitl16}" \
  --encoder_repository "${ENCODER_REPOSITORY:-${STATE_RUN}/encoder_source}" \
  --encoder_weights "${ENCODER_WEIGHTS:-${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --encoder_frame_batch "${ENCODER_FRAME_BATCH:-2}" --stage dynamics --stage2_preset large --dynamics_checkpoint_blocks \
  --state_checkpoint "${STATE_CHECKPOINT}" --config "${MODEL_CONFIG}" --seed "${SEED:-17}" \
  --posterior_geometry "${POSTERIOR_GEOMETRY:-inherit}" \
  --source_revision "${SOURCE_REVISION}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_name "${RUN_NAME:-$(basename "${OUT}")}" \
  2>&1 | tee -a "${OUT}/test.log"
