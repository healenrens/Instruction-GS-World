#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/object_video_v69_evaluation}"
ENCODER="${ENCODER:-dinov3_vitl16}"
if [[ "${ENCODER}" == vjepa2_1_vitl16 ]]; then
  ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${RUNTIME_ROOT}/third_party/vjepa2}"
  ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-${RUNTIME_ROOT}/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"
else
  ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${RUNTIME_ROOT}/third_party/dinov3}"
  ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT}"
"${VENV_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/evaluate_object_video_sequence_v69.py" \
  --manifest "${MANIFEST:-${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json}" --out "${OUT}" \
  --checkpoint "${CHECKPOINT}" --encoder_repository "${ENCODER_REPOSITORY}" --encoder_weights "${ENCODER_WEIGHTS}" \
  --encoder_frame_batch "${ENCODER_FRAME_BATCH:-2}" --items "${ITEMS:-400}" --visualize "${VISUALIZE:-40}" \
  --annotations "${ANNOTATIONS:-}" --seed "${SEED:-17}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_name "${RUN_NAME:-$(basename "${OUT}")}" \
  2>&1 | tee -a "${OUT}/evaluate.log"
