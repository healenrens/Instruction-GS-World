#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
if [ -z "${SOURCE_REVISION:-}" ]; then
  if [ -f "${ROOT}/SOURCE_REVISION" ]; then
    read -r SOURCE_REVISION < "${ROOT}/SOURCE_REVISION"
  else
    SOURCE_REVISION="$(git -C "${ROOT}" rev-parse HEAD)"
  fi
fi
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP_CHECKPOINT="${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
E0_STAGE=teacher_object_codec
E1_STAGE=teacher_transition_oracle
E0_RUN="${E0_RUN:-object_transition_v62_e0_seed17_${SOURCE_REVISION:0:7}}"
E1_RUN="${E1_RUN:-object_transition_v62_e1_seed17_${SOURCE_REVISION:0:7}}"
E0_OUT="${E0_OUT:-${RUNTIME_ROOT}/outputs/${E0_RUN}}"
E1_OUT="${E1_OUT:-${RUNTIME_ROOT}/outputs/${E1_RUN}}"
CODEC_CHECKPOINT="${CODEC_CHECKPOINT:-${E0_OUT}/latest.pt}"
E0_GATE="${E0_GATE:-${RUNTIME_ROOT}/outputs/v62_gates/${SOURCE_REVISION}_${E0_STAGE}.json}"
E1_GATE="${E1_GATE:-${RUNTIME_ROOT}/outputs/v62_gates/${SOURCE_REVISION}_${E1_STAGE}.json}"

export ROOT RUNTIME_ROOT VENV_ROOT SOURCE_REVISION DATA_INDEX
export DINO_CHECKPOINT SIGLIP_CHECKPOINT TRACKER_CHECKPOINT CODEC_CHECKPOINT
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

verify_stage() {
  local stage="$1"
  local report="$2"
  local codec_args=()
  if [ "${stage}" = "${E1_STAGE}" ]; then
    codec_args=(--codec_checkpoint "${CODEC_CHECKPOINT}")
  fi
  mkdir -p "$(dirname "${report}")"
  CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-0}" \
    "${PY}" "${ROOT}/code/scripts/verify_object_transition_v62.py" \
      --stage "${stage}" \
      --data_index "${DATA_INDEX}" \
      --dino_checkpoint "${DINO_CHECKPOINT}" \
      --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
      --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
      --source_revision "${SOURCE_REVISION}" \
      --output "${report}" \
      --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
      --batch "${VERIFY_BATCH:-12}" \
      --dino_frame_batch "${VERIFY_DINO_FRAME_BATCH:-16}" \
      --siglip_frame_batch "${VERIFY_SIGLIP_FRAME_BATCH:-16}" \
      --amp "${AMP:-bf16}" \
      "${codec_args[@]}"
}

train_stage() {
  local stage="$1"
  local report="$2"
  local run="$3"
  local out="$4"
  export STAGE="${stage}" GATE_REPORT="${report}" RUN_NAME="${run}" OUT="${out}"
  if [ "${stage}" = "${E1_STAGE}" ]; then export CODEC_CHECKPOINT; fi
  bash "${ROOT}/code/scripts/train_object_transition_v62.sh"
}

evaluate_stage() {
  local stage="$1"
  local checkpoint="$2"
  local name="$3"
  local output="${RUNTIME_ROOT}/outputs/v62_evaluations/${name}.json"
  mkdir -p "$(dirname "${output}")"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}" \
    "${PY}" "${ROOT}/code/scripts/evaluate_object_transition_v62.py" \
      --stage "${stage}" \
      --checkpoint "${checkpoint}" \
      --data_index "${DATA_INDEX}" \
      --dino_checkpoint "${DINO_CHECKPOINT}" \
      --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
      --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
      --output "${output}" \
      --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
      --items_per_source "${EVAL_ITEMS_PER_SOURCE:-32}" \
      --batch "${EVAL_BATCH:-4}" \
      --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-16}" \
      --siglip_frame_batch "${EVAL_SIGLIP_FRAME_BATCH:-16}" \
      --amp "${AMP:-bf16}" \
      --wandb_mode "${WANDB_MODE:-online}" \
      --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
      --wandb_entity "${WANDB_ENTITY:-}" \
      --wandb_name "${name}" \
      --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
}

status() {
  echo "[object-transition-v62-manager] launch_mode=foreground revision=${SOURCE_REVISION}"
  echo "[object-transition-v62-manager] e0_out=${E0_OUT} e1_out=${E1_OUT}"
  if [ -f "${E0_OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-40}" "${E0_OUT}/train.jsonl"; fi
  if [ -f "${E1_OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-40}" "${E1_OUT}/train.jsonl"; fi
}

case "${COMMAND}" in
  verify-e0) verify_stage "${E0_STAGE}" "${E0_GATE}" ;;
  verify-e1) verify_stage "${E1_STAGE}" "${E1_GATE}" ;;
  train-e0) train_stage "${E0_STAGE}" "${E0_GATE}" "${E0_RUN}" "${E0_OUT}" ;;
  train-e1) train_stage "${E1_STAGE}" "${E1_GATE}" "${E1_RUN}" "${E1_OUT}" ;;
  resume-e0)
    export RESUME="${RESUME:-${E0_OUT}/latest.pt}"
    train_stage "${E0_STAGE}" "${E0_GATE}" "${E0_RUN}" "${E0_OUT}"
    ;;
  resume-e1)
    export RESUME="${RESUME:-${E1_OUT}/latest.pt}"
    train_stage "${E1_STAGE}" "${E1_GATE}" "${E1_RUN}" "${E1_OUT}"
    ;;
  eval-e0) evaluate_stage "${E0_STAGE}" "${E0_CHECKPOINT:-${E0_OUT}/latest.pt}" "${E0_RUN}_held" ;;
  eval-e1) evaluate_stage "${E1_STAGE}" "${E1_CHECKPOINT:-${E1_OUT}/latest.pt}" "${E1_RUN}_held" ;;
  status) status ;;
  *) echo "usage: $0 {verify-e0|train-e0|resume-e0|eval-e0|verify-e1|train-e1|resume-e1|eval-e1|status}" ;;
esac
