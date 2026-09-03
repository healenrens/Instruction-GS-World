#!/usr/bin/env bash

set -u

COMMAND="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
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
STATE_RUN="${STATE_RUN:-continuous_predictive_object_field_v67_state_seed17_${SOURCE_REVISION:0:7}}"
DYNAMICS_RUN="${DYNAMICS_RUN:-continuous_predictive_object_field_v67_dynamics_seed17_${SOURCE_REVISION:0:7}}"
STATE_OUT="${STATE_OUT:-${RUNTIME_ROOT}/outputs/${STATE_RUN}}"
DYNAMICS_OUT="${DYNAMICS_OUT:-${RUNTIME_ROOT}/outputs/${DYNAMICS_RUN}}"
STATE_CHECKPOINT="${STATE_CHECKPOINT:-${STATE_OUT}/latest.pt}"

export ROOT RUNTIME_ROOT VENV_ROOT SOURCE_REVISION DATA_INDEX
export DINO_CHECKPOINT SIGLIP_CHECKPOINT TRACKER_CHECKPOINT STATE_CHECKPOINT
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

verify_stage() {
  local stage="$1"
  local state_args=()
  if [ "${stage}" = posterior_dynamics ]; then
    state_args=(--state_checkpoint "${STATE_CHECKPOINT}")
  fi
  CUDA_VISIBLE_DEVICES="${VERIFY_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    "${VENV_ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node 4 \
      "${ROOT}/code/scripts/verify_continuous_predictive_object_field_v67.py" \
      --stage "${stage}" \
      --data_index "${DATA_INDEX}" \
      --dino_checkpoint "${DINO_CHECKPOINT}" \
      --siglip_checkpoint "${SIGLIP_CHECKPOINT}" \
      --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
      --source_revision "${SOURCE_REVISION}" \
      --expected_world_size 4 \
      --amp "${AMP:-bf16}" \
      "${state_args[@]}"
}

train_stage() {
  local stage="$1"
  local run="$2"
  local out="$3"
  export STAGE="${stage}" RUN_NAME="${run}" OUT="${out}"
  bash "${ROOT}/code/scripts/train_continuous_predictive_object_field_v67.sh"
}

evaluate_stage() {
  local stage="$1"
  local checkpoint="$2"
  local run="$3"
  export STAGE="${stage}" CHECKPOINT="${checkpoint}" RUN_NAME="${run}"
  bash "${ROOT}/code/scripts/evaluate_continuous_predictive_object_field_v67.sh"
}

status() {
  echo "[continuous-object-field-v67-manager] launch_mode=foreground revision=${SOURCE_REVISION}"
  echo "[continuous-object-field-v67-manager] state_out=${STATE_OUT} dynamics_out=${DYNAMICS_OUT}"
  if [ -f "${STATE_OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-40}" "${STATE_OUT}/train.jsonl"; fi
  if [ -f "${DYNAMICS_OUT}/train.jsonl" ]; then tail -n "${STATUS_LINES:-40}" "${DYNAMICS_OUT}/train.jsonl"; fi
}

case "${COMMAND}" in
  verify-state) verify_stage predictive_state ;;
  train-state) train_stage predictive_state "${STATE_RUN}" "${STATE_OUT}" ;;
  resume-state)
    export RESUME="${RESUME:-${STATE_OUT}/latest.pt}"
    train_stage predictive_state "${STATE_RUN}" "${STATE_OUT}"
    ;;
  evaluate-state)
    evaluate_stage predictive_state "${CHECKPOINT:-${STATE_CHECKPOINT}}" \
      "${EVAL_RUN_NAME:-${STATE_RUN}_held_eval}"
    ;;
  verify-dynamics) verify_stage posterior_dynamics ;;
  train-dynamics) train_stage posterior_dynamics "${DYNAMICS_RUN}" "${DYNAMICS_OUT}" ;;
  resume-dynamics)
    export RESUME="${RESUME:-${DYNAMICS_OUT}/latest.pt}"
    train_stage posterior_dynamics "${DYNAMICS_RUN}" "${DYNAMICS_OUT}"
    ;;
  evaluate-dynamics)
    evaluate_stage posterior_dynamics \
      "${CHECKPOINT:-${DYNAMICS_OUT}/latest.pt}" \
      "${EVAL_RUN_NAME:-${DYNAMICS_RUN}_held_eval}"
    ;;
  status) status ;;
  *) echo "usage: $0 {verify-state|train-state|resume-state|evaluate-state|verify-dynamics|train-dynamics|resume-dynamics|evaluate-dynamics|status}" ;;
esac
