#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
SEED="${SEED:-17}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/v61_encoder_ablation_eval_${SOURCE_REVISION:0:7}}"
VARIANTS=(dino siglip2 siglip2_dino siglip2_dino_object)

mkdir -p "${LOG_ROOT}"
for index in 0 1 2 3; do
  variant="${VARIANTS[$index]}"
  run_name="continuous_carrier_v61_${variant}_probe_seed${SEED}_${SOURCE_REVISION:0:7}"
  eval_name="${run_name}_held_object_state_eval"
  nohup env \
    CUDA_VISIBLE_DEVICES="${index}" \
    ROOT="${ROOT}" \
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    VENV_ROOT="${RUNTIME_ROOT}" \
    VARIANT="${variant}" \
    SOURCE_REVISION="${SOURCE_REVISION}" \
    RUN_NAME="${run_name}" \
    CHECKPOINT="${RUNTIME_ROOT}/outputs/${run_name}/latest.pt" \
    EVAL_NAME="${eval_name}" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_GROUP="${WANDB_GROUP:-continuous-carrier-object-state-v61-eval}" \
    bash "${ROOT}/code/scripts/evaluate_continuous_carrier_object_state_v61.sh" \
    >"${LOG_ROOT}/${variant}.log" 2>&1 &
  echo "[v61-state-eval] variant=${variant} gpu=${index} pid=$! log=${LOG_ROOT}/${variant}.log"
done
