#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
SEED="${SEED:-17}"
VARIANTS=(dino siglip siglip_dino siglip_dino_object)

for variant in "${VARIANTS[@]}"; do
  run_name="continuous_carrier_v61_${variant}_probe_seed${SEED}_${SOURCE_REVISION:0:7}"
  eval_name="${run_name}_held_object_state_eval"
  echo "[v61-state-eval] starting variant=${variant} mode=foreground"
  env \
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
    bash "${ROOT}/code/scripts/evaluate_continuous_carrier_object_state_v61.sh"
  echo "[v61-state-eval] completed variant=${variant}"
done
