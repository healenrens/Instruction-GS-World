#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
STEPS="${STEPS:-3000}"
SEED="${SEED:-17}"
CAPACITIES=(4x32_global 4x32_bound 8x64_bound 16x32_root)

for capacity in "${CAPACITIES[@]}"; do
  run_name="continuous_carrier_dynamics_v61_${capacity}_probe_seed${SEED}_${SOURCE_REVISION:0:7}"
  gate_report="${RUNTIME_ROOT}/outputs/v61_dynamics_gates/${SOURCE_REVISION}_${capacity}.json"
  echo "[v61-effect-ablation] starting capacity=${capacity} mode=foreground"
  env \
    ROOT="${ROOT}" \
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    VENV_ROOT="${RUNTIME_ROOT}" \
    VARIANT="${VARIANT:-siglip_dino_object}" \
    EFFECT_CAPACITY="${capacity}" \
    SOURCE_REVISION="${SOURCE_REVISION}" \
    STATE_CHECKPOINT="${STATE_CHECKPOINT}" \
    GATE_REPORT="${gate_report}" \
    RUN_NAME="${run_name}" \
    OUT="${RUNTIME_ROOT}/outputs/${run_name}" \
    NPROC_PER_NODE="${NPROC_PER_NODE:-auto}" \
    BATCH_PER_GPU="${BATCH_PER_GPU:-16}" \
    TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}" \
    STEPS="${STEPS}" \
    SEED="${SEED}" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_GROUP="${WANDB_GROUP:-continuous-carrier-dynamics-v61-ablation}" \
    bash "${ROOT}/code/scripts/train_continuous_carrier_dynamics_v61.sh"
  echo "[v61-effect-ablation] completed capacity=${capacity}"
done
