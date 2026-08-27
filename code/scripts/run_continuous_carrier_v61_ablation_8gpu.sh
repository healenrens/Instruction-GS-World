#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
STEPS="${STEPS:-3000}"
SEED="${SEED:-17}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/v61_encoder_ablation_${SOURCE_REVISION:0:7}}"
VARIANTS=(dino siglip siglip_dino siglip_dino_object)
GPU_GROUPS=(0,1 2,3 4,5 6,7)

mkdir -p "${LOG_ROOT}"
for index in 0 1 2 3; do
  variant="${VARIANTS[$index]}"
  gpu_group="${GPU_GROUPS[$index]}"
  run_name="continuous_carrier_v61_${variant}_probe_seed${SEED}_${SOURCE_REVISION:0:7}"
  gate_report="${RUNTIME_ROOT}/outputs/v61_gates/${SOURCE_REVISION}_${variant}.json"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu_group}" \
    ROOT="${ROOT}" \
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    VENV_ROOT="${RUNTIME_ROOT}" \
    VARIANT="${variant}" \
    SOURCE_REVISION="${SOURCE_REVISION}" \
    GATE_REPORT="${gate_report}" \
    RUN_NAME="${run_name}" \
    OUT="${RUNTIME_ROOT}/outputs/${run_name}" \
    NPROC_PER_NODE=auto \
    BATCH_PER_GPU="${BATCH_PER_GPU:-16}" \
    TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}" \
    STEPS="${STEPS}" \
    SEED="${SEED}" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_GROUP="${WANDB_GROUP:-continuous-carrier-object-state-v61-ablation}" \
    bash "${ROOT}/code/scripts/train_continuous_carrier_object_state_v61.sh" \
    >"${LOG_ROOT}/${variant}.log" 2>&1 &
  echo "[v61-ablation] variant=${variant} gpu=${gpu_group} pid=$! log=${LOG_ROOT}/${variant}.log"
done
