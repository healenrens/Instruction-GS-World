#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
SOURCE="${SOURCE:-${ROOT}/outputs/canonical6_rgb_centergate0p1_activitypower1_v24_converted_seed17_${RUN_DATE}/joint_0000000.pt}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"
STEPS="${STEPS:-100}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-384}"
LR="${LR:-1e-4}"
EFFECT_WEIGHT="${EFFECT_WEIGHT:-0.5}"
SAVE_EVERY="${SAVE_EVERY:-20}"
OUT="${OUT:-${ROOT}/outputs/prior_canonical6_activitypower1_v24_steps${STEPS}_seed${SEED}_${RUN_DATE}}"

if [[ ! -f "${SOURCE}" ]]; then
  printf 'Missing canonical checkpoint: %s\n' "${SOURCE}" >&2
  exit 1
fi
printf -v FINAL_CHECKPOINT '%s/prior_%07d.pt' "${OUT}" "${STEPS}"
if [[ -e "${OUT}/latest.pt" || -e "${FINAL_CHECKPOINT}" ]]; then
  printf 'Refusing to overwrite Prior run: %s\n' "${OUT}" >&2
  exit 1
fi

cd "${ROOT}"
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${ROOT}/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=4 \
    code/scripts/train_adaptive_gaussian_prior.py \
    --checkpoint "${SOURCE}" \
    --data "${DATA}" \
    --dino "${DINO}" \
    --condition_cache "${CONDITION_CACHE}" \
    --out "${OUT}" \
    --steps "${STEPS}" \
    --batch "${BATCH}" \
    --grad_accum "${GRAD_ACCUM}" \
    --workers 2 \
    --max_train_items "${MAX_TRAIN_ITEMS}" \
    --lr "${LR}" \
    --lr_floor 1e-5 \
    --warmup_fraction 0.05 \
    --weight_decay 1e-4 \
    --effect_weight "${EFFECT_WEIGHT}" \
    --save_every "${SAVE_EVERY}" \
    --log_every 1 \
    --seed "${SEED}" \
    --amp bf16 \
    2>&1 | tee "${OUT}/train.console.log"
