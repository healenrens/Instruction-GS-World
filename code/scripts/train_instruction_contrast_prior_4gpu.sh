#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
SOURCE="${SOURCE:-${ROOT}/outputs/prior_canonical6_activitypower1_v24_steps100_seed17_${RUN_DATE}/prior_0000040.pt}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"
STEPS="${STEPS:-60}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-384}"
LR="${LR:-5e-5}"
EFFECT_WEIGHT="${EFFECT_WEIGHT:-0.5}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.5}"
RANK_WEIGHT="${RANK_WEIGHT:-0.5}"
RELATIVE_MARGIN="${RELATIVE_MARGIN:-0.05}"
ACTIVITY_FLOOR="${ACTIVITY_FLOOR:-0.1}"
TRAIN_LANGUAGE_PROJECTOR="${TRAIN_LANGUAGE_PROJECTOR:-0}"
TOKEN_CONDITIONER_ONLY="${TOKEN_CONDITIONER_ONLY:-0}"
FREEZE_TOKEN_CONDITIONER="${FREEZE_TOKEN_CONDITIONER:-0}"
DYNAMICS_RELATIVE_OBJECTIVE="${DYNAMICS_RELATIVE_OBJECTIVE:-0}"
PARAPHRASE_WEIGHT="${PARAPHRASE_WEIGHT:-0.0}"
TASK_CONTRAST_WEIGHT="${TASK_CONTRAST_WEIGHT:-0.0}"
SAVE_EVERY="${SAVE_EVERY:-20}"
OUT="${OUT:-${ROOT}/outputs/prior_canonical6_activitypower1_instructioncontrast_a${ANCHOR_WEIGHT}_r${RANK_WEIGHT}_m${RELATIVE_MARGIN}_from40_steps${STEPS}_seed${SEED}_${RUN_DATE}}"

if [[ ! -f "${SOURCE}" ]]; then
  printf 'Missing Prior warm start: %s\n' "${SOURCE}" >&2
  exit 1
fi
if [[ "${TRAIN_LANGUAGE_PROJECTOR}" != 0 && "${TRAIN_LANGUAGE_PROJECTOR}" != 1 ]]; then
  printf 'TRAIN_LANGUAGE_PROJECTOR must be 0 or 1\n' >&2
  exit 1
fi
if [[ "${TOKEN_CONDITIONER_ONLY}" != 0 && "${TOKEN_CONDITIONER_ONLY}" != 1 ]]; then
  printf 'TOKEN_CONDITIONER_ONLY must be 0 or 1\n' >&2
  exit 1
fi
if [[ "${FREEZE_TOKEN_CONDITIONER}" != 0 && "${FREEZE_TOKEN_CONDITIONER}" != 1 ]]; then
  printf 'FREEZE_TOKEN_CONDITIONER must be 0 or 1\n' >&2
  exit 1
fi
if [[ "${DYNAMICS_RELATIVE_OBJECTIVE}" != 0 && "${DYNAMICS_RELATIVE_OBJECTIVE}" != 1 ]]; then
  printf 'DYNAMICS_RELATIVE_OBJECTIVE must be 0 or 1\n' >&2
  exit 1
fi
language_projector_args=()
if [[ "${TRAIN_LANGUAGE_PROJECTOR}" == 1 ]]; then
  language_projector_args=(--train_language_projector)
fi
token_conditioner_args=()
if [[ "${TOKEN_CONDITIONER_ONLY}" == 1 ]]; then
  token_conditioner_args=(--token_conditioner_only)
fi
freeze_token_conditioner_args=()
if [[ "${FREEZE_TOKEN_CONDITIONER}" == 1 ]]; then
  freeze_token_conditioner_args=(--freeze_token_conditioner)
fi
dynamics_relative_args=()
if [[ "${DYNAMICS_RELATIVE_OBJECTIVE}" == 1 ]]; then
  dynamics_relative_args=(--dynamics_relative_objective)
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
    --lr_floor 5e-6 \
    --warmup_fraction 0.05 \
    --weight_decay 1e-4 \
    --effect_weight "${EFFECT_WEIGHT}" \
    --instruction_anchor_weight "${ANCHOR_WEIGHT}" \
    --instruction_rank_weight "${RANK_WEIGHT}" \
    --instruction_relative_margin "${RELATIVE_MARGIN}" \
    --action_activity_floor "${ACTIVITY_FLOOR}" \
    --paraphrase_positive_weight "${PARAPHRASE_WEIGHT}" \
    --task_semantic_contrast_weight "${TASK_CONTRAST_WEIGHT}" \
    "${language_projector_args[@]}" \
    "${token_conditioner_args[@]}" \
    "${freeze_token_conditioner_args[@]}" \
    "${dynamics_relative_args[@]}" \
    --save_every "${SAVE_EVERY}" \
    --log_every 1 \
    --seed "${SEED}" \
    --amp bf16 \
    2>&1 | tee "${OUT}/train.console.log"
