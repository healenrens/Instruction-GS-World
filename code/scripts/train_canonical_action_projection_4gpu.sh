#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
SOURCE="${SOURCE:-${ROOT}/outputs/canonical6_rgb_centergate0p1_zeromargin_v20_steps20_seed${SEED}_${RUN_DATE}/joint_0000005.pt}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"
JOINT_STEPS="${JOINT_STEPS:-20}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-5.0}"
LR="${LR:-1e-4}"
CANONICAL_ACTIVITY_GATE="${CANONICAL_ACTIVITY_GATE:-off}"
CANONICAL_ACTIVITY_POWER="${CANONICAL_ACTIVITY_POWER:-0.5}"
OUT="${OUT:-${ROOT}/outputs/canonical6_actionprojection_zeromargin_v24_steps${JOINT_STEPS}_seed${SEED}_${RUN_DATE}}"

ACTIVITY_ARGS=()
if [[ "${CANONICAL_ACTIVITY_GATE}" == "on" ]]; then
  ACTIVITY_ARGS+=(--canonical_activity_gate --canonical_activity_power "${CANONICAL_ACTIVITY_POWER}")
elif [[ "${CANONICAL_ACTIVITY_GATE}" != "off" ]]; then
  printf 'CANONICAL_ACTIVITY_GATE must be on or off\n' >&2
  exit 1
fi

if [[ ! -f "${SOURCE}" ]]; then
  printf 'Missing canonical checkpoint: %s\n' "${SOURCE}" >&2
  exit 1
fi
printf -v FINAL_CHECKPOINT '%s/joint_%07d.pt' "${OUT}" "${JOINT_STEPS}"
if [[ -e "${OUT}/latest.pt" || -e "${FINAL_CHECKPOINT}" ]]; then
  printf 'Refusing to overwrite action-projection run: %s\n' "${OUT}" >&2
  exit 1
fi

cd "${ROOT}"
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${ROOT}/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=4 \
    code/scripts/train_adaptive_gaussian_wm.py \
    --data "${DATA}" \
    --dino "${DINO}" \
    --condition_cache "${CONDITION_CACHE}" \
    --out "${OUT}" \
    --init_from "${SOURCE}" \
    --profile full \
    --representation_steps 0 \
    --joint_steps "${JOINT_STEPS}" \
    --batch 4 \
    --grad_accum 16 \
    --workers 2 \
    --max_train_items 384 \
    --lr "${LR}" \
    --lr_floor 1e-5 \
    --warmup_fraction 0.05 \
    --weight_decay 0.0 \
    --save_every 5 \
    --log_every 1 \
    --seed "${SEED}" \
    --amp bf16 \
    --language_condition auto \
    --rgb_supervision auto \
    --rgb_short_side 256 \
    --rgb_pad_multiple 16 \
    --rgb_render_chunk 8192 \
    --rgb_loss_weight 0.5 \
    --rgb_ssim_weight 0.2 \
    --language_effect_weight 0.0 \
    --zero_action_margin_weight "${MARGIN_WEIGHT}" \
    --zero_action_relative_margin 0.01 \
    --posterior_dynamics_gate \
    --posterior_update_scope action_projection \
    "${ACTIVITY_ARGS[@]}" \
    --action_anchor object_slot \
    --canonical_center_gate 0.1 \
    --action_residual_dim 0 \
    --semantic_action_basis rgb \
    2>&1 | tee "${OUT}/train.console.log"
