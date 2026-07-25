#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260719}"
SOURCE="${SOURCE:-${ROOT}/outputs/object_slot_anchor_v13b_overfit256_seed17_20260719/joint_0000100.pt}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"

if [[ ! -f "${SOURCE}" ]]; then
  printf 'Missing source checkpoint: %s\n' "${SOURCE}" >&2
  exit 1
fi

cd "${ROOT}"
for residual_dim in 0 8 16; do
  out="${ROOT}/outputs/object_slot_residual_r${residual_dim}_v14_overfit256_seed${SEED}_${RUN_DATE}"
  if [[ -e "${out}/latest.pt" || -e "${out}/joint_0000100.pt" ]]; then
    printf 'Refusing to overwrite existing run: %s\n' "${out}" >&2
    exit 1
  fi
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${ROOT}/.venv/bin/torchrun" \
      --standalone \
      --nproc_per_node=4 \
      code/scripts/train_adaptive_gaussian_wm.py \
      --data "${DATA}" \
      --dino "${DINO}" \
      --condition_cache "${CONDITION_CACHE}" \
      --out "${out}" \
      --init_from "${SOURCE}" \
      --profile full \
      --representation_steps 0 \
      --joint_steps 100 \
      --batch 4 \
      --grad_accum 16 \
      --workers 2 \
      --max_train_items 256 \
      --lr 2e-4 \
      --lr_floor 2e-5 \
      --warmup_fraction 0.05 \
      --weight_decay 1e-4 \
      --save_every 50 \
      --log_every 5 \
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
      --posterior_dynamics_gate \
      --action_anchor object_slot \
      --action_residual_dim "${residual_dim}" \
      2>&1 | tee "${out}/train.console.log"
done
