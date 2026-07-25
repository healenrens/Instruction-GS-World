#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260719}"
RESIDUAL_DIM="${RESIDUAL_DIM:-8}"
RESIDUAL_GATE="${RESIDUAL_GATE:-0.1}"
RESIDUAL_DROPOUT="${RESIDUAL_DROPOUT:-0.75}"
GATE_TAG="${GATE_TAG:-0p1}"
DROPOUT_TAG="${DROPOUT_TAG:-0p75}"
RUN="${RUN:-${ROOT}/outputs/object_slot_residual_r${RESIDUAL_DIM}_gate${GATE_TAG}_drop${DROPOUT_TAG}_v16_overfit256_seed${SEED}_${RUN_DATE}}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"
CHECKPOINT="${RUN}/joint_0000100.pt"

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'Missing dropout checkpoint: %s\n' "${CHECKPOINT}" >&2
  exit 1
fi
cd "${ROOT}"
for split in train heldseed heldtask; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${ROOT}/.venv/bin/python" \
      code/scripts/evaluate_posterior_dynamics_gate.py \
      --checkpoint "${CHECKPOINT}" \
      --data "${DATA}" \
      --dino "${DINO}" \
      --condition_cache "${CONDITION_CACHE}" \
      --split "${split}" \
      --max_items 144 \
      --batch 4 \
      --workers 2 \
      --amp bf16 \
      --output "${RUN}/posterior_gate_step100_${split}.json"
done
for split in heldseed heldtask; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${ROOT}/.venv/bin/python" \
      code/scripts/evaluate_action_component_ablation.py \
      --checkpoint "${CHECKPOINT}" \
      --data "${DATA}" \
      --dino "${DINO}" \
      --condition_cache "${CONDITION_CACHE}" \
      --split "${split}" \
      --max_items 64 \
      --batch 4 \
      --workers 2 \
      --amp bf16 \
      --output "${RUN}/action_component_ablation_${split}.json"
done
"${ROOT}/.venv/bin/python" \
  code/scripts/summarize_single_action_layout.py \
  --run "${RUN}" \
  --residual_dim "${RESIDUAL_DIM}" \
  --residual_gate "${RESIDUAL_GATE}" \
  --residual_dropout "${RESIDUAL_DROPOUT}" \
  --checkpoint_step 100 \
  --output "${RUN}/action_layout_gate_summary.json"
