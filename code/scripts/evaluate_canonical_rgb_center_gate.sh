#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
RUN="${RUN:-${ROOT}/outputs/object_slot_canonical6_rgb_centergate0p1_v19_all384_steps150_seed${SEED}_${RUN_DATE}}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-50}"
POSTERIOR_ITEMS="${POSTERIOR_ITEMS:-144}"
COMPONENT_ITEMS="${COMPONENT_ITEMS:-64}"
printf -v CHECKPOINT '%s/joint_%07d.pt' "${RUN}" "${CHECKPOINT_STEP}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'Missing canonical-only checkpoint: %s\n' "${CHECKPOINT}" >&2
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
      --max_items "${POSTERIOR_ITEMS}" \
      --batch 4 \
      --workers 2 \
      --amp bf16 \
      --output "${RUN}/posterior_gate_step${CHECKPOINT_STEP}_${split}.json"
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
      --max_items "${COMPONENT_ITEMS}" \
      --batch 4 \
      --workers 2 \
      --amp bf16 \
      --output "${RUN}/action_component_ablation_step${CHECKPOINT_STEP}_${split}.json"
done
"${ROOT}/.venv/bin/python" \
  code/scripts/summarize_single_action_layout.py \
  --run "${RUN}" \
  --residual_dim 0 \
  --canonical_center_gate 0.1 \
  --residual_gate 1.0 \
  --residual_dropout 0.0 \
  --semantic_action_basis rgb \
  --checkpoint_step "${CHECKPOINT_STEP}" \
  --output "${RUN}/action_layout_gate_summary_step${CHECKPOINT_STEP}.json"
