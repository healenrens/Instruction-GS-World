#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260719}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"
GPUS=("${GPU0}" "${GPU1}" "${GPU2}")
RESIDUALS=(0 8 16)

cd "${ROOT}"
for index in "${!RESIDUALS[@]}"; do
  residual_dim="${RESIDUALS[${index}]}"
  gpu="${GPUS[${index}]}"
  run="${ROOT}/outputs/object_slot_residual_r${residual_dim}_v14_overfit256_seed${SEED}_${RUN_DATE}"
  checkpoint="${run}/joint_0000100.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    printf 'Missing checkpoint: %s\n' "${checkpoint}" >&2
    exit 1
  fi
  (
    for split in train heldseed heldtask; do
      CUDA_VISIBLE_DEVICES="${gpu}" \
        "${ROOT}/.venv/bin/python" \
          code/scripts/evaluate_posterior_dynamics_gate.py \
          --checkpoint "${checkpoint}" \
          --data "${DATA}" \
          --dino "${DINO}" \
          --condition_cache "${CONDITION_CACHE}" \
          --split "${split}" \
          --max_items 144 \
          --batch 4 \
          --workers 2 \
          --amp bf16 \
          --output "${run}/posterior_gate_step100_${split}.json"
    done
    for split in heldseed heldtask; do
      CUDA_VISIBLE_DEVICES="${gpu}" \
        "${ROOT}/.venv/bin/python" \
          code/scripts/evaluate_action_component_ablation.py \
          --checkpoint "${checkpoint}" \
          --data "${DATA}" \
          --dino "${DINO}" \
          --condition_cache "${CONDITION_CACHE}" \
          --split "${split}" \
          --max_items 64 \
          --batch 4 \
          --workers 2 \
          --amp bf16 \
          --output "${run}/action_component_ablation_${split}.json"
    done
  ) > "${run}/posterior_gate_all.log" 2>&1 &
done
wait

"${ROOT}/.venv/bin/python" \
  code/scripts/summarize_action_residual_sweep.py \
  --run_template \
  "${ROOT}/outputs/object_slot_residual_r{residual_dim}_v14_overfit256_seed${SEED}_${RUN_DATE}" \
  --checkpoint_step 100 \
  --output \
  "${ROOT}/outputs/action_residual_sweep_v14_seed${SEED}_${RUN_DATE}_summary.json"
