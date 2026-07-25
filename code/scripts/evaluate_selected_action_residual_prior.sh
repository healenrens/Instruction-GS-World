#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260719}"
SUMMARY="${SUMMARY:-${ROOT}/outputs/action_residual_sweep_v14_seed${SEED}_${RUN_DATE}_summary.json}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"

residual_dim="$(
  "${ROOT}/.venv/bin/python" - "${SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
selected = summary.get("selected_for_prior")
if selected is None:
    raise SystemExit("No residual configuration passed both held gates")
print(selected)
PY
)"
run="${ROOT}/outputs/prior_object_slot_residual_r${residual_dim}_v14_b16a8_100_seed${SEED}_${RUN_DATE}"
checkpoint="${run}/prior_0000100.pt"
if [[ ! -f "${checkpoint}" ]]; then
  printf 'Missing selected Prior checkpoint: %s\n' "${checkpoint}" >&2
  exit 1
fi

cd "${ROOT}"
for split in train heldseed heldtask; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${ROOT}/.venv/bin/python" \
      code/scripts/evaluate_adaptive_gaussian_prior_gate.py \
      --checkpoint "${checkpoint}" \
      --data "${DATA}" \
      --dino "${DINO}" \
      --condition_cache "${CONDITION_CACHE}" \
      --split "${split}" \
      --max_items 64 \
      --batch 4 \
      --workers 2 \
      --amp bf16 \
      --output "${run}/prior_gate_step100_${split}.json"
done
