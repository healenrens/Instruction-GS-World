#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_IDS="${GPU_IDS:-0,1}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260719}"
SUMMARY="${SUMMARY:-${ROOT}/outputs/action_residual_sweep_v14_seed${SEED}_${RUN_DATE}_summary.json}"
DATA="${DATA:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1}"
DINO="${DINO:-${ROOT}/data/rt2_causal_pairs_e2e_20260717_v1_dino32}"
CONDITION_CACHE="${CONDITION_CACHE:-${ROOT}/data/rt2_instruction_qwen_text_20260719_v1.pt}"

if [[ ! -f "${SUMMARY}" ]]; then
  printf 'Missing residual sweep summary: %s\n' "${SUMMARY}" >&2
  exit 1
fi
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
source="${ROOT}/outputs/object_slot_residual_r${residual_dim}_v14_overfit256_seed${SEED}_${RUN_DATE}/joint_0000100.pt"
out="${ROOT}/outputs/prior_object_slot_residual_r${residual_dim}_v14_b16a8_100_seed${SEED}_${RUN_DATE}"
if [[ ! -f "${source}" ]]; then
  printf 'Missing selected Dynamics checkpoint: %s\n' "${source}" >&2
  exit 1
fi
if [[ -e "${out}/latest.pt" || -e "${out}/prior_0000100.pt" ]]; then
  printf 'Refusing to overwrite existing Prior run: %s\n' "${out}" >&2
  exit 1
fi

cd "${ROOT}"
mkdir -p "${out}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${ROOT}/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=2 \
    code/scripts/train_adaptive_gaussian_prior.py \
    --checkpoint "${source}" \
    --data "${DATA}" \
    --dino "${DINO}" \
    --condition_cache "${CONDITION_CACHE}" \
    --out "${out}" \
    --steps 100 \
    --batch 16 \
    --grad_accum 8 \
    --workers 2 \
    --max_train_items 256 \
    --lr 1e-4 \
    --lr_floor 1e-5 \
    --warmup_fraction 0.05 \
    --weight_decay 1e-4 \
    --effect_weight 0.5 \
    --save_every 50 \
    --log_every 5 \
    --seed "${SEED}" \
    --amp bf16 \
    2>&1 | tee "${out}/train.console.log"
