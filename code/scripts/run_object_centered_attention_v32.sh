#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/readout_parent_snapshots/selected_parent.pt}"
BASELINE_REPORT="${BASELINE_REPORT:-${RUNTIME_ROOT}/outputs/v31_carrier_audits/object_centered_carrier_v31_heldseed_6727cb3.json}"
PY="${PY:-${RUNTIME_ROOT}/.venv/bin/python}"
SPLIT="${SPLIT:-heldseed}"
MAX_ITEMS="${MAX_ITEMS:-128}"
WORKERS="${WORKERS:-2}"
EXTRA_BUDGETS="${EXTRA_BUDGETS:-64,128,256}"
MINIMUM_UTILITY_FRACTION="${MINIMUM_UTILITY_FRACTION:-0.001}"
MAXIMUM_SCENE_FRACTION="${MAXIMUM_SCENE_FRACTION:-0.25}"
MINIMUM_COMPACT_RECOVERY="${MINIMUM_COMPACT_RECOVERY:-0.70}"
MAXIMUM_TOKEN_RATIO="${MAXIMUM_TOKEN_RATIO:-1.10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-3201}"

cd "${ROOT}"
COMMIT="$(git rev-parse HEAD)"
RUN_NAME="${RUN_NAME:-object_centered_attention_v32_${SPLIT}_${COMMIT:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/v32_attention_audits/${RUN_NAME}.json}"
LOG="${LOG:-${RUNTIME_ROOT}/logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

test -x "${PY}"
test -f "${DATA}/episode_manifest.json"
test -f "${CHECKPOINT}"
test -f "${BASELINE_REPORT}"
test -z "$(git status --porcelain --untracked-files=no)"

echo "[object-centered-attention-v32] commit=${COMMIT}"
echo "[object-centered-attention-v32] checkpoint=${CHECKPOINT}"
echo "[object-centered-attention-v32] baseline=${BASELINE_REPORT}"
echo "[object-centered-attention-v32] report=${OUT}"
echo "[object-centered-attention-v32] log=${LOG}"

"${PY}" "${ROOT}/code/scripts/audit_object_centered_attention_v32.py" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --baseline_report "${BASELINE_REPORT}" \
  --output "${OUT}" \
  --split "${SPLIT}" \
  --max_items "${MAX_ITEMS}" \
  --workers "${WORKERS}" \
  --extra_budgets "${EXTRA_BUDGETS}" \
  --minimum_utility_fraction "${MINIMUM_UTILITY_FRACTION}" \
  --maximum_scene_fraction "${MAXIMUM_SCENE_FRACTION}" \
  --minimum_compact_recovery "${MINIMUM_COMPACT_RECOVERY}" \
  --maximum_token_ratio "${MAXIMUM_TOKEN_RATIO}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}" 2>&1 | tee "${LOG}"

echo "[object-centered-attention-v32] completed report=${OUT}"
