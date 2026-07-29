#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/readout_parent_snapshots/selected_parent.pt}"
PY="${PY:-${RUNTIME_ROOT}/.venv/bin/python}"
SPLIT="${SPLIT:-heldseed}"
MAX_ITEMS="${MAX_ITEMS:-128}"
WORKERS="${WORKERS:-2}"
EXTRA_BUDGETS="${EXTRA_BUDGETS:-8,16,32,64}"
MINIMUM_MARGINAL_GAIN="${MINIMUM_MARGINAL_GAIN:-0.00001}"
MINIMUM_GAP_RECOVERY="${MINIMUM_GAP_RECOVERY:-0.70}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-3101}"

cd "${ROOT}"
COMMIT="$(git rev-parse HEAD)"
RUN_NAME="${RUN_NAME:-object_centered_carrier_v31_${SPLIT}_${COMMIT:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/v31_carrier_audits/${RUN_NAME}.json}"
LOG="${LOG:-${RUNTIME_ROOT}/logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

test -x "${PY}"
test -f "${DATA}/episode_manifest.json"
test -f "${CHECKPOINT}"
test -z "$(git status --porcelain --untracked-files=no)"

echo "[object-centered-carrier-v31] commit=${COMMIT}"
echo "[object-centered-carrier-v31] checkpoint=${CHECKPOINT}"
echo "[object-centered-carrier-v31] report=${OUT}"
echo "[object-centered-carrier-v31] log=${LOG}"

"${PY}" "${ROOT}/code/scripts/audit_object_centered_carriers_v31.py" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUT}" \
  --split "${SPLIT}" \
  --max_items "${MAX_ITEMS}" \
  --workers "${WORKERS}" \
  --extra_budgets "${EXTRA_BUDGETS}" \
  --minimum_marginal_gain "${MINIMUM_MARGINAL_GAIN}" \
  --minimum_gap_recovery "${MINIMUM_GAP_RECOVERY}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}" 2>&1 | tee "${LOG}"

echo "[object-centered-carrier-v31] completed report=${OUT}"
