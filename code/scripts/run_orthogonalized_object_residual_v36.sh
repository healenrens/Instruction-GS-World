#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/readout_parent_snapshots/selected_parent.pt}"
V35_REPORT="${V35_REPORT:-${RUNTIME_ROOT}/outputs/v35_partition_audits/partitioned_object_field_v35_heldseed_4d5f6e1.json}"
PY="${PY:-${RUNTIME_ROOT}/.venv/bin/python}"
SPLIT="${SPLIT:-heldseed}"
MAX_ITEMS="${MAX_ITEMS:-128}"
WORKERS="${WORKERS:-2}"
BUDGETS="${BUDGETS:-64,128,256}"
COMPACT_BUDGET="${COMPACT_BUDGET:-128}"
RIDGE="${RIDGE:-0.0001}"
RELATIVE_SINGULAR_CUTOFF="${RELATIVE_SINGULAR_CUTOFF:-0.001}"
MINIMUM_COLUMN_FRACTION="${MINIMUM_COLUMN_FRACTION:-0.0001}"
MINIMUM_UTILITY_FRACTION="${MINIMUM_UTILITY_FRACTION:-0.001}"
MAXIMUM_SCENE_FRACTION="${MAXIMUM_SCENE_FRACTION:-0.25}"
MINIMUM_COMPACT_RECOVERY="${MINIMUM_COMPACT_RECOVERY:-0.70}"
MAXIMUM_TOKEN_RATIO="${MAXIMUM_TOKEN_RATIO:-1.10}"
MAXIMUM_COEFFICIENT_RMS="${MAXIMUM_COEFFICIENT_RMS:-50.0}"
MAXIMUM_CONDITION_NUMBER="${MAXIMUM_CONDITION_NUMBER:-2000.0}"
MINIMUM_RETAINED_ENERGY="${MINIMUM_RETAINED_ENERGY:-0.99}"
MAXIMUM_COLUMN_AMPLIFICATION="${MAXIMUM_COLUMN_AMPLIFICATION:-2.0}"
MAXIMUM_TRANSPORT_RATIO="${MAXIMUM_TRANSPORT_RATIO:-1.0}"
MAXIMUM_DYNAMICS_RATIO="${MAXIMUM_DYNAMICS_RATIO:-2.0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-3601}"

cd "${ROOT}"
COMMIT="$(git rev-parse HEAD)"
RUN_NAME="${RUN_NAME:-orthogonalized_object_residual_v36_${SPLIT}_${COMMIT:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/v36_orthogonal_audits/${RUN_NAME}.json}"
LOG="${LOG:-${RUNTIME_ROOT}/logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

test -x "${PY}"
test -f "${DATA}/episode_manifest.json"
test -f "${CHECKPOINT}"
test -f "${V35_REPORT}"
test -z "$(git status --porcelain --untracked-files=no)"

echo "[orthogonalized-object-residual-v36] commit=${COMMIT}"
echo "[orthogonalized-object-residual-v36] checkpoint=${CHECKPOINT}"
echo "[orthogonalized-object-residual-v36] v35=${V35_REPORT}"
echo "[orthogonalized-object-residual-v36] report=${OUT}"
echo "[orthogonalized-object-residual-v36] log=${LOG}"

"${PY}" "${ROOT}/code/scripts/audit_orthogonalized_object_residual_v36.py" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --v35_report "${V35_REPORT}" \
  --output "${OUT}" \
  --split "${SPLIT}" \
  --max_items "${MAX_ITEMS}" \
  --workers "${WORKERS}" \
  --budgets "${BUDGETS}" \
  --compact_budget "${COMPACT_BUDGET}" \
  --ridge "${RIDGE}" \
  --relative_singular_cutoff "${RELATIVE_SINGULAR_CUTOFF}" \
  --minimum_column_fraction "${MINIMUM_COLUMN_FRACTION}" \
  --minimum_utility_fraction "${MINIMUM_UTILITY_FRACTION}" \
  --maximum_scene_fraction "${MAXIMUM_SCENE_FRACTION}" \
  --minimum_compact_recovery "${MINIMUM_COMPACT_RECOVERY}" \
  --maximum_token_ratio "${MAXIMUM_TOKEN_RATIO}" \
  --maximum_coefficient_rms "${MAXIMUM_COEFFICIENT_RMS}" \
  --maximum_condition_number "${MAXIMUM_CONDITION_NUMBER}" \
  --minimum_retained_energy "${MINIMUM_RETAINED_ENERGY}" \
  --maximum_column_amplification "${MAXIMUM_COLUMN_AMPLIFICATION}" \
  --maximum_transport_ratio "${MAXIMUM_TRANSPORT_RATIO}" \
  --maximum_dynamics_ratio "${MAXIMUM_DYNAMICS_RATIO}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}" 2>&1 | tee "${LOG}"

echo "[orthogonalized-object-residual-v36] completed report=${OUT}"
