#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/readout_parent_snapshots/selected_parent.pt}"
V33_REPORT="${V33_REPORT:-${RUNTIME_ROOT}/outputs/v33_residual_audits/object_centered_residual_field_v33_heldseed_150855a.json}"
PY="${PY:-${RUNTIME_ROOT}/.venv/bin/python}"
SPLIT="${SPLIT:-heldseed}"
MAX_ITEMS="${MAX_ITEMS:-128}"
WORKERS="${WORKERS:-2}"
BUDGETS="${BUDGETS:-64,128,256}"
COMPACT_BUDGET="${COMPACT_BUDGET:-128}"
RIDGE="${RIDGE:-0.0001}"
MINIMUM_UTILITY_FRACTION="${MINIMUM_UTILITY_FRACTION:-0.001}"
MAXIMUM_SCENE_FRACTION="${MAXIMUM_SCENE_FRACTION:-0.25}"
CENTER_PERTURBATION_FRACTION="${CENTER_PERTURBATION_FRACTION:-0.01}"
SCALE_LOG_PERTURBATION="${SCALE_LOG_PERTURBATION:-0.01}"
MINIMUM_COMPACT_RECOVERY="${MINIMUM_COMPACT_RECOVERY:-0.70}"
MAXIMUM_TOKEN_RATIO="${MAXIMUM_TOKEN_RATIO:-1.10}"
MAXIMUM_COEFFICIENT_RATIO="${MAXIMUM_COEFFICIENT_RATIO:-10.0}"
MAXIMUM_SENSITIVITY_RATIO="${MAXIMUM_SENSITIVITY_RATIO:-1.0}"
MAXIMUM_DYNAMICS_RATIO="${MAXIMUM_DYNAMICS_RATIO:-2.0}"
MAXIMUM_CONDITION_NUMBER="${MAXIMUM_CONDITION_NUMBER:-5000000}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-3401}"

cd "${ROOT}"
COMMIT="$(git rev-parse HEAD)"
RUN_NAME="${RUN_NAME:-stable_object_residual_transport_v34_${SPLIT}_${COMMIT:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/v34_transport_audits/${RUN_NAME}.json}"
LOG="${LOG:-${RUNTIME_ROOT}/logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

test -x "${PY}"
test -f "${DATA}/episode_manifest.json"
test -f "${CHECKPOINT}"
test -f "${V33_REPORT}"
test -z "$(git status --porcelain --untracked-files=no)"

echo "[stable-object-residual-v34] commit=${COMMIT}"
echo "[stable-object-residual-v34] checkpoint=${CHECKPOINT}"
echo "[stable-object-residual-v34] v33=${V33_REPORT}"
echo "[stable-object-residual-v34] report=${OUT}"
echo "[stable-object-residual-v34] log=${LOG}"

"${PY}" "${ROOT}/code/scripts/audit_stable_object_residual_transport_v34.py" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --v33_report "${V33_REPORT}" \
  --output "${OUT}" \
  --split "${SPLIT}" \
  --max_items "${MAX_ITEMS}" \
  --workers "${WORKERS}" \
  --budgets "${BUDGETS}" \
  --compact_budget "${COMPACT_BUDGET}" \
  --ridge "${RIDGE}" \
  --minimum_utility_fraction "${MINIMUM_UTILITY_FRACTION}" \
  --maximum_scene_fraction "${MAXIMUM_SCENE_FRACTION}" \
  --center_perturbation_fraction "${CENTER_PERTURBATION_FRACTION}" \
  --scale_log_perturbation "${SCALE_LOG_PERTURBATION}" \
  --minimum_compact_recovery "${MINIMUM_COMPACT_RECOVERY}" \
  --maximum_token_ratio "${MAXIMUM_TOKEN_RATIO}" \
  --maximum_coefficient_ratio "${MAXIMUM_COEFFICIENT_RATIO}" \
  --maximum_sensitivity_ratio "${MAXIMUM_SENSITIVITY_RATIO}" \
  --maximum_dynamics_ratio "${MAXIMUM_DYNAMICS_RATIO}" \
  --maximum_condition_number "${MAXIMUM_CONDITION_NUMBER}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}" 2>&1 | tee "${LOG}"

echo "[stable-object-residual-v34] completed report=${OUT}"
