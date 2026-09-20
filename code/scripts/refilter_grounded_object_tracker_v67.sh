#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DEPLOYED_REVISION=local-unversioned
if [ -f "${ROOT}/SOURCE_REVISION" ]; then read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"; fi
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
RUN_NAME="${RUN_NAME:-grounded_motion_top50_v67_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}}"
INPUT_REVIEW="${INPUT_REVIEW:-${RUNTIME_ROOT}/outputs/tracker_visual_reviews/grounded_object_coverage_v67_10s_seed17_36dc6be}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
echo "[grounded-topk] foreground=cpu input=${INPUT_REVIEW} output=${OUT}"
"${VENV_ROOT}/.venv/bin/python" -u "${ROOT}/code/scripts/refilter_grounded_object_tracker_v67.py" \
  --stage "${REVIEW_STAGE:-run}" --input_review "${INPUT_REVIEW}" --out "${OUT}" \
  --source_revision "${SOURCE_REVISION}" --motion_top_fraction "${MOTION_TOP_FRACTION:-0.5}" \
  --display_width "${DISPLAY_WIDTH:-640}" --reuse_completed "${REUSE_COMPLETED:-1}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
  2>&1 | tee -a "${OUT}/refilter.log"
