#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DEPLOYED_REVISION=local-unversioned
if [ -f "${ROOT}/SOURCE_REVISION" ]; then read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"; fi
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
INPUT_REVIEW="${INPUT_REVIEW:-${RUNTIME_ROOT}/outputs/tracker_visual_reviews/tracker_motion_review_v67_10s_seed17_5d82a59}"
RUN_NAME="${RUN_NAME:-tracker_motion_review_v67_moving_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}}"

export PYTHONUNBUFFERED=1
mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
echo "[tracker-moving] foreground=cpu input=${INPUT_REVIEW} output=${OUT}"

"${VENV_ROOT}/.venv/bin/python" -u "${ROOT}/code/scripts/render_moving_tracker_review_v67.py" \
  --input_review "${INPUT_REVIEW}" \
  --out "${OUT}" \
  --source_revision "${SOURCE_REVISION}" \
  --minimum_motion_pixels "${MINIMUM_MOTION_PIXELS:-12}" \
  --minimum_motion_fraction "${MINIMUM_MOTION_FRACTION:-0.02}" \
  --minimum_visible_frames "${MINIMUM_VISIBLE_FRAMES:-6}" \
  --trails_seconds "${TRAILS_SECONDS:-0}" \
  --display_width "${DISPLAY_WIDTH:-640}" \
  --reuse_completed "${REUSE_COMPLETED:-1}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
  2>&1 | tee -a "${OUT}/render.log"
