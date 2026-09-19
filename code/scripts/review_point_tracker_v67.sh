#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DEPLOYED_REVISION=local-unversioned
if [ -f "${ROOT}/SOURCE_REVISION" ]; then read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"; fi
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
RUN_NAME="${RUN_NAME:-tracker_visual_review_v67_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}}"

export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
echo "[tracker-review] foreground=single_gpu output=${OUT} stage=${REVIEW_STAGE:-run}"

"${VENV_ROOT}/.venv/bin/python" -u "${ROOT}/code/scripts/review_point_tracker_v67.py" \
  --stage "${REVIEW_STAGE:-run}" \
  --out "${OUT}" \
  --source_revision "${SOURCE_REVISION}" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --tracker_version "${TRACKER_VERSION:-3}" \
  --cases_per_source "${CASES_PER_SOURCE:-5}" \
  --steps_ms "${TEMPORAL_STEP_MS:-100,200,400}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --seed "${SEED:-17}" \
  --grid_side "${GRID_SIDE:-16}" \
  --queries_json "${QUERIES_JSON:-}" \
  --display_width "${DISPLAY_WIDTH:-640}" \
  --reuse_completed "${REUSE_COMPLETED:-1}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
  2>&1 | tee -a "${OUT}/review.log"
