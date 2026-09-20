#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DEPLOYED_REVISION=local-unversioned
if [ -f "${ROOT}/SOURCE_REVISION" ]; then read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"; fi
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
RUN_NAME="${RUN_NAME:-grounded_object_tracker_v67_10s_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}}"

export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
echo "[grounded-tracker] foreground=single_gpu output=${OUT} stage=${REVIEW_STAGE:-run}"

"${VENV_ROOT}/.venv/bin/python" -u "${ROOT}/code/scripts/review_grounded_object_tracker_v67.py" \
  --stage "${REVIEW_STAGE:-run}" \
  --out "${OUT}" --source_revision "${SOURCE_REVISION}" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --case_manifest "${CASE_MANIFEST:-}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --tracker_version "${TRACKER_VERSION:-3}" \
  --grounding_model "${GROUNDING_MODEL:-${RUNTIME_ROOT}/models/grounding-dino-base}" \
  --sam_model "${SAM_MODEL:-${RUNTIME_ROOT}/models/sam2.1-hiera-large}" \
  --cases_per_source "${CASES_PER_SOURCE:-5}" --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --seed "${SEED:-17}" --clip_seconds "${CLIP_SECONDS:-10}" --steps_ms "${TEMPORAL_STEP_MS:-400}" \
  --point_budget "${POINT_BUDGET:-2048}" --points_per_pass "${POINTS_PER_PASS:-256}" \
  --query_every_seconds "${QUERY_EVERY_SECONDS:-2}" \
  --minimum_region_points "${MINIMUM_REGION_POINTS:-4}" --maximum_region_points "${MAXIMUM_REGION_POINTS:-96}" \
  --robot_point_fraction "${ROBOT_POINT_FRACTION:-0.15}" --other_context_fraction "${OTHER_CONTEXT_FRACTION:-0.05}" \
  --sam_grid_side "${SAM_GRID_SIDE:-12}" --sam_crop_divisions "${SAM_CROP_DIVISIONS:-2}" \
  --sam_prompt_batch "${SAM_PROMPT_BATCH:-16}" --sam_min_area "${SAM_MIN_AREA:-8}" \
  --sam_score_threshold "${SAM_SCORE_THRESHOLD:-0.70}" --sam_stability_threshold "${SAM_STABILITY_THRESHOLD:-0.90}" \
  --mask_dedup_iou "${MASK_DEDUP_IOU:-0.80}" \
  --robot_box_threshold "${ROBOT_BOX_THRESHOLD:-0.25}" --robot_text_threshold "${ROBOT_TEXT_THRESHOLD:-0.20}" \
  --robot_overlap_threshold "${ROBOT_OVERLAP_THRESHOLD:-0.10}" --scene_area_fraction "${SCENE_AREA_FRACTION:-0.40}" \
  --max_masks_per_frame "${MAX_MASKS_PER_FRAME:-48}" --max_context_masks "${MAX_CONTEXT_MASKS:-8}" \
  --motion_floor_pixels "${MOTION_FLOOR_PIXELS:-1.5}" --motion_region_fraction "${MOTION_REGION_FRACTION:-0.08}" \
  --motion_noise_multiplier "${MOTION_NOISE_MULTIPLIER:-3.0}" --minimum_visible_frames "${MINIMUM_VISIBLE_FRAMES:-6}" \
  --display_width "${DISPLAY_WIDTH:-640}" --reuse_completed "${REUSE_COMPLETED:-1}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
  2>&1 | tee -a "${OUT}/review.log"
