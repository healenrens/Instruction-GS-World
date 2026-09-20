#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
SOURCE_REVISION="${SOURCE_REVISION:-local-unversioned}"
RUN_NAME="${RUN_NAME:-grounded_motion_data_v68_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/data/${RUN_NAME}}"
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${OUT}" "${RUNTIME_ROOT}/wandb"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node "${DATA_GPUS:-1}" \
  "${ROOT}/code/scripts/build_grounded_motion_data_v68.py" \
  --out "${OUT}" --source_revision "${SOURCE_REVISION}" --operation "${OPERATION:-build}" --stage "${DATA_STAGE:-run}" \
  --seed "${SEED:-17}" --tracker_version "${TRACKER_VERSION:-3}" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --case_manifest "${CASE_MANIFEST:-}" --camera_overrides "${CAMERA_OVERRIDES:-}" \
  --partition "${DATA_PARTITION:-held}" --cases_per_source "${CASES_PER_SOURCE:-80}" \
  --all_episode_windows "${ALL_EPISODE_WINDOWS:-0}" --clip_seconds "${CLIP_SECONDS:-10}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --grounding_model "${GROUNDING_MODEL:-${RUNTIME_ROOT}/models/grounding-dino-base}" \
  --sam_model "${SAM_MODEL:-${RUNTIME_ROOT}/models/sam2.1-hiera-large}" \
  --point_budget "${POINT_BUDGET:-2048}" --pilot_point_budget "${PILOT_POINT_BUDGET:-512}" \
  --points_per_pass "${POINTS_PER_PASS:-256}" --motion_top_fraction "${MOTION_TOP_FRACTION:-0.75}" \
  --background_grid_side "${BACKGROUND_GRID_SIDE:-16}" --background_ransac_px "${BACKGROUND_RANSAC_PX:-2}" \
  --relay_max_error_px "${RELAY_MAX_ERROR_PX:-3}" --render "${RENDER:-1}" --reuse_completed "${REUSE_COMPLETED:-1}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${RUN_NAME}" --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
  2>&1 | tee -a "${OUT}/build.log" && \
"${PY}" "${ROOT}/code/scripts/merge_grounded_motion_data_v68.py" --out "${OUT}"
