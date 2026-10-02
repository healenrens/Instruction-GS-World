#!/usr/bin/env bash
# Test-machine entry only. Training jobs use the published local release instead.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
export VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
export SOURCE_REVISION="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
export MANIFEST="${MANIFEST:-${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json}"
export ENCODER="${ENCODER:-dinov3_vitl16}"
export CUDA_VISIBLE_DEVICES="${TEST_GPU:-0}"
export ENCODER_FRAME_BATCH="${ENCODER_FRAME_BATCH:-2}"
export RUN_NAME="object_video_v69_resume_fix_${SOURCE_REVISION:0:7}_$(date +%Y%m%d_%H%M%S)_swanlab"
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export SWANLAB_MODE=disabled
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-instruct-gs-world}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
unset RESUME STATE_CHECKPOINT MODEL_CONFIG STOP_AFTER WANDB_RUN_ID WANDB_RESUME
bash "${SOURCE_ROOT}/code/scripts/deploy_object_video_sequence_v69_runtime.sh"
export ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${SOURCE_REVISION}"
if [[ ! -f "${MANIFEST}" ]]; then
  "${VENV_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/prepare_object_video_manifest_v69.py" \
    --input "${V68_DATA:-${RUNTIME_ROOT}/data/grounded_motion_v68_train20k_allpoints75_recovered_r1}" \
    --output "${MANIFEST}"
fi
echo "[object-video-v69-test] revision=${SOURCE_REVISION} out=${OUT}"
echo "[object-video-v69-test] comparisons=${OUT}/state_resume_comparison.json,${OUT}/dynamics_resume_comparison.json"
bash "${ROOT}/code/scripts/test_object_video_sequence_v69.sh"
