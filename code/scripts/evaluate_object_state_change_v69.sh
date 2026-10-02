#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/object_video_v69_state_seed17_40e4586_20260929_003931/latest.pt}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/v69_state_change_held_$(date +%Y%m%d_%H%M%S)_swanlab}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-auto}"
if [[ "${EVAL_NPROC_PER_NODE}" == auto ]]; then
  EVAL_NPROC_PER_NODE="$("${PY}" -c 'import torch; print(min(4, torch.cuda.device_count()))')"
fi
ENCODER="${ENCODER:-dinov3_vitl16}"
if [[ "${ENCODER}" == vjepa2_1_vitl16 ]]; then
  ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${RUNTIME_ROOT}/third_party/vjepa2}"
else
  ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${RUNTIME_ROOT}/third_party/dinov3}"
fi
SOURCE_REVISION="${SOURCE_REVISION:-$(<"${ROOT}/SOURCE_REVISION")}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT}"
echo "[state-change-v69] foreground processes=${EVAL_NPROC_PER_NODE} checkpoint=${CHECKPOINT} out=${OUT} log=${OUT}/evaluate.log"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node "${EVAL_NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/evaluate_object_state_change_v69.py" \
  --checkpoint "${CHECKPOINT}" --manifest "${MANIFEST:-}" \
  --out "${OUT}" --encoder_repository "${ENCODER_REPOSITORY}" --encoder_frame_batch "${ENCODER_FRAME_BATCH:-2}" \
  --items_per_source "${ITEMS_PER_SOURCE:-80}" --visualize_per_source "${VISUALIZE_PER_SOURCE:-8}" \
  --motion_threshold_px "${MOTION_THRESHOLD_PX:-5}" --annotations "${ANNOTATIONS:-}" --seed "${SEED:-17}" \
  --source_revision "${SOURCE_REVISION}" --swanlab_project "${SWANLAB_PROJECT:-instruct-gs-world}" \
  --swanlab_workspace "${SWANLAB_WORKSPACE:-}" \
  --swanlab_name "${RUN_NAME:-$(basename "${OUT}")}" --swanlab_mode disabled \
  2>&1 | tee -a "${OUT}/evaluate.log"
