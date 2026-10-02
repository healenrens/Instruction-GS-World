#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
SOURCE_REVISION="${SOURCE_REVISION:-$(<"${ROOT}/SOURCE_REVISION")}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/object_video_v69_single_gpu_${SOURCE_REVISION:0:7}_swanlab}"
ENCODER="${ENCODER:-dinov3_vitl16}"
if [[ "${ENCODER}" == vjepa2_1_vitl16 ]]; then
  ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${RUNTIME_ROOT}/third_party/vjepa2}"
  ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-${RUNTIME_ROOT}/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"
else
  ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${RUNTIME_ROOT}/third_party/dinov3}"
  ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT}"
echo "[object-video-v69-test] full_model=true device=cuda:0 log=${OUT}/test.log"
"${PY}" "${SCRIPT_DIR}/test_object_video_sequence_v69.py" \
  --manifest "${MANIFEST:-${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json}" --out "${OUT}" \
  --encoder "${ENCODER}" --encoder_repository "${ENCODER_REPOSITORY}" --encoder_weights "${ENCODER_WEIGHTS}" \
  --encoder_frame_batch "${ENCODER_FRAME_BATCH:-2}" --config "${MODEL_CONFIG:-}" --seed "${SEED:-17}" \
  --posterior_geometry "${POSTERIOR_GEOMETRY:-inherit}" \
  --source_revision "${SOURCE_REVISION}" --swanlab_project "${SWANLAB_PROJECT:-instruct-gs-world}" \
  --swanlab_workspace "${SWANLAB_WORKSPACE:-}" \
  --swanlab_mode disabled --swanlab_name "${RUN_NAME:-$(basename "${OUT}")}" \
  2>&1 | tee -a "${OUT}/test.log"
