#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
DATA_ROOT="${DATA_ROOT:-${RUNTIME_ROOT}/data/language_object_effect_v70_step8750}"
MODEL_PATH="${MODEL_PATH:-/mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/labeled_manifest.json}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
MODE="${MODE:-flow}"
RUN_NAME="${RUN_NAME:-language_object_effect_v70_${MODE}_seed17}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"

case "${1:-train}" in
  export)
    exec "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${EXPORT_GPUS:-1}" \
      "${ROOT}/code/scripts/export_language_effect_labels_v70.py" \
      --manifest "${DATA_ROOT}/language_manifest.json" --frame_batch "${DINO_FRAME_BATCH:-8}"
    ;;
  test)
    exec "${PY}" "${ROOT}/code/scripts/test_language_object_effect_v70.py" \
      --manifest "${MANIFEST}" --model_path "${MODEL_PATH}" \
      --out "${TEST_OUT:-${RUNTIME_ROOT}/outputs/v70_tests/$(date +%Y%m%d_%H%M%S)}" \
      --batch "${TEST_BATCH:-1}" --nproc_per_node "${TEST_GPUS:-1}"
    ;;
  train|resume)
    EXTRA=()
    if [ -n "${RESUME:-}" ]; then EXTRA+=(--resume "${RESUME}"); fi
    mkdir -p "${OUT}"
    exec "${PY}" -m torch.distributed.run --standalone --nproc_per_node=8 \
      "${ROOT}/code/scripts/train_language_object_effect_v70.py" \
      --manifest "${MANIFEST}" --model_path "${MODEL_PATH}" --out "${OUT}" \
      --mode "${MODE}" --batch "${BATCH_PER_GPU:-4}" --accum "${GRAD_ACCUM:-8}" \
      --steps "${STEPS:-10000}" --workers "${WORKERS_PER_RANK:-2}" \
      --frame_batch "${DINO_FRAME_BATCH:-8}" --visual_tokens 4096 --text_tokens 512 \
      --lr_text 1e-5 --lr_expert 1e-4 --clip 1 --checkpoint_every 500 --snapshot_every 2500 \
      --swanlab_project "${SWANLAB_PROJ_NAME:-instruct-gs-world}" \
      --swanlab_workspace "${SWANLAB_WORKSPACE_NAME:-}" --swanlab_name "${RUN_NAME}" \
      --swanlab_mode "${SWANLAB_MODE:-online}" "${EXTRA[@]}"
    ;;
  evaluate)
    EXTRA=()
    if [ -n "${CONFLICTS:-}" ]; then EXTRA+=(--conflicts "${CONFLICTS}"); fi
    exec "${PY}" "${ROOT}/code/scripts/evaluate_language_object_effect_v70.py" \
      --manifest "${MANIFEST}" --checkpoint "${CHECKPOINT}" \
      --output "${EVAL_OUT:-${RUNTIME_ROOT}/outputs/v70_evaluation/${RUN_NAME}}" \
      --partition "${PARTITION:-diagnostic}" --limit "${EVAL_CASES:-128}" \
      --samples "${EVAL_SAMPLES:-4}" --flow_steps 10 "${EXTRA[@]}"
    ;;
esac
