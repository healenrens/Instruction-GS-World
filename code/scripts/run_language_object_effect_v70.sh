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
      --manifest "${DATA_ROOT}/language_manifest.json" --frame_batch "${DINO_FRAME_BATCH:-32}" \
      --batch "${EXPORT_BATCH_PER_GPU:-4}" --workers "${EXPORT_WORKERS_PER_RANK:-4}" \
      --prefetch "${EXPORT_PREFETCH:-1}"
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
    if [ -n "${INIT_FROM:-}" ]; then EXTRA+=(--init_from "${INIT_FROM}"); fi
    NNODES="${NNODES:-1}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    LAUNCH=(--nproc_per_node="${NPROC_PER_NODE}" --max_restarts=0)
    if [ "${NNODES}" -eq 1 ]; then
      LAUNCH+=(--standalone)
    else
      LAUNCH+=(--nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        --rdzv_id="${RUN_NAME}")
    fi
    mkdir -p "${OUT}"
    echo "[language-effect-v70] nodes=${NNODES} node_rank=${NODE_RANK:-0} gpus_per_node=${NPROC_PER_NODE} out=${OUT}"
    exec "${PY}" -m torch.distributed.run "${LAUNCH[@]}" \
      --log_dir "${OUT}/torchrun" --tee 3 \
      "${ROOT}/code/scripts/train_language_object_effect_v70.py" \
      --manifest "${MANIFEST}" --model_path "${MODEL_PATH}" --out "${OUT}" \
      --mode "${MODE}" --batch "${BATCH_PER_GPU:-4}" --accum "${GRAD_ACCUM:-8}" \
      --steps "${STEPS:-10000}" --workers "${WORKERS_PER_RANK:-2}" \
      --frame_batch "${DINO_FRAME_BATCH:-8}" --visual_tokens 4096 --text_tokens 512 \
      --lr_text "${LR_TEXT:-1e-5}" --lr_expert "${LR_EXPERT:-1e-4}" \
      --warmup_fraction "${WARMUP_FRACTION:-0.05}" \
      --clip 1 --checkpoint_every 500 --snapshot_every 2500 \
      --swanlab_project "${SWANLAB_PROJ_NAME:-instruct-gs-world}" \
      --swanlab_workspace "${SWANLAB_WORKSPACE_NAME:-}" --swanlab_name "${RUN_NAME}" \
      --swanlab_mode "${SWANLAB_MODE:-online}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  evaluate)
    EXTRA=()
    if [ -n "${CONFLICTS:-}" ]; then EXTRA+=(--conflicts "${CONFLICTS}"); fi
    exec "${PY}" "${ROOT}/code/scripts/evaluate_language_object_effect_v70.py" \
      --manifest "${MANIFEST}" --checkpoint "${CHECKPOINT}" \
      --output "${EVAL_OUT:-${RUNTIME_ROOT}/outputs/v70_evaluation/${RUN_NAME}}" \
      --partition "${PARTITION:-diagnostic}" --limit "${EVAL_CASES:-128}" \
      --samples "${EVAL_SAMPLES:-4}" --flow_steps 10 ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
esac
