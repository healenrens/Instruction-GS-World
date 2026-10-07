#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
export SOURCE_REVISION="$(cat "${ROOT}/SOURCE_REVISION")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TRAIN_OUT="${TRAIN_OUT:-${RUNTIME_ROOT}/outputs/language_object_effect_v70_flow_32gpu_from500_daca36e}"
MANIFEST="${MANIFEST:-${RUNTIME_ROOT}/data/language_object_effect_v70_step8750/labeled_manifest.json}"
EVAL_OUT="${EVAL_OUT:-${RUNTIME_ROOT}/outputs/v70_evaluation/paired_2500_5000_${SOURCE_REVISION:0:7}}"
FIRST_CHECKPOINT="${FIRST_CHECKPOINT:-${TRAIN_OUT}/step_0002500}"
SECOND_CHECKPOINT="${SECOND_CHECKPOINT:-${TRAIN_OUT}/step_0005000}"
echo "[v70-paired-eval] first=${FIRST_CHECKPOINT} second=${SECOND_CHECKPOINT} output=${EVAL_OUT}"
EXTRA=()
if [ -n "${CONFLICTS:-}" ]; then EXTRA+=(--conflicts "${CONFLICTS}"); fi
CHECKPOINTS=("${FIRST_CHECKPOINT}" "${SECOND_CHECKPOINT}")
LABELS=(first second)
PLAN=()
for i in 0 1; do
  "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${EVAL_GPUS:-4}" --max_restarts=0 \
    "${ROOT}/code/scripts/evaluate_language_object_effect_v70.py" \
    --checkpoint "${CHECKPOINTS[$i]}" --output "${EVAL_OUT}/${LABELS[$i]}" \
    --manifest "${MANIFEST}" --partition diagnostic --limit "${EVAL_CASES:-256}" \
    --samples "${EVAL_SAMPLES:-4}" --videos "${EVAL_VIDEOS:-24}" --seed "${EVAL_SEED:-17}" \
    --flow_steps "${FLOW_STEPS:-10}" --frame_batch "${DINO_FRAME_BATCH:-8}" \
    --motion_floor_px "${MOTION_FLOOR_PX:-1.0}" ${PLAN[@]+"${PLAN[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
  PLAN=(--plan "${EVAL_OUT}/first/selection.json")
done
exec "${PY}" "${ROOT}/code/scripts/compare_language_object_effect_v70.py" \
  --first "${EVAL_OUT}/first/report.json" --second "${EVAL_OUT}/second/report.json" \
  --output "${EVAL_OUT}" --seed "${EVAL_SEED:-17}"
