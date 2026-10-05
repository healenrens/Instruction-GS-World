#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
export VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
export DATA_ROOT="${DATA_ROOT:-${RUNTIME_ROOT}/data/language_object_effect_v70_step8750}"
export MODEL_PATH="${MODEL_PATH:-/mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct}"
export EXPORT_GPUS="${EXPORT_GPUS:-1}"
export TEST_GPUS="${TEST_GPUS:-1}"
export TEST_BATCH="${TEST_BATCH:-1}"
export DINO_FRAME_BATCH="${DINO_FRAME_BATCH:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MANIFEST="${DATA_ROOT}/labeled_manifest.json"
export TEST_OUT="${TEST_OUT:-${RUNTIME_ROOT}/outputs/v70_tests/step8750_$(date +%Y%m%d_%H%M%S)}"
PY="${VENV_ROOT}/.venv/bin/python"
STAGE2_MANIFEST="${STAGE2_MANIFEST:-${RUNTIME_ROOT}/outputs/object_video_v69_state_seed17_40e4586_20260929_003931/dataset.json}"
TRACE="${RUNTIME_ROOT}/outputs/v70_language_audits/stage2_language_full_metadata_trace_20261006.jsonl"
AUDIT="${DATA_ROOT}/language_audit.json"

echo "[v70-prepare] fixed_teacher=${DATA_ROOT}/teacher.pt"
echo "[v70-prepare] phase=language_audit"
"${PY}" "${ROOT}/code/scripts/inspect_language_sources_v70.py" \
  --stage2_manifest "${STAGE2_MANIFEST}" --verified_trace "${TRACE}" --output "${AUDIT}"

echo "[v70-prepare] phase=language_windows"
"${PY}" "${ROOT}/code/scripts/prepare_language_manifest_v70.py" \
  --manifest "${STAGE2_MANIFEST}" --teacher_checkpoint "${DATA_ROOT}/teacher.pt" \
  --audit "${AUDIT}" --verified_trace "${TRACE}" --tokenizer_path "${MODEL_PATH}" \
  --output "${DATA_ROOT}/language_manifest.json"

echo "[v70-prepare] phase=label_export"
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" export

echo "[v70-prepare] phase=full_test world=${TEST_GPUS} out=${TEST_OUT}"
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" test
printf '[v70-prepare] completed manifest=%s report=%s\n' \
  "${MANIFEST}" "${TEST_OUT}/resume_report.json"
