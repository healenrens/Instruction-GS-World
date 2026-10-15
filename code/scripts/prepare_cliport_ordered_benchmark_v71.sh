#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_PYTHON="${CPU_PYTHON:-${WORK_ROOT}/environments/cliport-v71/bin/python}"
PLAN="${PLAN:-${WORK_ROOT}/metadata/cliport_event_v71_plan_c682ec9.json}"
COLLECTION="${COLLECTION:-${WORK_ROOT}/data/benchmarks/cliport_ordered_v71}"
MANIFEST="${MANIFEST:-${WORK_ROOT}/metadata/cliport_event_v71.json}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs/cliport_ordered_v71_data}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="${WORK_ROOT}/tmp/benchmark-v71"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/benchmark"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/benchmark-pycache"
mkdir -p "${TMPDIR}" "${LOG_ROOT}"
args=(--environment-report "${WORK_ROOT}/metadata/object_event_benchmark_environment_v71.json"
      --out "${COLLECTION}" --plan "${PLAN}" --workers "${GENERATE_WORKERS:-4}"
      --source-revision "${SOURCE_REVISION:-}" --history-seconds 3 --future-seconds 10)
if [ "${RESUME_DATA:-0}" = 1 ]; then args+=(--resume); fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'COLLECTION=%s\nPROGRESS=%s/progress.json\nLOG=%s/%s.log\n' \
  "${COLLECTION}" "${COLLECTION}" "${LOG_ROOT}" "${stamp}"
"${CPU_PYTHON}" "${ROOT}/code/scripts/generate_cliport_event_sequences_v71.py" "${args[@]}" \
  2>&1 | tee "${LOG_ROOT}/${stamp}.log"
"${CPU_PYTHON}" "${ROOT}/code/scripts/prepare_cliport_event_data_v71.py" \
  --collection "${COLLECTION}/manifest.json" --out "${MANIFEST}"
printf 'MANIFEST=%s\nREVIEW=%s/index.html\n' "${MANIFEST}" "${COLLECTION}"
