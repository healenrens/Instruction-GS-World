#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE="${SOURCE:-${ROOT}/data/rt2_joint_src}"
OUT="${OUT:-${ROOT}/data/rt2_visual_sequences_no_language_v1}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/rt2_visual_sequence_cache_20260720}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRAME_BATCH="${FRAME_BATCH:-13}"
LIMIT="${LIMIT:-0}"

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[visual-sequence] expected exactly four GPU ids" >&2
    exit 2
fi
if [[ "${JOBS_PER_GPU}" -lt 1 ]]; then
    echo "[visual-sequence] JOBS_PER_GPU must be positive" >&2
    exit 2
fi
if [[ ! -d "${SOURCE}" ]]; then
    echo "[visual-sequence] source not found: ${SOURCE}" >&2
    exit 2
fi
mkdir -p "${OUT}" "${LOG_ROOT}"
cd "${ROOT}"

pids=()
total_shards=$(("${#gpus[@]}" * JOBS_PER_GPU))
for ((shard = 0; shard < total_shards; shard++)); do
    gpu="${gpus[$((shard % ${#gpus[@]}))]}"
    log="${LOG_ROOT}/shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        .venv/bin/python code/scripts/cache_rt2_visual_sequences.py \
        --source "${SOURCE}" \
        --out "${OUT}" \
        --frame_batch "${FRAME_BATCH}" \
        --shard "${shard}" \
        --nshard "${total_shards}" \
        --limit "${LIMIT}" \
        >"${log}" 2>&1 &
    pid="$!"
    pids+=("${pid}")
    echo "[visual-sequence] shard=${shard}/${total_shards} gpu=${gpu} pid=${pid}"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if [[ "${status}" -ne 0 ]]; then
    echo "[visual-sequence] at least one cache shard failed" >&2
    exit "${status}"
fi

echo "[visual-sequence] all shards complete: ${OUT}"
