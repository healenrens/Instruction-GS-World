#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pfs/public/xuhaoming/Cosmos-3-Finetune/data/RoboTwin2}"
OUT="${OUT:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/rt2_visual_episode_cache_20260720}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRAME_BATCH="${FRAME_BATCH:-52}"
CPU_THREADS="${CPU_THREADS:-16}"
JPEG_WORKERS="${JPEG_WORKERS:-4}"
WINDOW_LENGTHS="${WINDOW_LENGTHS:-25,50,75,100}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[visual-episode] expected exactly four GPU ids" >&2
    exit 2
fi
if [[ "${JOBS_PER_GPU}" -lt 1 || ! -d "${SOURCE_ROOT}" ]]; then
    echo "[visual-episode] invalid worker count or source root" >&2
    exit 2
fi
mkdir -p "${OUT}" "${LOG_ROOT}"
printf "%s\n" "$$" >"${LOG_ROOT}/launcher.pid"
rm -f "${LOG_ROOT}/launcher.exit_code"
record_exit() {
    printf "%s\n" "$?" >"${LOG_ROOT}/launcher.exit_code"
}
trap record_exit EXIT
cd "${ROOT}"

common=(
    --source_root "${SOURCE_ROOT}"
    --out "${OUT}"
    --frame_batch "${FRAME_BATCH}"
    --cpu_threads "${CPU_THREADS}"
    --jpeg_workers "${JPEG_WORKERS}"
    --window_lengths "${WINDOW_LENGTHS}"
    --sample_stride "${SAMPLE_STRIDE}"
    --limit "${LIMIT}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    common+=(--overwrite)
elif [[ "${OVERWRITE}" != "0" ]]; then
    echo "[visual-episode] OVERWRITE must be 0 or 1" >&2
    exit 2
fi

.venv/bin/python code/scripts/cache_rt2_visual_episodes.py \
    "${common[@]}" \
    --prepare_manifest \
    >"${LOG_ROOT}/prepare_manifest.log" 2>&1

pids=()
total_shards=$(("${#gpus[@]}" * JOBS_PER_GPU))
for ((shard = 0; shard < total_shards; shard++)); do
    gpu="${gpus[$((shard % ${#gpus[@]}))]}"
    log="${LOG_ROOT}/shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        .venv/bin/python code/scripts/cache_rt2_visual_episodes.py \
        "${common[@]}" \
        --shard "${shard}" \
        --nshard "${total_shards}" \
        >"${log}" 2>&1 &
    pids+=("$!")
    echo "[visual-episode] shard=${shard}/${total_shards} gpu=${gpu} pid=$!"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if [[ "${status}" -ne 0 ]]; then
    echo "[visual-episode] at least one cache shard failed" >&2
    exit "${status}"
fi

.venv/bin/python code/scripts/cache_rt2_visual_episodes.py \
    "${common[@]}" \
    --finalize_manifest \
    >"${LOG_ROOT}/finalize_manifest.log" 2>&1
.venv/bin/python code/scripts/verify_rt2_visual_episode_cache.py \
    --data "${OUT}" \
    --output "${LOG_ROOT}/verification.json" \
    >"${LOG_ROOT}/verify.log" 2>&1
sha256sum "${OUT}/episode_manifest.json" \
    >"${OUT}/episode_manifest.verified.sha256"
echo "[visual-episode] all shards complete: ${OUT}"
