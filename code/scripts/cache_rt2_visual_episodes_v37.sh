#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pfs/public/xuhaoming/Cosmos-3-Finetune/data/RoboTwin2}"
OUT="${OUT:-${RUNTIME_ROOT}/data/rt2_visual_episodes_dinov2l_native_v2}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/rt2_visual_episodes_dinov2l_native_v2}"
GPU_IDS="${GPU_IDS:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRAME_BATCH="${FRAME_BATCH:-13}"
CPU_THREADS="${CPU_THREADS:-8}"
JPEG_WORKERS="${JPEG_WORKERS:-2}"
WINDOW_LENGTHS="${WINDOW_LENGTHS:-25,50,75,100}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
OVERWRITE="${OVERWRITE:-0}"
PY="${VENV_ROOT}/.venv/bin/python"

for path in "${ROOT}" "${RUNTIME_ROOT}" "${SOURCE_ROOT}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[v37-cache] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
if [[ ! -x "${PY}" ]]; then
    echo "[v37-cache] missing Python environment: ${PY}" >&2
    exit 2
fi
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "[v37-cache] repository is not clean" >&2
    git -C "${ROOT}" status --short >&2
    exit 2
fi
if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[v37-cache] JOBS_PER_GPU must be positive" >&2
    exit 2
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "[v37-cache] OVERWRITE must be 0 or 1" >&2
    exit 2
fi

if [[ "${GPU_IDS}" == "auto" ]]; then
    visible_count="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
    if ! [[ "${visible_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[v37-cache] no CUDA devices are visible" >&2
        exit 2
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
        if [[ "${#gpus[@]}" -ne "${visible_count}" ]]; then
            echo "[v37-cache] CUDA_VISIBLE_DEVICES does not match torch visibility" >&2
            exit 2
        fi
    else
        gpus=()
        for ((index = 0; index < visible_count; index++)); do
            gpus+=("${index}")
        done
    fi
else
    IFS=',' read -r -a gpus <<< "${GPU_IDS}"
    if [[ "${#gpus[@]}" -eq 0 ]]; then
        echo "[v37-cache] GPU_IDS is empty" >&2
        exit 2
    fi
fi

mkdir -p "${OUT}" "${LOG_ROOT}"
cd "${ROOT}"
commit="$(git rev-parse HEAD)"
common=(
    --source_root "${SOURCE_ROOT}"
    --out "${OUT}"
    --model vit_large_patch14_dinov2.lvd142m
    --image_size 518
    --feature_dim 1024
    --feature_contract backbone_native
    --projection_seed 0
    --frame_batch "${FRAME_BATCH}"
    --cpu_threads "${CPU_THREADS}"
    --jpeg_workers "${JPEG_WORKERS}"
    --window_lengths "${WINDOW_LENGTHS}"
    --sample_stride "${SAMPLE_STRIDE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    common+=(--overwrite)
fi

echo "[v37-cache] commit=${commit} gpus=${gpus[*]} out=${OUT}"
"${PY}" code/scripts/cache_rt2_visual_episodes.py \
    "${common[@]}" --prepare_manifest \
    2>&1 | tee "${LOG_ROOT}/prepare_manifest.log"

pids=()
logs=()
total_shards=$(( ${#gpus[@]} * JOBS_PER_GPU ))
for ((shard = 0; shard < total_shards; shard++)); do
    gpu="${gpus[$((shard % ${#gpus[@]}))]}"
    log="${LOG_ROOT}/shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
        code/scripts/cache_rt2_visual_episodes.py \
        "${common[@]}" --shard "${shard}" --nshard "${total_shards}" \
        >"${log}" 2>&1 &
    pids+=("$!")
    logs+=("${log}")
    echo "[v37-cache] shard=${shard}/${total_shards} gpu=${gpu} pid=$! log=${log}"
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
        failed=1
        echo "[v37-cache] failed log=${logs[${index}]}" >&2
        tail -n 120 "${logs[${index}]}" >&2
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo "[v37-cache] one or more cache shards failed" >&2
    exit 1
fi

"${PY}" code/scripts/cache_rt2_visual_episodes.py \
    "${common[@]}" --finalize_manifest \
    2>&1 | tee "${LOG_ROOT}/finalize_manifest.log"
"${PY}" code/scripts/verify_rt2_visual_episode_cache.py \
    --data "${OUT}" --output "${LOG_ROOT}/verification.json" \
    2>&1 | tee "${LOG_ROOT}/verify.log"
(
    cd "${OUT}"
    sha256sum episode_manifest.json >episode_manifest.verified.sha256
)
echo "[v37-cache] completed data=${OUT} verification=${LOG_ROOT}/verification.json"
