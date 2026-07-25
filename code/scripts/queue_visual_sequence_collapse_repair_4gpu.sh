#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LAUNCHER="${ROOT}/code/scripts/train_visual_sequence_collapse_repair_4gpu.sh"

if [[ "${MIN_FREE_MIB}" -lt 1 || "${POLL_SECONDS}" -lt 1 ]]; then
    echo "[collapse-repair-queue] thresholds must be positive" >&2
    exit 2
fi
if [[ ! -x "${LAUNCHER}" ]]; then
    echo "[collapse-repair-queue] launcher is missing: ${LAUNCHER}" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[collapse-repair-queue] expected exactly four GPU ids" >&2
    exit 2
fi

while true; do
    ready=1
    status=()
    for gpu in "${gpus[@]}"; do
        free_mib="$(
            nvidia-smi --id="${gpu}" --query-gpu=memory.free \
                --format=csv,noheader,nounits
        )"
        status+=("${gpu}:${free_mib}MiB")
        if [[ "${free_mib}" -lt "${MIN_FREE_MIB}" ]]; then
            ready=0
        fi
    done
    printf '[collapse-repair-queue] %s free=%s threshold=%sMiB\n' \
        "$(date --iso-8601=seconds)" "${status[*]}" "${MIN_FREE_MIB}"
    if [[ "${ready}" -eq 1 ]]; then
        break
    fi
    sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
exec env GPU_IDS="${GPU_IDS}" bash "${LAUNCHER}"
