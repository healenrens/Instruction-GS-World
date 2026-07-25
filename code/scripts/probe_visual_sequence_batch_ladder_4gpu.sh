#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${ROOT}/data/rt2_visual_sequences_no_language_v1}"
SOURCE="${SOURCE:-${ROOT}/outputs/canonical6_rgb_centergate0p1_activitypower1_v24_converted_seed17_20260720/joint_0000000.pt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/visual_sequence_batch_ladder_20260720}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
STEPS="${STEPS:-2}"
LR="${LR:-2e-4}"
LR_FLOOR="${LR_FLOOR:-2e-5}"
MIN_HEADROOM="${MIN_HEADROOM:-0.15}"

if [[ "${DATA}" != /* || "${SOURCE}" != /* || "${OUT_ROOT}" != /* ]]; then
    echo "[batch-ladder] paths must be absolute" >&2
    exit 2
fi
if [[ ! -d "${DATA}" || ! -f "${SOURCE}" || "${STEPS}" -lt 1 ]]; then
    echo "[batch-ladder] data, source, or step configuration is invalid" >&2
    exit 2
fi
if [[ -e "${OUT_ROOT}/summary.json" ]]; then
    echo "[batch-ladder] refusing to overwrite ${OUT_ROOT}/summary.json" >&2
    exit 2
fi

cd "${ROOT}"
batches=(1 2 4 8)
accumulations=(64 32 16 8)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    for index in 0 1 2 3; do
        DRY_RUN=1 DATA="${DATA}" SOURCE="${SOURCE}" \
            OUT="${OUT_ROOT}/b${batches[$index]}_a${accumulations[$index]}" \
            GPU_IDS="${GPU_IDS}" JOINT_STEPS="${STEPS}" \
            BATCH_PER_GPU="${batches[$index]}" \
            GRAD_ACCUM="${accumulations[$index]}" \
            LR="${LR}" LR_FLOOR="${LR_FLOOR}" \
            bash code/scripts/train_visual_sequence_core_4gpu.sh
    done
    exit 0
fi

mkdir -p "${OUT_ROOT}"
run_arguments=()
for index in 0 1 2 3; do
    batch="${batches[$index]}"
    accumulation="${accumulations[$index]}"
    label="b${batch}_a${accumulation}"
    out="${OUT_ROOT}/${label}"
    run_arguments+=(--run "${label}=${out}")
    if [[ -f "${out}/latest.pt" ]]; then
        echo "[batch-ladder] reuse completed ${label}"
        continue
    fi
    exit_code=0
    if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        DATA="${DATA}" SOURCE="${SOURCE}" OUT="${out}" GPU_IDS="${GPU_IDS}" \
        TRAIN_MODE=posterior JOINT_STEPS="${STEPS}" \
        BATCH_PER_GPU="${batch}" GRAD_ACCUM="${accumulation}" \
        LR="${LR}" LR_FLOOR="${LR_FLOOR}" \
        SAVE_EVERY=100000 LOG_EVERY=1 \
        bash code/scripts/train_visual_sequence_core_4gpu.sh; then
        exit_code=0
    else
        exit_code=$?
    fi
    mkdir -p "${out}"
    printf "%s\n" "${exit_code}" >"${out}/exit_code.txt"
done

.venv/bin/python code/scripts/summarize_visual_sequence_batch_ladder.py \
    "${run_arguments[@]}" \
    --expected_effective_batch 256 \
    --min_headroom "${MIN_HEADROOM}" \
    --output "${OUT_ROOT}/summary.json"
