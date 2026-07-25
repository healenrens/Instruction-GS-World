#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${RUN_NAME}}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-12000}"
GPU_IDS="${GPU_IDS:-2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
MAX_ITEMS="${MAX_ITEMS:-144}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_queue_20260723.pid}"
TRAINING_CODE_MANIFEST="${TRAINING_CODE_MANIFEST:-${RUN_DIR}/training_code_manifest.sha256}"
RUNTIME_CODE_MANIFEST="${RUNTIME_CODE_MANIFEST:-${RUN_DIR}/training_runtime_code_manifest.sha256}"
EVALUATION_CODE_MANIFEST="${EVALUATION_CODE_MANIFEST:-${RUN_DIR}/evaluation_code_manifest.sha256}"
CODE_PROVENANCE_EXCEPTION="${CODE_PROVENANCE_EXCEPTION:-${RUN_DIR}/code_provenance_exception.json}"

if [[ "${RUN_DIR}" != /* || "${DATA}" != /* ]]; then
    echo "[representation-monitor] run and data paths must be absolute" >&2
    exit 2
fi
verify_code() {
    if ! "${ROOT}/.venv/bin/python" \
        "${ROOT}/code/scripts/verify_visual_sequence_code_provenance.py" \
        --root "${ROOT}" \
        --training_manifest "${TRAINING_CODE_MANIFEST}" \
        --runtime_manifest "${RUNTIME_CODE_MANIFEST}" \
        --evaluation_manifest "${EVALUATION_CODE_MANIFEST}" \
        --exception "${CODE_PROVENANCE_EXCEPTION}" >/dev/null; then
        echo "[representation-monitor] training or evaluation code changed" >&2
        exit 4
    fi
}
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]]; then
    echo "[representation-monitor] verified episode data is missing" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 2 ]]; then
    echo "[representation-monitor] expected exactly two evaluation GPU ids" >&2
    exit 2
fi

checkpoint="${RUN_DIR}/joint_$(printf '%07d' "${CHECKPOINT_STEP}").pt"
while [[ ! -f "${checkpoint}" ]]; do
    if [[ -s "${QUEUE_PID_FILE}" ]] \
        && ! kill -0 "$(cat "${QUEUE_PID_FILE}")" 2>/dev/null; then
        echo "[representation-monitor] training exited before ${checkpoint}" >&2
        exit 3
    fi
    printf '[representation-monitor] %s waiting_for=%s\n' \
        "$(date --iso-8601=seconds)" "${checkpoint}"
    sleep "${POLL_SECONDS}"
done

while true; do
    free=()
    ready=1
    for gpu in "${gpus[@]}"; do
        value="$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
        free+=("${gpu}:${value}MiB")
        if (( value < MIN_FREE_MIB )); then
            ready=0
        fi
    done
    if [[ "${ready}" -eq 1 ]]; then
        break
    fi
    printf '[representation-monitor] %s free=%s threshold=%sMiB\n' \
        "$(date --iso-8601=seconds)" "${free[*]}" "${MIN_FREE_MIB}"
    sleep "${POLL_SECONDS}"
done

eval_dir="${RUN_DIR}/representation_step${CHECKPOINT_STEP}"
verify_code
mkdir -p "${eval_dir}"
cd "${ROOT}"
pids=()
temporary_outputs=()
final_outputs=()
for index in 0 1; do
    split="heldseed"
    if [[ "${index}" -eq 1 ]]; then
        split="heldtask"
    fi
    output="${eval_dir}/${split}_${MAX_ITEMS}.json"
    if [[ -f "${output}" ]]; then
        echo "[representation-monitor] preserving existing report: ${output}"
        continue
    fi
    temporary="${output}.tmp.$$"
    rm -f "${temporary}"
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" .venv/bin/python \
        code/scripts/evaluate_visual_sequence_representation.py \
        --checkpoint "${checkpoint}" \
        --data "${DATA}" \
        --split "${split}" \
        --max_items "${MAX_ITEMS}" \
        --batch "${BATCH}" \
        --workers "${WORKERS}" \
        --device cuda \
        --amp bf16 \
        --output "${temporary}" \
        >"${eval_dir}/${split}.log" 2>&1 &
    pids+=("$!")
    temporary_outputs+=("${temporary}")
    final_outputs+=("${output}")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}" \
        && [[ -s "${temporary_outputs[$index]}" ]] \
        && [[ ! -e "${final_outputs[$index]}" ]]; then
        mv "${temporary_outputs[$index]}" "${final_outputs[$index]}"
    else
        rm -f "${temporary_outputs[$index]}"
        status=1
    fi
done
if [[ "${status}" -ne 0 ]]; then
    echo "[representation-monitor] evaluation failed" >&2
    exit "${status}"
fi
echo "[representation-monitor] complete reports=${eval_dir}"
