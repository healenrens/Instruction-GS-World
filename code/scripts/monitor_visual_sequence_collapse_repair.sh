#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${RUN_NAME}}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
MAX_ITEMS="${MAX_ITEMS:-1024}"
HELDSEED_MIN_SAMPLES="${HELDSEED_MIN_SAMPLES:-1024}"
HELDTASK_MIN_SAMPLES="${HELDTASK_MIN_SAMPLES:-450}"
MIN_CLUSTERS="${MIN_CLUSTERS:-100}"
MIN_FINAL_RETENTION="${MIN_FINAL_RETENTION:-0.70}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-4000,8000,12000}"
FINAL_CHECKPOINT_STEP="${FINAL_CHECKPOINT_STEP:-12000}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_queue_20260723.pid}"
TRAINING_CODE_MANIFEST="${TRAINING_CODE_MANIFEST:-${RUN_DIR}/training_code_manifest.sha256}"
RUNTIME_CODE_MANIFEST="${RUNTIME_CODE_MANIFEST:-${RUN_DIR}/training_runtime_code_manifest.sha256}"
EVALUATION_CODE_MANIFEST="${EVALUATION_CODE_MANIFEST:-${RUN_DIR}/evaluation_code_manifest.sha256}"
CODE_PROVENANCE_EXCEPTION="${CODE_PROVENANCE_EXCEPTION:-${RUN_DIR}/code_provenance_exception.json}"

if [[ "${RUN_DIR}" != /* || "${DATA}" != /* ]]; then
    echo "[collapse-monitor] run and data paths must be absolute" >&2
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
        echo "[collapse-monitor] training or evaluation code changed" >&2
        exit 4
    fi
}
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 2 ]]; then
    echo "[collapse-monitor] expected exactly two evaluation GPU ids" >&2
    exit 2
fi
IFS=',' read -r -a steps <<< "${CHECKPOINT_STEPS}"
if [[ "${#steps[@]}" -lt 1 ]]; then
    echo "[collapse-monitor] no checkpoint steps were requested" >&2
    exit 2
fi

cd "${ROOT}"
final_checkpoint="${RUN_DIR}/joint_$(printf '%07d' "${FINAL_CHECKPOINT_STEP}").pt"
while [[ ! -f "${final_checkpoint}" ]]; do
    if [[ -s "${QUEUE_PID_FILE}" ]] \
        && ! kill -0 "$(cat "${QUEUE_PID_FILE}")" 2>/dev/null; then
        echo "[collapse-monitor] training exited before ${final_checkpoint}" >&2
        exit 3
    fi
    printf '[collapse-monitor] %s waiting_for_final=%s\n' \
        "$(date --iso-8601=seconds)" "${final_checkpoint}"
    sleep "${POLL_SECONDS}"
done

for step in "${steps[@]}"; do
    checkpoint="${RUN_DIR}/joint_$(printf '%07d' "${step}").pt"
    while [[ ! -f "${checkpoint}" ]]; do
        if [[ -s "${QUEUE_PID_FILE}" ]] \
            && ! kill -0 "$(cat "${QUEUE_PID_FILE}")" 2>/dev/null; then
            echo "[collapse-monitor] training queue exited before ${checkpoint}" >&2
            exit 3
        fi
        printf '[collapse-monitor] %s waiting_for=%s\n' \
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
        printf '[collapse-monitor] %s free=%s threshold=%sMiB\n' \
            "$(date --iso-8601=seconds)" "${free[*]}" "${MIN_FREE_MIB}"
        sleep "${POLL_SECONDS}"
    done
    verify_code

    eval_dir="${RUN_DIR}/temporal_regions_step${step}"
    mkdir -p "${eval_dir}"
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
            echo "[collapse-monitor] preserving existing report: ${output}"
            continue
        fi
        temporary="${output}.tmp.$$"
        rm -f "${temporary}"
        CUDA_VISIBLE_DEVICES="${gpus[$index]}" .venv/bin/python \
            code/scripts/evaluate_visual_sequence_temporal_regions.py \
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
        echo "[collapse-monitor] evaluation failed at step ${step}" >&2
        exit "${status}"
    fi

    gate="${eval_dir}/collapse_gate.json"
    gate_args=(
        --heldseed "${eval_dir}/heldseed_${MAX_ITEMS}.json"
        --heldtask "${eval_dir}/heldtask_${MAX_ITEMS}.json"
        --train_log "${RUN_DIR}/train.jsonl"
        --output "${gate}"
        --minimum_heldseed_samples "${HELDSEED_MIN_SAMPLES}"
        --minimum_heldtask_samples "${HELDTASK_MIN_SAMPLES}"
        --minimum_clusters "${MIN_CLUSTERS}"
        --maximum_step "${step}"
    )
    if .venv/bin/python code/scripts/verify_visual_sequence_collapse_gate.py \
        "${gate_args[@]}" >"${eval_dir}/collapse_gate.log" 2>&1; then
        echo "[collapse-monitor] step=${step} gate=pass report=${gate}"
    else
        echo "[collapse-monitor] step=${step} gate=fail report=${gate}"
    fi
done

longitudinal="${RUN_DIR}/longitudinal_gate_step${FINAL_CHECKPOINT_STEP}.json"
if [[ -f "${longitudinal}" ]]; then
    echo "[collapse-monitor] preserving existing report: ${longitudinal}"
elif .venv/bin/python \
    code/scripts/verify_visual_sequence_longitudinal_gate.py \
    --step4000 "${RUN_DIR}/temporal_regions_step4000/collapse_gate.json" \
    --step8000 "${RUN_DIR}/temporal_regions_step8000/collapse_gate.json" \
    --step12000 "${RUN_DIR}/temporal_regions_step12000/collapse_gate.json" \
    --minimum_final_retention "${MIN_FINAL_RETENTION}" \
    --output "${longitudinal}" \
    >"${RUN_DIR}/longitudinal_gate_step${FINAL_CHECKPOINT_STEP}.log" 2>&1; then
    echo "[collapse-monitor] longitudinal gate=pass report=${longitudinal}"
else
    echo "[collapse-monitor] longitudinal gate=fail report=${longitudinal}"
fi
