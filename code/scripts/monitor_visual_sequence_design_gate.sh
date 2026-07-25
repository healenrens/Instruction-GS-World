#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${RUN_NAME}}"
SEQUENCE_DATA="${SEQUENCE_DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-12000}"
DYNAMICS_MAX_ITEMS="${DYNAMICS_MAX_ITEMS:-1024}"
HELDSEED_MIN_DYNAMICS_SAMPLES="${HELDSEED_MIN_DYNAMICS_SAMPLES:-1024}"
HELDTASK_MIN_DYNAMICS_SAMPLES="${HELDTASK_MIN_DYNAMICS_SAMPLES:-450}"
REPRESENTATION_MAX_ITEMS="${REPRESENTATION_MAX_ITEMS:-144}"
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_queue_20260723.pid}"
COLLAPSE_MONITOR_PID_FILE="${COLLAPSE_MONITOR_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_monitor_20260723.pid}"
COMPONENT_MONITOR_PID_FILE="${COMPONENT_MONITOR_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_component_monitor_20260723.pid}"
REPRESENTATION_MONITOR_PID_FILE="${REPRESENTATION_MONITOR_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_representation_monitor_20260723.pid}"
LONGITUDINAL_GATE="${LONGITUDINAL_GATE:-${RUN_DIR}/longitudinal_gate_step${CHECKPOINT_STEP}.json}"
TRANSFER_GATE="${TRANSFER_GATE:-${RUN_DIR}/component_ablation_step${CHECKPOINT_STEP}/transfer_gate.json}"
TRAINING_CODE_MANIFEST="${TRAINING_CODE_MANIFEST:-${RUN_DIR}/training_code_manifest.sha256}"
RUNTIME_CODE_MANIFEST="${RUNTIME_CODE_MANIFEST:-${RUN_DIR}/training_runtime_code_manifest.sha256}"
EVALUATION_CODE_MANIFEST="${EVALUATION_CODE_MANIFEST:-${RUN_DIR}/evaluation_code_manifest.sha256}"
CODE_PROVENANCE_EXCEPTION="${CODE_PROVENANCE_EXCEPTION:-${RUN_DIR}/code_provenance_exception.json}"

if [[ "${RUN_DIR}" != /* || "${SEQUENCE_DATA}" != /* ]]; then
    echo "[design-monitor] run and data paths must be absolute" >&2
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
        echo "[design-monitor] training or evaluation code changed" >&2
        exit 4
    fi
}

wait_for_file() {
    local label="$1"
    local path="$2"
    local pid_file="$3"
    while [[ ! -f "${path}" ]]; do
        if [[ -s "${pid_file}" ]] \
            && ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
            echo "[design-monitor] ${label} monitor exited before ${path}" >&2
            exit 3
        fi
        printf '[design-monitor] %s waiting_for=%s\n' \
            "$(date --iso-8601=seconds)" "${path}"
        sleep "${POLL_SECONDS}"
    done
}

checkpoint="${RUN_DIR}/joint_$(printf '%07d' "${CHECKPOINT_STEP}").pt"
while [[ ! -f "${checkpoint}" ]]; do
    if [[ -s "${QUEUE_PID_FILE}" ]] \
        && ! kill -0 "$(cat "${QUEUE_PID_FILE}")" 2>/dev/null; then
        echo "[design-monitor] training exited before ${checkpoint}" >&2
        exit 3
    fi
    printf '[design-monitor] %s waiting_for=%s\n' \
        "$(date --iso-8601=seconds)" "${checkpoint}"
    sleep "${POLL_SECONDS}"
done

collapse="${RUN_DIR}/temporal_regions_step${CHECKPOINT_STEP}/collapse_gate.json"
component_seed="${RUN_DIR}/component_ablation_step${CHECKPOINT_STEP}/heldseed_${DYNAMICS_MAX_ITEMS}.json"
component_task="${RUN_DIR}/component_ablation_step${CHECKPOINT_STEP}/heldtask_${DYNAMICS_MAX_ITEMS}.json"
representation_seed="${RUN_DIR}/representation_step${CHECKPOINT_STEP}/heldseed_${REPRESENTATION_MAX_ITEMS}.json"
representation_task="${RUN_DIR}/representation_step${CHECKPOINT_STEP}/heldtask_${REPRESENTATION_MAX_ITEMS}.json"
wait_for_file collapse "${collapse}" "${COLLAPSE_MONITOR_PID_FILE}"
wait_for_file component "${component_seed}" "${COMPONENT_MONITOR_PID_FILE}"
wait_for_file component "${component_task}" "${COMPONENT_MONITOR_PID_FILE}"
wait_for_file representation "${representation_seed}" "${REPRESENTATION_MONITOR_PID_FILE}"
wait_for_file representation "${representation_task}" "${REPRESENTATION_MONITOR_PID_FILE}"
wait_for_file longitudinal "${LONGITUDINAL_GATE}" "${COLLAPSE_MONITOR_PID_FILE}"
wait_for_file transfer "${TRANSFER_GATE}" "${COMPONENT_MONITOR_PID_FILE}"

while true; do
    free="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
    if (( free >= MIN_FREE_MIB )); then
        break
    fi
    printf '[design-monitor] %s gpu=%s free=%sMiB threshold=%sMiB\n' \
        "$(date --iso-8601=seconds)" "${GPU_ID}" "${free}" "${MIN_FREE_MIB}"
    sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
verify_code
causal="${RUN_DIR}/causal_contract_step${CHECKPOINT_STEP}.json"
if [[ ! -f "${causal}" ]]; then
    temporary_causal="${causal}.tmp.$$"
    rm -f "${temporary_causal}"
    if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" .venv/bin/python \
        code/scripts/verify_visual_sequence_core.py \
        --data "${SEQUENCE_DATA}" \
        --checkpoint "${checkpoint}" \
        --split heldseed \
        --samples 32 \
        --report "${temporary_causal}" \
        >"${RUN_DIR}/causal_contract_step${CHECKPOINT_STEP}.log" 2>&1; then
        rm -f "${temporary_causal}"
        echo "[design-monitor] causal contract evaluation failed" >&2
        exit 3
    fi
    if [[ ! -s "${temporary_causal}" || -e "${causal}" ]]; then
        rm -f "${temporary_causal}"
        echo "[design-monitor] causal report was not published" >&2
        exit 3
    fi
    mv "${temporary_causal}" "${causal}"
fi

design_gate="${RUN_DIR}/design_gate_step${CHECKPOINT_STEP}.json"
if [[ -f "${design_gate}" ]]; then
    echo "[design-monitor] preserving existing report: ${design_gate}"
    exit 0
fi
.venv/bin/python code/scripts/verify_visual_sequence_design_gate.py \
    --representation_heldseed "${representation_seed}" \
    --representation_heldtask "${representation_task}" \
    --component_heldseed "${component_seed}" \
    --component_heldtask "${component_task}" \
    --collapse_gate "${collapse}" \
    --longitudinal_gate "${LONGITUDINAL_GATE}" \
    --transfer_gate "${TRANSFER_GATE}" \
    --causal_contract "${causal}" \
    --sequence_data_root "${SEQUENCE_DATA}" \
    --minimum_representation_samples "${REPRESENTATION_MAX_ITEMS}" \
    --minimum_heldseed_dynamics_samples \
        "${HELDSEED_MIN_DYNAMICS_SAMPLES}" \
    --minimum_heldtask_dynamics_samples \
        "${HELDTASK_MIN_DYNAMICS_SAMPLES}" \
    --output "${design_gate}" \
    >"${RUN_DIR}/design_gate_step${CHECKPOINT_STEP}.log" 2>&1
echo "[design-monitor] gate=pass report=${design_gate}"
evidence_manifest="${RUN_DIR}/design_evidence_manifest.sha256"
if [[ -f "${evidence_manifest}" ]]; then
    echo "[design-monitor] preserving existing evidence: ${evidence_manifest}"
else
    ROOT="${ROOT}" RUN_DIR="${RUN_DIR}" \
        bash code/scripts/freeze_visual_sequence_design_evidence.sh
fi
