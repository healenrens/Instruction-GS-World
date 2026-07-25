#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${RUN_NAME}}"
FINAL_STEP="${FINAL_STEP:-12000}"
ACTIVE_PID_FILE="${ACTIVE_PID_FILE:-${ROOT}/logs/visual_sequence_collapse_repair_v27_active.pid}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-${ROOT}/code/scripts/queue_visual_sequence_collapse_repair_4gpu.sh}"
TRAINING_CODE_MANIFEST="${TRAINING_CODE_MANIFEST:-${CODE_MANIFEST:-${RUN_DIR}/training_code_manifest.sha256}}"
RUNTIME_CODE_MANIFEST="${RUNTIME_CODE_MANIFEST:-${RUN_DIR}/training_runtime_code_manifest.sha256}"
EVALUATION_CODE_MANIFEST="${EVALUATION_CODE_MANIFEST:-${RUN_DIR}/evaluation_code_manifest.sha256}"
CODE_PROVENANCE_EXCEPTION="${CODE_PROVENANCE_EXCEPTION:-${RUN_DIR}/code_provenance_exception.json}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FAILURE_GRACE_SECONDS="${FAILURE_GRACE_SECONDS:-30}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${RUN_DIR}" != /* || "${ACTIVE_PID_FILE}" != /* ]]; then
    echo "[collapse-supervisor] run and PID paths must be absolute" >&2
    exit 2
fi
if [[ ! -x "${QUEUE_SCRIPT}" \
      || ! -s "${ACTIVE_PID_FILE}" \
      || ! -s "${TRAINING_CODE_MANIFEST}" \
      || ! -s "${RUNTIME_CODE_MANIFEST}" \
      || ! -s "${EVALUATION_CODE_MANIFEST}" \
      || ! -s "${CODE_PROVENANCE_EXCEPTION}" ]]; then
    echo "[collapse-supervisor] queue, PID, or code manifest is missing" >&2
    exit 2
fi
if [[ "${MAX_RESTARTS}" -lt 0 || "${POLL_SECONDS}" -lt 1 ]]; then
    echo "[collapse-supervisor] restart and polling values are invalid" >&2
    exit 2
fi

write_active_pid() {
    local pid="$1"
    local temporary="${ACTIVE_PID_FILE}.tmp.$$"
    printf '%s\n' "${pid}" >"${temporary}"
    mv "${temporary}" "${ACTIVE_PID_FILE}"
}

verify_code() {
    "${ROOT}/.venv/bin/python" \
        "${ROOT}/code/scripts/verify_visual_sequence_code_provenance.py" \
        --root "${ROOT}" \
        --training_manifest "${TRAINING_CODE_MANIFEST}" \
        --runtime_manifest "${RUNTIME_CODE_MANIFEST}" \
        --evaluation_manifest "${EVALUATION_CODE_MANIFEST}" \
        --exception "${CODE_PROVENANCE_EXCEPTION}" >/dev/null
}

final_checkpoint="${RUN_DIR}/joint_$(printf '%07d' "${FINAL_STEP}").pt"
active_pid="$(cat "${ACTIVE_PID_FILE}")"
restarts=0
while true; do
    while kill -0 "${active_pid}" 2>/dev/null; do
        printf '[collapse-supervisor] %s active_pid=%s restarts=%s\n' \
            "$(date --iso-8601=seconds)" "${active_pid}" "${restarts}"
        sleep "${POLL_SECONDS}"
    done
    sleep "${FAILURE_GRACE_SECONDS}"
    if [[ -f "${final_checkpoint}" ]]; then
        echo "[collapse-supervisor] training complete: ${final_checkpoint}"
        exit 0
    fi
    if [[ ! -e "${RUN_DIR}/latest.pt" ]]; then
        echo "[collapse-supervisor] training exited before the first checkpoint" >&2
        exit 3
    fi
    if (( restarts >= MAX_RESTARTS )); then
        echo "[collapse-supervisor] exhausted ${MAX_RESTARTS} restarts" >&2
        exit 4
    fi
    if ! verify_code; then
        echo "[collapse-supervisor] training code changed; refusing resume" >&2
        exit 5
    fi
    restarts=$((restarts + 1))
    resume_log="${RUN_DIR}/resume_${restarts}.console.log"
    printf '[collapse-supervisor] %s restart=%s resume=%s\n' \
        "$(date --iso-8601=seconds)" "${restarts}" "${RUN_DIR}/latest.pt"
    env \
        ROOT="${ROOT}" \
        RUN_NAME="${RUN_NAME}" \
        OUT="${RUN_DIR}" \
        RESUME="${RUN_DIR}/latest.pt" \
        WANDB_MODE="${WANDB_MODE}" \
        bash "${QUEUE_SCRIPT}" >>"${resume_log}" 2>&1 &
    active_pid="$!"
    write_active_pid "${active_pid}"
done
