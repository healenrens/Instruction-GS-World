#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
STAGE="${STAGE:-representation}"
SEED="${SEED:-17}"
short_commit="$(git -C "${ROOT}" rev-parse --short=7 HEAD 2>/dev/null || true)"
RUN_NAME="${RUN_NAME:-object_memory_jepa_v39_${STAGE}_seed${SEED}_${short_commit}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
PID_FILE="${LOG_ROOT}/background.pid"
LAUNCHER_LOG="${LOG_ROOT}/background.launcher.log"
LAUNCHER="${ROOT}/code/scripts/train_object_memory_jepa_v39.sh"
PY="${VENV_ROOT}/.venv/bin/python"

require_paths() {
    for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
        if [[ "${path}" != /* || ! -d "${path}" ]]; then
            echo "[object-memory-v39-manager] missing absolute directory: ${path}" >&2
            return 2
        fi
    done
    [[ -f "${LAUNCHER}" ]] || {
        echo "[object-memory-v39-manager] launcher is missing: ${LAUNCHER}" >&2
        return 2
    }
    [[ "${GATE_REPORT:-}" == /* && -f "${GATE_REPORT}" ]] || {
        echo "[object-memory-v39-manager] GATE_REPORT is missing" >&2
        return 2
    }
    [[ -n "${WANDB_ENTITY:-}" ]] || {
        echo "[object-memory-v39-manager] WANDB_ENTITY is missing" >&2
        return 2
    }
    command -v setsid >/dev/null || {
        echo "[object-memory-v39-manager] setsid is required" >&2
        return 2
    }
}

read_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        tr -d '[:space:]' <"${PID_FILE}"
    fi
}

is_alive() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null
}

show_status() {
    local pid
    pid="$(read_pid)"
    if is_alive "${pid}"; then
        echo "[object-memory-v39-manager] state=running pid=${pid}"
    elif [[ -n "${pid}" ]]; then
        echo "[object-memory-v39-manager] state=stopped stale_pid=${pid}"
    else
        echo "[object-memory-v39-manager] state=not_started"
    fi
    echo "[object-memory-v39-manager] out=${OUT}"
    echo "[object-memory-v39-manager] launcher_log=${LAUNCHER_LOG}"
    if [[ -f "${OUT}/checkpoint_manifest.json" ]]; then
        echo "===== ${OUT}/checkpoint_manifest.json ====="
        cat "${OUT}/checkpoint_manifest.json"
    else
        echo "[object-memory-v39-manager] checkpoint=not_yet_saved"
    fi
    if [[ -f "${OUT}/wandb_run_id.txt" ]]; then
        echo "[object-memory-v39-manager] wandb_run_id=$(tr -d '[:space:]' <"${OUT}/wandb_run_id.txt")"
    fi
    if [[ -f "${LOG_ROOT}/train.log" ]]; then
        echo "===== ${LOG_ROOT}/train.log ====="
        tail -n "${STATUS_LINES:-100}" "${LOG_ROOT}/train.log"
    elif [[ -f "${LAUNCHER_LOG}" ]]; then
        tail -n "${STATUS_LINES:-100}" "${LAUNCHER_LOG}"
    fi
}

start_background() {
    local auto_resume="$1"
    local pgid pid temporary
    require_paths
    mkdir -p "${LOG_ROOT}"
    pid="$(read_pid)"
    if is_alive "${pid}"; then
        echo "[object-memory-v39-manager] already running pid=${pid}" >&2
        return 2
    fi
    : >"${LAUNCHER_LOG}"
    nohup setsid env \
        ROOT="${ROOT}" RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${VENV_ROOT}" \
        DATA="${DATA}" STAGE="${STAGE}" SEED="${SEED}" RUN_NAME="${RUN_NAME}" \
        OUT="${OUT}" LOG_ROOT="${LOG_ROOT}" AUTO_RESUME="${auto_resume}" \
        RESUME= INIT_FROM="${INIT_FROM:-}" GATE_REPORT="${GATE_REPORT:-}" \
        REPRESENTATION_GATE_REPORT="${REPRESENTATION_GATE_REPORT:-}" \
        POSTERIOR_GATE_REPORT="${POSTERIOR_GATE_REPORT:-}" \
        TEACHER_SIDECAR="${TEACHER_SIDECAR:-}" \
        NPROC_PER_NODE="${NPROC_PER_NODE:-auto}" \
        BATCH_PER_GPU="${BATCH_PER_GPU:-4}" GRAD_ACCUM="${GRAD_ACCUM:-auto}" \
        TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}" \
        WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}" \
        JIT_DINO_BATCH="${JIT_DINO_BATCH:-16}" MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-0}" \
        HISTORY_SPAN_FRAMES="${HISTORY_SPAN_FRAMES:-15,30,45}" \
        SHORT_HORIZON_FRAMES="${SHORT_HORIZON_FRAMES:-30}" \
        GOAL_QUERY_SECONDS="${GOAL_QUERY_SECONDS:-6.0}" \
        GOAL_TAIL_GUARD_FRAMES="${GOAL_TAIL_GUARD_FRAMES:-0}" \
        GOAL_PROBE_FRAMES="${GOAL_PROBE_FRAMES:-3}" \
        GOAL_STABILITY_THRESHOLD="${GOAL_STABILITY_THRESHOLD:-0.05}" \
        GOAL_ROLLOUT_WEIGHT="${GOAL_ROLLOUT_WEIGHT:-1.0}" \
        PATH_CONSISTENCY_WEIGHT="${PATH_CONSISTENCY_WEIGHT:-0.25}" \
        STEPS="${STEPS:-30000}" CORE_LR="${CORE_LR:-}" ACTION_LR="${ACTION_LR:-}" \
        SAVE_EVERY="${SAVE_EVERY:-5000}" RECOVERY_EVERY="${RECOVERY_EVERY:-500}" \
        LOG_EVERY="${LOG_EVERY:-20}" WANDB_MODE="${WANDB_MODE:-online}" \
        WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}" \
        WANDB_ENTITY="${WANDB_ENTITY:-}" WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}" \
        WANDB_GROUP="${WANDB_GROUP:-object-memory-jepa-v39}" \
        WANDB_TAGS="${WANDB_TAGS:-}" WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}" \
        WANDB_RUN_ID= \
        bash "${LAUNCHER}" >>"${LAUNCHER_LOG}" 2>&1 < /dev/null &
    pid=$!
    temporary="${PID_FILE}.tmp.$$"
    printf '%s\n' "${pid}" >"${temporary}"
    mv "${temporary}" "${PID_FILE}"
    sleep 2
    if ! is_alive "${pid}"; then
        if wait "${pid}"; then
            echo "[object-memory-v39-manager] process completed during startup"
            return 0
        fi
        echo "[object-memory-v39-manager] launch failed; log=${LAUNCHER_LOG}" >&2
        tail -n 160 "${LAUNCHER_LOG}" >&2
        return 1
    fi
    pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
    if [[ "${pgid}" != "${pid}" ]]; then
        echo "[object-memory-v39-manager] background process group differs" >&2
        kill "${pid}" 2>/dev/null || true
        return 1
    fi
    echo "[object-memory-v39-manager] started pid=${pid} auto_resume=${auto_resume}"
    echo "[object-memory-v39-manager] log=${LAUNCHER_LOG}"
}

stop_background() {
    local pid
    pid="$(read_pid)"
    if ! is_alive "${pid}"; then
        echo "[object-memory-v39-manager] no live process"
        return 0
    fi
    kill -TERM -- "-${pid}"
    for _ in {1..60}; do
        is_alive "${pid}" || break
        sleep 1
    done
    if is_alive "${pid}"; then
        echo "[object-memory-v39-manager] process did not stop after 60 seconds" >&2
        return 1
    fi
    echo "[object-memory-v39-manager] stopped pid=${pid}"
}

load_resume_settings() {
    local checkpoint
    local -a saved
    [[ -d "${ROOT}" && -x "${PY}" ]] || {
        echo "[object-memory-v39-manager] source or Python environment is missing" >&2
        return 2
    }
    if [[ -f "${OUT}/checkpoint_manifest.json" ]]; then
        checkpoint="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "${OUT}/checkpoint_manifest.json")"
    else
        checkpoint="$(readlink -f "${OUT}/latest.pt")"
    fi
    mapfile -d '' -t saved < <(
        "${PY}" "${ROOT}/code/scripts/read_resume_launch_v39.py" \
            --checkpoint "${checkpoint}"
    )
    if [[ "${#saved[@]}" -ne 35 ]]; then
        echo "[object-memory-v39-manager] invalid saved launch settings" >&2
        return 2
    fi
    STAGE="${saved[0]}"; DATA="${saved[1]}"; GATE_REPORT="${saved[2]}"
    TEACHER_SIDECAR="${saved[3]}"; REPRESENTATION_GATE_REPORT="${saved[4]}"
    POSTERIOR_GATE_REPORT="${saved[5]}"; SEED="${saved[6]}"
    BATCH_PER_GPU="${saved[7]}"; GRAD_ACCUM="${saved[8]}"
    TARGET_GLOBAL_BATCH="${saved[9]}"; WORKERS_PER_RANK="${saved[10]}"
    JIT_DINO_BATCH="${saved[11]}"; STEPS="${saved[12]}"; CORE_LR="${saved[13]}"
    ACTION_LR="${saved[14]}"; SAVE_EVERY="${saved[15]}"
    RECOVERY_EVERY="${saved[16]}"; LOG_EVERY="${saved[17]}"
    NPROC_PER_NODE="${saved[18]}"; WANDB_MODE="${saved[19]}"
    WANDB_PROJECT="${saved[20]}"; WANDB_ENTITY="${saved[21]}"
    WANDB_NAME="${saved[22]}"; WANDB_GROUP="${saved[23]}"
    WANDB_TAGS="${saved[24]}"; WANDB_DIR="${saved[25]}"
    HISTORY_SPAN_FRAMES="${saved[26]}"; SHORT_HORIZON_FRAMES="${saved[27]}"
    GOAL_QUERY_SECONDS="${saved[28]}"; GOAL_TAIL_GUARD_FRAMES="${saved[29]}"
    GOAL_PROBE_FRAMES="${saved[30]}"; GOAL_STABILITY_THRESHOLD="${saved[31]}"
    GOAL_ROLLOUT_WEIGHT="${saved[32]}"; PATH_CONSISTENCY_WEIGHT="${saved[33]}"
    MAX_TRAIN_ITEMS="${saved[34]}"; INIT_FROM=
}

case "${ACTION}" in
    start) start_background 0 ;;
    resume)
        if [[ ! -e "${OUT}/latest.pt" \
            && ! -f "${OUT}/checkpoint_manifest.json" ]]; then
            echo "[object-memory-v39-manager] no resumable checkpoint in ${OUT}" >&2
            exit 2
        fi
        load_resume_settings
        start_background 1
        ;;
    status) show_status ;;
    stop) stop_background ;;
    *) echo "usage: $0 {start|resume|status|stop}" >&2; exit 2 ;;
esac
