#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pfs/public/fanyupeng/dataset/robotwin2_lerobot}"
SOURCE_VARIANTS="${SOURCE_VARIANTS:-demo_clean,demo_randomized}"
EXPECTED_SOURCE_FPS="${EXPECTED_SOURCE_FPS:-30}"
OUT="${OUT:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/rt2_visual_episodes_rgb_native_30hz_v4}"
PID_FILE="${LOG_ROOT}/background.pid"
LAUNCHER_LOG="${LOG_ROOT}/background.launcher.log"
LAUNCHER="${ROOT}/code/scripts/cache_rt2_rgb_episodes_v38.sh"

require_paths() {
    for path in "${ROOT}" "${RUNTIME_ROOT}" "${SOURCE_ROOT}"; do
        if [[ "${path}" != /* || ! -d "${path}" ]]; then
            echo "[v38-rgb-manager] missing absolute directory: ${path}" >&2
            return 2
        fi
    done
    if [[ ! -f "${LAUNCHER}" ]]; then
        echo "[v38-rgb-manager] launcher is missing: ${LAUNCHER}" >&2
        return 2
    fi
    command -v setsid >/dev/null || {
        echo "[v38-rgb-manager] setsid is required" >&2
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
        echo "[v38-rgb-manager] state=running pid=${pid}"
    elif [[ -n "${pid}" ]]; then
        echo "[v38-rgb-manager] state=stopped stale_pid=${pid}"
    else
        echo "[v38-rgb-manager] state=not_started"
    fi
    echo "[v38-rgb-manager] data=${OUT}"
    echo "[v38-rgb-manager] launcher_log=${LAUNCHER_LOG}"
    if [[ -f "${OUT}/episode_manifest.verified.sha256" ]] \
        && (cd "${OUT}" && sha256sum -c --status episode_manifest.verified.sha256); then
        echo "[v38-rgb-manager] cache=complete"
    elif [[ -f "${OUT}/episode_manifest.pending.json" ]]; then
        echo "[v38-rgb-manager] cache=partial_resumable"
    else
        echo "[v38-rgb-manager] cache=absent_or_unprepared"
    fi
    if [[ -f "${LAUNCHER_LOG}" ]]; then
        tail -n "${STATUS_LINES:-80}" "${LAUNCHER_LOG}"
    fi
    for log in "${LOG_ROOT}"/shard_*.log; do
        [[ -f "${log}" ]] || continue
        echo "===== ${log} ====="
        tail -n "${SHARD_STATUS_LINES:-8}" "${log}"
    done
}

start_background() {
    local mode="$1"
    local pgid pid temporary
    require_paths
    mkdir -p "${LOG_ROOT}"
    pid="$(read_pid)"
    if is_alive "${pid}"; then
        echo "[v38-rgb-manager] already running pid=${pid}" >&2
        return 2
    fi
    : >"${LAUNCHER_LOG}"
    nohup setsid env \
        ROOT="${ROOT}" RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${VENV_ROOT}" \
        SOURCE_ROOT="${SOURCE_ROOT}" SOURCE_VARIANTS="${SOURCE_VARIANTS}" \
        EXPECTED_SOURCE_FPS="${EXPECTED_SOURCE_FPS}" OUT="${OUT}" \
        LOG_ROOT="${LOG_ROOT}" CACHE_MODE="${mode}" \
        CACHE_JOBS="${CACHE_JOBS:-8}" FRAME_BATCH="${FRAME_BATCH:-32}" \
        JPEG_WORKERS="${JPEG_WORKERS:-2}" JPEG_QUALITY="${JPEG_QUALITY:-95}" \
        WINDOW_LENGTHS="${WINDOW_LENGTHS:-45,90,135,180}" \
        SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}" OVERWRITE="${OVERWRITE:-0}" \
        bash "${LAUNCHER}" >>"${LAUNCHER_LOG}" 2>&1 < /dev/null &
    pid=$!
    temporary="${PID_FILE}.tmp.$$"
    printf '%s\n' "${pid}" >"${temporary}"
    mv "${temporary}" "${PID_FILE}"
    sleep 1
    if ! is_alive "${pid}"; then
        if wait "${pid}"; then
            echo "[v38-rgb-manager] process completed during startup"
            return 0
        fi
        echo "[v38-rgb-manager] launch failed; log=${LAUNCHER_LOG}" >&2
        tail -n 120 "${LAUNCHER_LOG}" >&2
        return 1
    fi
    pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
    if [[ "${pgid}" != "${pid}" ]]; then
        echo "[v38-rgb-manager] background process group differs" >&2
        kill "${pid}" 2>/dev/null || true
        return 1
    fi
    echo "[v38-rgb-manager] started mode=${mode} pid=${pid}"
    echo "[v38-rgb-manager] log=${LAUNCHER_LOG}"
}

stop_background() {
    local pid
    pid="$(read_pid)"
    if ! is_alive "${pid}"; then
        echo "[v38-rgb-manager] no live process"
        return 0
    fi
    kill -TERM -- "-${pid}"
    for _ in {1..30}; do
        is_alive "${pid}" || break
        sleep 1
    done
    if is_alive "${pid}"; then
        echo "[v38-rgb-manager] process did not stop after 30 seconds" >&2
        return 1
    fi
    echo "[v38-rgb-manager] stopped pid=${pid}; partial episodes remain resumable"
}

load_resume_settings() {
    local -a saved
    local pending="${OUT}/episode_manifest.pending.json"
    if [[ ! -x "${VENV_ROOT}/.venv/bin/python" ]]; then
        echo "[v38-rgb-manager] Python environment is missing" >&2
        return 2
    fi
    mapfile -d '' -t saved < <(
        "${VENV_ROOT}/.venv/bin/python" \
            "${ROOT}/code/scripts/read_rgb_cache_resume_v38.py" \
            --pending "${pending}"
    )
    if [[ "${#saved[@]}" -ne 6 ]]; then
        echo "[v38-rgb-manager] invalid saved cache settings" >&2
        return 2
    fi
    SOURCE_ROOT="${saved[0]}"
    SOURCE_VARIANTS="${saved[1]}"
    EXPECTED_SOURCE_FPS="${saved[2]}"
    WINDOW_LENGTHS="${saved[3]}"
    SAMPLE_STRIDE="${saved[4]}"
    JPEG_QUALITY="${saved[5]}"
}

case "${ACTION}" in
    start) start_background fresh ;;
    resume)
        load_resume_settings
        start_background resume
        ;;
    status) show_status ;;
    stop) stop_background ;;
    *)
        echo "usage: $0 {start|resume|status|stop}" >&2
        exit 2
        ;;
esac
