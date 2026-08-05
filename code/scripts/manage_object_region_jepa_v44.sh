#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
VIDEO_VAE_MODEL="${VIDEO_VAE_MODEL:-/mnt/pfs/public/xuhaoming/models/Wan2.2-TI2V-5B-Diffusers}"
VIDEO_VAE_CONTRACT="${VIDEO_VAE_CONTRACT:-${VIDEO_VAE_MODEL}/vae_contract.json}"
VIDEO_VAE_PYTHONPATH="${VIDEO_VAE_PYTHONPATH:-${RUNTIME_ROOT}/v44_runtime/diffusers-0.35.2}"
SEED="${SEED:-17}"
short_commit="$(git -C "${ROOT}" rev-parse --short=7 HEAD 2>/dev/null || printf unknown)"
RUN_NAME="${RUN_NAME:-object_region_jepa_v44_seed${SEED}_${short_commit}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v44_gates/${short_commit}_startup.json}"
PID_FILE="${LOG_ROOT}/background.pid"
LAUNCHER_LOG="${LOG_ROOT}/background.launcher.log"
LAUNCHER="${ROOT}/code/scripts/train_object_region_jepa_v44.sh"
VERIFIER="${ROOT}/code/scripts/verify_object_region_jepa_v44.py"
PY="${VENV_ROOT}/.venv/bin/python"
export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${VIDEO_VAE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"

require_paths() {
    local path
    for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}" "${VIDEO_VAE_MODEL}" \
        "${VIDEO_VAE_PYTHONPATH}"; do
        [[ "${path}" == /* && -d "${path}" ]] || {
            echo "[object-region-v44-manager] missing directory: ${path}" >&2
            return 2
        }
    done
    [[ -x "${PY}" && -f "${LAUNCHER}" && -f "${VERIFIER}" \
        && -f "${VIDEO_VAE_CONTRACT}" ]] || {
        echo "[object-region-v44-manager] runtime component is missing" >&2
        return 2
    }
}

read_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        tr -d '[:space:]' <"${PID_FILE}"
    fi
}
is_alive() { local pid="${1:-}"; [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; }

show_status() {
    local pid
    pid="$(read_pid)"
    if is_alive "${pid}"; then
        echo "[object-region-v44-manager] state=running pid=${pid}"
    elif [[ -n "${pid}" ]]; then
        echo "[object-region-v44-manager] state=stopped stale_pid=${pid}"
    else
        echo "[object-region-v44-manager] state=not_started"
    fi
    echo "[object-region-v44-manager] out=${OUT}"
    echo "[object-region-v44-manager] gate=${GATE_REPORT}"
    echo "[object-region-v44-manager] video_vae=${VIDEO_VAE_MODEL}"
    echo "[object-region-v44-manager] video_vae_pythonpath=${VIDEO_VAE_PYTHONPATH}"
    echo "[object-region-v44-manager] launcher_log=${LAUNCHER_LOG}"
    if [[ -f "${OUT}/checkpoint_manifest.json" ]]; then
        command cat "${OUT}/checkpoint_manifest.json"
    else
        echo "[object-region-v44-manager] checkpoint=not_yet_saved"
    fi
    if [[ -f "${OUT}/wandb_run_id.txt" ]]; then
        echo "[object-region-v44-manager] wandb_run_id=$(tr -d '[:space:]' <"${OUT}/wandb_run_id.txt")"
    fi
    if [[ -f "${LOG_ROOT}/train.log" ]]; then
        tail -n "${STATUS_LINES:-120}" "${LOG_ROOT}/train.log"
    elif [[ -f "${LAUNCHER_LOG}" ]]; then
        tail -n "${STATUS_LINES:-120}" "${LAUNCHER_LOG}"
    fi
}

run_verifier() {
    local -a command
    require_paths
    mkdir -p "$(dirname "${GATE_REPORT}")"
    command=(
        "${PY}" "${VERIFIER}"
        --data "${DATA}" --output "${GATE_REPORT}"
        --video_vae_model "${VIDEO_VAE_MODEL}"
        --video_vae_contract "${VIDEO_VAE_CONTRACT}"
        --video_vae_pythonpath "${VIDEO_VAE_PYTHONPATH}"
        --expected_local_gpus auto
        --jit_dino_batch "${JIT_DINO_BATCH:-16}"
        --video_vae_batch "${VIDEO_VAE_BATCH:-1}"
        --history_span_frames "${HISTORY_SPAN_FRAMES:-15,30,45}"
        --goal_query_seconds "${GOAL_QUERY_SECONDS:-6.0}"
        --goal_tail_guard_frames "${GOAL_TAIL_GUARD_FRAMES:-0}"
        --goal_probe_frames "${GOAL_PROBE_FRAMES:-3}"
        --goal_stability_threshold "${GOAL_STABILITY_THRESHOLD:-0.05}"
        --goal_gate_candidates "${GOAL_GATE_CANDIDATES:-32}"
        --goal_rollout_weight "${GOAL_ROLLOUT_WEIGHT:-1.0}"
        --path_consistency_weight "${PATH_CONSISTENCY_WEIGHT:-0.25}"
    )
    [[ -n "${TEACHER_SIDECAR:-}" ]] && command+=(--teacher_sidecar "${TEACHER_SIDECAR}")
    [[ -n "${INIT_FROM:-}" ]] && command+=(--init_from "${INIT_FROM}")
    echo "[object-region-v44-manager] verifier=${GATE_REPORT}"
    "${command[@]}"
}

start_background() {
    local auto_resume="$1" pid temporary pgid
    require_paths
    [[ -f "${GATE_REPORT}" ]] || {
        echo "[object-region-v44-manager] gate report is missing" >&2
        return 2
    }
    [[ -n "${WANDB_ENTITY:-}" ]] || {
        echo "[object-region-v44-manager] WANDB_ENTITY is missing" >&2
        return 2
    }
    command -v setsid >/dev/null || {
        echo "[object-region-v44-manager] setsid is required" >&2
        return 2
    }
    mkdir -p "${LOG_ROOT}"
    pid="$(read_pid)"
    if is_alive "${pid}"; then
        echo "[object-region-v44-manager] already running pid=${pid}" >&2
        return 2
    fi
    if [[ "${auto_resume}" == "0" ]]; then
        : >"${LAUNCHER_LOG}"
    else
        printf '\n[object-region-v44-manager] resume requested at %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${LAUNCHER_LOG}"
    fi
    nohup setsid env \
        ROOT="${ROOT}" RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${VENV_ROOT}" \
        DATA="${DATA}" VIDEO_VAE_MODEL="${VIDEO_VAE_MODEL}" \
        VIDEO_VAE_CONTRACT="${VIDEO_VAE_CONTRACT}" \
        VIDEO_VAE_PYTHONPATH="${VIDEO_VAE_PYTHONPATH}" \
        VIDEO_VAE_BATCH="${VIDEO_VAE_BATCH:-1}" SEED="${SEED}" \
        RUN_NAME="${RUN_NAME}" OUT="${OUT}" LOG_ROOT="${LOG_ROOT}" \
        AUTO_RESUME="${auto_resume}" RESUME= INIT_FROM="${INIT_FROM:-}" \
        GATE_REPORT="${GATE_REPORT}" TEACHER_SIDECAR="${TEACHER_SIDECAR:-}" \
        NPROC_PER_NODE="${NPROC_PER_NODE:-auto}" \
        BATCH_PER_GPU="${BATCH_PER_GPU:-4}" GRAD_ACCUM="${GRAD_ACCUM:-auto}" \
        TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}" \
        WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}" \
        JIT_DINO_BATCH="${JIT_DINO_BATCH:-16}" \
        MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-0}" \
        HISTORY_SPAN_FRAMES="${HISTORY_SPAN_FRAMES:-15,30,45}" \
        SHORT_HORIZON_FRAMES="${SHORT_HORIZON_FRAMES:-30}" \
        GOAL_QUERY_SECONDS="${GOAL_QUERY_SECONDS:-6.0}" \
        GOAL_TAIL_GUARD_FRAMES="${GOAL_TAIL_GUARD_FRAMES:-0}" \
        GOAL_PROBE_FRAMES="${GOAL_PROBE_FRAMES:-3}" \
        GOAL_STABILITY_THRESHOLD="${GOAL_STABILITY_THRESHOLD:-0.05}" \
        GOAL_ROLLOUT_WEIGHT="${GOAL_ROLLOUT_WEIGHT:-1.0}" \
        PATH_CONSISTENCY_WEIGHT="${PATH_CONSISTENCY_WEIGHT:-0.25}" \
        STEPS=50000 SAVE_EVERY="${SAVE_EVERY:-5000}" \
        RECOVERY_EVERY="${RECOVERY_EVERY:-500}" LOG_EVERY="${LOG_EVERY:-20}" \
        WANDB_MODE="${WANDB_MODE:-online}" \
        WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}" \
        WANDB_ENTITY="${WANDB_ENTITY}" WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}" \
        WANDB_GROUP="${WANDB_GROUP:-object-region-jepa-v44}" \
        WANDB_TAGS="${WANDB_TAGS:-}" WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}" \
        WANDB_RUN_ID= bash "${LAUNCHER}" >>"${LAUNCHER_LOG}" 2>&1 < /dev/null &
    pid=$!
    temporary="${PID_FILE}.tmp.$$"
    printf '%s\n' "${pid}" >"${temporary}"
    mv "${temporary}" "${PID_FILE}"
    sleep 2
    if ! is_alive "${pid}"; then
        wait "${pid}" || true
        echo "[object-region-v44-manager] launch failed; log=${LAUNCHER_LOG}" >&2
        tail -n 160 "${LAUNCHER_LOG}" >&2
        return 1
    fi
    pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
    [[ "${pgid}" == "${pid}" ]] || {
        kill "${pid}" 2>/dev/null || true
        echo "[object-region-v44-manager] process group differs" >&2
        return 1
    }
    echo "[object-region-v44-manager] started pid=${pid} auto_resume=${auto_resume}"
    echo "[object-region-v44-manager] log=${LAUNCHER_LOG}"
}

load_resume_settings() {
    local checkpoint
    local -a saved
    if [[ -f "${OUT}/checkpoint_manifest.json" ]]; then
        checkpoint="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "${OUT}/checkpoint_manifest.json")"
    else
        checkpoint="$(readlink -f "${OUT}/latest.pt")"
    fi
    mapfile -d '' -t saved < <(
        "${PY}" "${ROOT}/code/scripts/read_resume_launch_v44.py" --checkpoint "${checkpoint}"
    )
    [[ "${#saved[@]}" -eq 35 ]] || {
        echo "[object-region-v44-manager] invalid saved launch settings" >&2
        return 2
    }
    DATA="${saved[0]}"; GATE_REPORT="${saved[1]}"; TEACHER_SIDECAR="${saved[2]}"
    SEED="${saved[3]}"; BATCH_PER_GPU="${saved[4]}"; GRAD_ACCUM="${saved[5]}"
    TARGET_GLOBAL_BATCH="${saved[6]}"; WORKERS_PER_RANK="${saved[7]}"
    JIT_DINO_BATCH="${saved[8]}"; MAX_TRAIN_ITEMS="${saved[9]}"
    HISTORY_SPAN_FRAMES="${saved[10]}"; SHORT_HORIZON_FRAMES="${saved[11]}"
    GOAL_QUERY_SECONDS="${saved[12]}"; GOAL_TAIL_GUARD_FRAMES="${saved[13]}"
    GOAL_PROBE_FRAMES="${saved[14]}"; GOAL_STABILITY_THRESHOLD="${saved[15]}"
    GOAL_ROLLOUT_WEIGHT="${saved[16]}"; PATH_CONSISTENCY_WEIGHT="${saved[17]}"
    SAVE_EVERY="${saved[18]}"; RECOVERY_EVERY="${saved[19]}"; LOG_EVERY="${saved[20]}"
    WANDB_MODE="${saved[21]}"; WANDB_PROJECT="${saved[22]}"; WANDB_ENTITY="${saved[23]}"
    WANDB_NAME="${saved[24]}"; WANDB_GROUP="${saved[25]}"; WANDB_TAGS="${saved[26]}"
    WANDB_DIR="${saved[27]}"; VIDEO_VAE_MODEL="${saved[28]}"
    VIDEO_VAE_CONTRACT="${saved[29]}"; VIDEO_VAE_PYTHONPATH="${saved[30]}"
    VIDEO_VAE_BATCH="${saved[33]}"; NPROC_PER_NODE="${saved[34]}"; INIT_FROM=
}

stop_background() {
    local pid
    pid="$(read_pid)"
    if ! is_alive "${pid}"; then
        echo "[object-region-v44-manager] no live process"
        return 0
    fi
    kill -TERM -- "-${pid}"
    for _ in {1..60}; do is_alive "${pid}" || break; sleep 1; done
    is_alive "${pid}" && {
        echo "[object-region-v44-manager] process did not stop" >&2
        return 1
    }
    echo "[object-region-v44-manager] stopped pid=${pid}"
}

case "${ACTION}" in
    verify) run_verifier ;;
    start) start_background 0 ;;
    resume)
        [[ -e "${OUT}/latest.pt" || -f "${OUT}/checkpoint_manifest.json" ]] || {
            echo "[object-region-v44-manager] no resumable checkpoint" >&2
            false
        }
        load_resume_settings
        start_background 1
        ;;
    status) show_status ;;
    stop) stop_background ;;
    *) echo "usage: $0 {verify|start|resume|status|stop}" >&2; false ;;
esac
