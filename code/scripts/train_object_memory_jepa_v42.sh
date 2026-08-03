#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
STAGE="${STAGE:-representation}"
GATE_REPORT="${GATE_REPORT:-}"
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"
REPRESENTATION_GATE_REPORT="${REPRESENTATION_GATE_REPORT:-}"
POSTERIOR_GATE_REPORT="${POSTERIOR_GATE_REPORT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
SEED="${SEED:-17}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
JIT_DINO_BATCH="${JIT_DINO_BATCH:-16}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-0}"
HISTORY_SPAN_FRAMES="${HISTORY_SPAN_FRAMES:-15,30,45}"
SHORT_HORIZON_FRAMES="${SHORT_HORIZON_FRAMES:-30}"
GOAL_QUERY_SECONDS="${GOAL_QUERY_SECONDS:-6.0}"
GOAL_TAIL_GUARD_FRAMES="${GOAL_TAIL_GUARD_FRAMES:-0}"
GOAL_PROBE_FRAMES="${GOAL_PROBE_FRAMES:-3}"
GOAL_STABILITY_THRESHOLD="${GOAL_STABILITY_THRESHOLD:-0.05}"
GOAL_ROLLOUT_WEIGHT="${GOAL_ROLLOUT_WEIGHT:-1.0}"
PATH_CONSISTENCY_WEIGHT="${PATH_CONSISTENCY_WEIGHT:-0.25}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
RECOVERY_EVERY="${RECOVERY_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
REQUESTED_WANDB_RUN_ID="${WANDB_RUN_ID:-}"
unset WANDB_RUN_ID

if [[ "${STAGE}" != "representation" && "${STAGE}" != "posterior" \
    && "${STAGE}" != "prior" ]]; then
    echo "[object-memory-v42] STAGE must be representation, posterior, or prior" >&2
    exit 2
fi
if [[ "${AUTO_RESUME}" != "0" && "${AUTO_RESUME}" != "1" ]]; then
    echo "[object-memory-v42] AUTO_RESUME must be 0 or 1" >&2
    exit 2
fi
for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[object-memory-v42] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
if [[ ! -x "${PY}" || ! -x "${TORCHRUN}" ]]; then
    echo "[object-memory-v42] VENV_ROOT has no usable .venv: ${VENV_ROOT}" >&2
    exit 2
fi
if [[ "${GATE_REPORT}" != /* || ! -f "${GATE_REPORT}" ]]; then
    echo "[object-memory-v42] GATE_REPORT is missing" >&2
    exit 2
fi
if ! git -C "${ROOT}" diff --quiet \
    || ! git -C "${ROOT}" diff --cached --quiet; then
    echo "[object-memory-v42] tracked repository files are modified" >&2
    git -C "${ROOT}" status --short --untracked-files=no >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[object-memory-v42] RGB episode manifest checksum failed" >&2
    exit 2
fi
if [[ -n "${TEACHER_SIDECAR}" ]] \
    && [[ "${TEACHER_SIDECAR}" != /* || ! -d "${TEACHER_SIDECAR}" ]]; then
    echo "[object-memory-v42] TEACHER_SIDECAR is invalid" >&2
    exit 2
fi
VISIBLE_GPU_COUNT="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    TORCHRUN_NPROC=gpu
    RESUME_WORLD_SIZE="${VISIBLE_GPU_COUNT}"
elif [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    TORCHRUN_NPROC="${NPROC_PER_NODE}"
    RESUME_WORLD_SIZE="${NPROC_PER_NODE}"
else
    echo "[object-memory-v42] NPROC_PER_NODE must be auto or positive" >&2
    exit 2
fi
if [[ "${RESUME_WORLD_SIZE}" -lt 1 ]]; then
    echo "[object-memory-v42] no visible CUDA GPU" >&2
    exit 2
fi
if [[ "${RESUME_WORLD_SIZE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
    echo "[object-memory-v42] requested ${RESUME_WORLD_SIZE} ranks but only ${VISIBLE_GPU_COUNT} GPUs are visible" >&2
    exit 2
fi
if [[ "${GRAD_ACCUM}" == "auto" ]]; then
    TRAINER_GRAD_ACCUM=0
elif [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    TRAINER_GRAD_ACCUM="${GRAD_ACCUM}"
else
    echo "[object-memory-v42] GRAD_ACCUM must be auto or positive" >&2
    exit 2
fi

short_commit="$(git -C "${ROOT}" rev-parse --short=7 HEAD)"
RUN_NAME="${RUN_NAME:-object_memory_jepa_v42_${STAGE}_seed${SEED}_${short_commit}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-object-memory-jepa-v42}"
WANDB_TAGS="${WANDB_TAGS:-object-memory,jepa,jit-dino,dual-horizon,dynamic-history,30hz,no-language,no-rgb,v42,${STAGE}}"
for path in "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "[object-memory-v42] output paths must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ "${AUTO_RESUME}" == "1" && -z "${RESUME}" && -z "${INIT_FROM}" ]]; then
    if [[ -f "${OUT}/checkpoint_manifest.json" ]]; then
        RESUME="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "${OUT}/checkpoint_manifest.json")"
        temporary_link="${OUT}/latest.pt.resume.$$"
        ln -s "$(basename "${RESUME}")" "${temporary_link}"
        mv -Tf "${temporary_link}" "${OUT}/latest.pt"
    elif [[ -e "${OUT}/latest.pt" ]]; then
        RESUME="${OUT}/latest.pt"
    fi
fi
if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "[object-memory-v42] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi

if [[ "${STAGE}" == "representation" ]]; then
    STEPS="${STEPS:-15000}"
    CORE_LR="${CORE_LR:-2e-4}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    if [[ -z "${RESUME}" ]] \
        && [[ "${INIT_FROM}" != /* || ! -f "${INIT_FROM}" ]]; then
        echo "[object-memory-v42] representation requires a clean v39/v40 INIT_FROM" >&2
        exit 2
    fi
    if [[ -n "${REPRESENTATION_GATE_REPORT}" \
        || -n "${POSTERIOR_GATE_REPORT}" ]]; then
        echo "[object-memory-v42] representation forbids held promotion gates" >&2
        exit 2
    fi
    phase_args=(--representation_steps "${STEPS}" --joint_steps 0)
elif [[ "${STAGE}" == "posterior" ]]; then
    STEPS="${STEPS:-15000}"
    CORE_LR="${CORE_LR:-2e-5}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    if [[ -z "${RESUME}" ]]; then
        if [[ "${INIT_FROM}" != /* || ! -f "${INIT_FROM}" ]]; then
            echo "[object-memory-v42] posterior requires representation INIT_FROM" >&2
            exit 2
        fi
        if [[ "${REPRESENTATION_GATE_REPORT}" != /* \
            || ! -f "${REPRESENTATION_GATE_REPORT}" ]]; then
            echo "[object-memory-v42] posterior requires a representation held gate" >&2
            exit 2
        fi
    fi
    phase_args=(--representation_steps 0 --joint_steps "${STEPS}")
else
    STEPS="${STEPS:-15000}"
    CORE_LR="${CORE_LR:-2e-4}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    if [[ -z "${RESUME}" ]]; then
        if [[ "${INIT_FROM}" != /* || ! -f "${INIT_FROM}" ]]; then
            echo "[object-memory-v42] prior requires posterior INIT_FROM" >&2
            exit 2
        fi
        if [[ "${POSTERIOR_GATE_REPORT}" != /* \
            || ! -f "${POSTERIOR_GATE_REPORT}" ]]; then
            echo "[object-memory-v42] prior requires a posterior held gate" >&2
            exit 2
        fi
    fi
    phase_args=(--representation_steps 0 --joint_steps "${STEPS}")
fi
LR_FLOOR="$("${PY}" -c "print(float('${CORE_LR}') * 0.1)")"
if [[ "${WANDB_MODE}" != "online" || -z "${WANDB_ENTITY}" ]]; then
    echo "[object-memory-v42] online W&B and WANDB_ENTITY are required" >&2
    exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[object-memory-v42] W&B credentials are missing" >&2
    exit 2
fi

init_args=()
if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[object-memory-v42] RESUME checkpoint is missing" >&2
        exit 2
    fi
    RESUME="$(readlink -f "${RESUME}")"
    "${PY}" "${ROOT}/code/scripts/verify_resume_checkpoint_v42.py" \
        --out "${OUT}" --checkpoint "${RESUME}" \
        --git_commit "$(git -C "${ROOT}" rev-parse HEAD)" \
        --world_size "${RESUME_WORLD_SIZE}" --stage "${STAGE}"
    init_args=(--resume "${RESUME}")
else
    if [[ -e "${OUT}/latest.pt" || -e "${OUT}/run_contract.json" \
        || -e "${OUT}/train.jsonl" || -e "${OUT}/wandb_run_id.txt" ]]; then
        echo "[object-memory-v42] refusing to overwrite ${OUT}" >&2
        exit 2
    fi
    if [[ -n "${INIT_FROM}" ]]; then
        init_args=(--init_from "$(readlink -f "${INIT_FROM}")")
    fi
fi
sidecar_args=()
[[ -n "${TEACHER_SIDECAR}" ]] && sidecar_args=(--teacher_sidecar "${TEACHER_SIDECAR}")
promotion_args=()
[[ -n "${REPRESENTATION_GATE_REPORT}" ]] && promotion_args+=(--representation_gate_report "${REPRESENTATION_GATE_REPORT}")
[[ -n "${POSTERIOR_GATE_REPORT}" ]] && promotion_args+=(--posterior_gate_report "${POSTERIOR_GATE_REPORT}")
wandb_id_args=()
[[ -n "${REQUESTED_WANDB_RUN_ID}" ]] && wandb_id_args=(--wandb_run_id "${REQUESTED_WANDB_RUN_ID}")

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"

mkdir -p "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"
command -v flock >/dev/null || { echo "[object-memory-v42] flock is required" >&2; exit 2; }
exec 9>"${OUT}/.v42_train.lock"
flock -n 9 || { echo "[object-memory-v42] another process owns ${OUT}" >&2; exit 2; }
cd "${ROOT}"
cmd=(
    "${TORCHRUN}" --standalone --nnodes 1 --nproc_per_node "${TORCHRUN_NPROC}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}" --data_format sequence --feature_source jit
    --jit_dino_batch "${JIT_DINO_BATCH}"
    --temporal_contract dynamic_dual_horizon_v1
    --history_frames 4 --history_frames_min 1 --history_frames_max 4
    --history_span_frames "${HISTORY_SPAN_FRAMES}"
    --future_frames 2 --short_horizon_frames "${SHORT_HORIZON_FRAMES}"
    --goal_query_seconds "${GOAL_QUERY_SECONDS}"
    --goal_tail_guard_frames "${GOAL_TAIL_GUARD_FRAMES}"
    --goal_probe_frames "${GOAL_PROBE_FRAMES}"
    --goal_stability_threshold "${GOAL_STABILITY_THRESHOLD}"
    --goal_rollout_weight "${GOAL_ROLLOUT_WEIGHT}"
    --path_consistency_weight "${PATH_CONSISTENCY_WEIGHT}"
    --out "${OUT}" --profile full --architecture object_memory_v3
    --training_stage "${STAGE}" --readout_scope off "${phase_args[@]}"
    --batch "${BATCH_PER_GPU}" --grad_accum "${TRAINER_GRAD_ACCUM}"
    --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
    --max_train_items "${MAX_TRAIN_ITEMS}"
    --lr "${CORE_LR}" --lr_floor "${LR_FLOOR}"
    --core_lr "${CORE_LR}" --action_lr "${ACTION_LR}" --readout_lr "${CORE_LR}"
    --current_readout_weight 0 --readout_regularization_weight 0
    --carrier_support_weight 0 --carrier_compact_weight 0 --gaussian_children 1
    --warmup_steps 0 --warmup_fraction 0.05 --weight_decay 1e-4
    --save_every "${SAVE_EVERY}" --recovery_every "${RECOVERY_EVERY}"
    --log_every "${LOG_EVERY}" --seed "${SEED}" --amp bf16
    --language_condition off --rgb_supervision off
    --language_effect_weight 0 --zero_action_margin_weight 0
    --gate_report "${GATE_REPORT}"
    --wandb_mode "${WANDB_MODE}" --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}" --wandb_name "${WANDB_NAME}"
    --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}"
    --wandb_dir "${WANDB_DIR}"
    "${sidecar_args[@]}" "${promotion_args[@]}" "${wandb_id_args[@]}"
    "${init_args[@]}"
)
echo "[object-memory-v42] commit=$(git rev-parse HEAD) stage=${STAGE} nproc=${TORCHRUN_NPROC}"
printf '[object-memory-v42] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/train.log") 2>&1
exec "${cmd[@]}"
