#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_dinov2l_native_30hz_v3}"
STAGE="${STAGE:-representation}"
GATE_REPORT="${GATE_REPORT:-}"
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"
REPRESENTATION_GATE_REPORT="${REPRESENTATION_GATE_REPORT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
SEED="${SEED:-17}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
RECOVERY_EVERY="${RECOVERY_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
REQUESTED_WANDB_RUN_ID="${WANDB_RUN_ID:-}"
unset WANDB_RUN_ID

if [[ "${STAGE}" != "representation" && "${STAGE}" != "posterior" ]]; then
    echo "[object-memory-v37] STAGE must be representation or posterior" >&2
    exit 2
fi
for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[object-memory-v37] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
if [[ ! -x "${PY}" || ! -x "${TORCHRUN}" ]]; then
    echo "[object-memory-v37] VENV_ROOT has no usable .venv: ${VENV_ROOT}" >&2
    exit 2
fi
if [[ "${GATE_REPORT}" != /* || ! -f "${GATE_REPORT}" ]]; then
    echo "[object-memory-v37] GATE_REPORT is missing" >&2
    exit 2
fi
if ! git -C "${ROOT}" diff --quiet \
    || ! git -C "${ROOT}" diff --cached --quiet; then
    echo "[object-memory-v37] tracked repository files are modified" >&2
    git -C "${ROOT}" status --short --untracked-files=no >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[object-memory-v37] episode manifest checksum failed" >&2
    exit 2
fi
if [[ -n "${TEACHER_SIDECAR}" ]] \
    && [[ "${TEACHER_SIDECAR}" != /* || ! -d "${TEACHER_SIDECAR}" ]]; then
    echo "[object-memory-v37] TEACHER_SIDECAR is invalid" >&2
    exit 2
fi
if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "[object-memory-v37] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    TORCHRUN_NPROC=gpu
elif [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    TORCHRUN_NPROC="${NPROC_PER_NODE}"
else
    echo "[object-memory-v37] NPROC_PER_NODE must be auto or positive" >&2
    exit 2
fi
if [[ "${GRAD_ACCUM}" == "auto" ]]; then
    TRAINER_GRAD_ACCUM=0
elif [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    TRAINER_GRAD_ACCUM="${GRAD_ACCUM}"
else
    echo "[object-memory-v37] GRAD_ACCUM must be auto or positive" >&2
    exit 2
fi

if [[ "${STAGE}" == "representation" ]]; then
    STEPS="${STEPS:-30000}"
    CORE_LR="${CORE_LR:-2e-4}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    if [[ -n "${INIT_FROM}" || -n "${REPRESENTATION_GATE_REPORT}" ]]; then
        echo "[object-memory-v37] fresh representation forbids init and held gate" >&2
        exit 2
    fi
    phase_args=(--representation_steps "${STEPS}" --joint_steps 0)
else
    STEPS="${STEPS:-30000}"
    CORE_LR="${CORE_LR:-2e-5}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    if [[ -z "${RESUME}" ]]; then
        if [[ "${INIT_FROM}" != /* || ! -f "${INIT_FROM}" ]]; then
            echo "[object-memory-v37] posterior requires representation INIT_FROM" >&2
            exit 2
        fi
        if [[ "${REPRESENTATION_GATE_REPORT}" != /* \
            || ! -f "${REPRESENTATION_GATE_REPORT}" ]]; then
            echo "[object-memory-v37] posterior requires a passed held gate" >&2
            exit 2
        fi
    fi
    phase_args=(--representation_steps 0 --joint_steps "${STEPS}")
fi
LR_FLOOR="$("${PY}" -c "print(float('${CORE_LR}') * 0.1)")"
short_commit="$(git -C "${ROOT}" rev-parse --short=7 HEAD)"
RUN_NAME="${RUN_NAME:-object_memory_jepa_v37_${STAGE}_seed${SEED}_${short_commit}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-object-memory-jepa-v37}"
WANDB_TAGS="${WANDB_TAGS:-object-memory,jepa,full-dino,change-residual,30hz,no-language,no-rgb,v37,${STAGE}}"
for path in "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "[object-memory-v37] output paths must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ "${WANDB_MODE}" != "online" || -z "${WANDB_ENTITY}" ]]; then
    echo "[object-memory-v37] online W&B and WANDB_ENTITY are required" >&2
    exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[object-memory-v37] W&B credentials are missing" >&2
    exit 2
fi

init_args=()
if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[object-memory-v37] RESUME checkpoint is missing" >&2
        exit 2
    fi
    if [[ "$(dirname "$(readlink -f "${RESUME}")")" \
        != "$(readlink -m "${OUT}")" ]]; then
        echo "[object-memory-v37] RESUME must belong to OUT" >&2
        exit 2
    fi
    init_args=(--resume "$(readlink -f "${RESUME}")")
else
    if [[ -e "${OUT}/latest.pt" || -e "${OUT}/run_contract.json" \
        || -e "${OUT}/train.jsonl" || -e "${OUT}/wandb_run_id.txt" ]]; then
        echo "[object-memory-v37] refusing to overwrite ${OUT}" >&2
        exit 2
    fi
    if [[ -n "${INIT_FROM}" ]]; then
        init_args=(--init_from "$(readlink -f "${INIT_FROM}")")
    fi
fi
sidecar_args=()
if [[ -n "${TEACHER_SIDECAR}" ]]; then
    sidecar_args=(--teacher_sidecar "${TEACHER_SIDECAR}")
fi
held_gate_args=()
if [[ -n "${REPRESENTATION_GATE_REPORT}" ]]; then
    held_gate_args=(--representation_gate_report "${REPRESENTATION_GATE_REPORT}")
fi
wandb_id_args=()
if [[ -n "${REQUESTED_WANDB_RUN_ID}" ]]; then
    wandb_id_args=(--wandb_run_id "${REQUESTED_WANDB_RUN_ID}")
fi

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
cd "${ROOT}"
cmd=(
    "${TORCHRUN}" --standalone --nnodes 1 --nproc_per_node "${TORCHRUN_NPROC}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}" --data_format sequence
    --history_frames 4 --future_frames 4 --sequence_anchors 3,5,8
    --out "${OUT}" --profile full --architecture object_memory_v1
    --training_stage "${STAGE}" --readout_scope off
    "${phase_args[@]}"
    --batch "${BATCH_PER_GPU}" --grad_accum "${TRAINER_GRAD_ACCUM}"
    --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
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
    "${sidecar_args[@]}" "${held_gate_args[@]}" "${wandb_id_args[@]}"
    "${init_args[@]}"
)
echo "[object-memory-v37] commit=$(git rev-parse HEAD) stage=${STAGE} nproc=${TORCHRUN_NPROC}"
printf '[object-memory-v37] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/train.log") 2>&1
exec "${cmd[@]}"
