#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
SCOPE="${SCOPE:-isolated}"
GATE_REPORT="${GATE_REPORT:-}"
DENSE_PREFLIGHT_REPORT="${DENSE_PREFLIGHT_REPORT:-}"
READOUT_GATE_REPORT="${READOUT_GATE_REPORT:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"

if [[ "${SCOPE}" != "isolated" && "${SCOPE}" != "joint" ]]; then
    echo "[dense-v30] SCOPE must be isolated or joint" >&2
    exit 2
fi
for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[dense-v30] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
if [[ ! -x "${VENV_ROOT}/.venv/bin/python" ]] \
    || [[ ! -x "${VENV_ROOT}/.venv/bin/torchrun" ]]; then
    echo "[dense-v30] VENV_ROOT has no usable .venv: ${VENV_ROOT}" >&2
    exit 2
fi
if [[ "${GATE_REPORT}" != /* || ! -f "${GATE_REPORT}" ]]; then
    echo "[dense-v30] GATE_REPORT is missing" >&2
    exit 2
fi
if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "[dense-v30] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi
if [[ -z "${RESUME}" && ( -z "${INIT_FROM}" || ! -f "${INIT_FROM}" ) ]]; then
    echo "[dense-v30] initial launch requires INIT_FROM" >&2
    exit 2
fi
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "[dense-v30] repository is not clean" >&2
    git -C "${ROOT}" status --short >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[dense-v30] episode manifest verification failed" >&2
    exit 2
fi

SEED="${SEED:-17}"
STEPS="${STEPS:-2000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
RECOVERY_EVERY="${RECOVERY_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-20}"
CORE_LR="${CORE_LR:-2e-5}"
ACTION_LR="${ACTION_LR:-2e-4}"
READOUT_LR="${READOUT_LR:-2e-4}"
CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-}"
READOUT_REGULARIZATION_WEIGHT="${READOUT_REGULARIZATION_WEIGHT:-0.01}"
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    TORCHRUN_NPROC=gpu
elif [[ "${NPROC_PER_NODE}" == "gpu" ]] \
    || [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    TORCHRUN_NPROC="${NPROC_PER_NODE}"
else
    echo "[dense-v30] invalid NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 2
fi
if [[ "${GRAD_ACCUM}" == "auto" ]]; then
    TRAINER_GRAD_ACCUM=0
elif [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    TRAINER_GRAD_ACCUM="${GRAD_ACCUM}"
else
    echo "[dense-v30] invalid GRAD_ACCUM=${GRAD_ACCUM}" >&2
    exit 2
fi
if [[ "${SCOPE}" == "isolated" ]]; then
    CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-1.0}"
    if [[ "${DENSE_PREFLIGHT_REPORT}" != /* \
        || ! -f "${DENSE_PREFLIGHT_REPORT}" ]]; then
        echo "[dense-v30] isolated launch requires DENSE_PREFLIGHT_REPORT" >&2
        exit 2
    fi
    if [[ -n "${READOUT_GATE_REPORT}" ]]; then
        echo "[dense-v30] isolated launch forbids READOUT_GATE_REPORT" >&2
        exit 2
    fi
else
    CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-0.2}"
    if [[ "${READOUT_GATE_REPORT}" != /* || ! -f "${READOUT_GATE_REPORT}" ]]; then
        echo "[dense-v30] joint launch requires READOUT_GATE_REPORT" >&2
        exit 2
    fi
fi

RUN_NAME="${RUN_NAME:-dense_object_readout_v30_${SCOPE}_seed${SEED}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-dense-object-readout-v30}"
WANDB_TAGS="${WANDB_TAGS:-object-memory,jepa,no-language,no-rgb,v30,dense-readout,${SCOPE}}"
for path in "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "[dense-v30] output paths must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ "${WANDB_MODE}" != "online" ]]; then
    echo "[dense-v30] W&B must run online" >&2
    exit 2
fi
if [[ -z "${WANDB_ENTITY}" ]]; then
    echo "[dense-v30] WANDB_ENTITY is required" >&2
    exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[dense-v30] W&B credentials are missing" >&2
    exit 2
fi

if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[dense-v30] RESUME checkpoint is missing" >&2
        exit 2
    fi
    if [[ "$(dirname "$(readlink -f "${RESUME}")")" \
        != "$(readlink -m "${OUT}")" ]]; then
        echo "[dense-v30] RESUME must belong to OUT" >&2
        exit 2
    fi
    if [[ -z "${WANDB_RUN_ID}" && -s "${OUT}/wandb_run_id.txt" ]]; then
        WANDB_RUN_ID="$(<"${OUT}/wandb_run_id.txt")"
    fi
    init_args=(--resume "$(readlink -f "${RESUME}")")
else
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[dense-v30] refusing to overwrite ${OUT}" >&2
        exit 2
    fi
    init_args=(--init_from "$(readlink -f "${INIT_FROM}")")
fi
sidecar_args=()
if [[ -n "${TEACHER_SIDECAR}" ]]; then
    sidecar_args=(--teacher_sidecar "${TEACHER_SIDECAR}")
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"

mkdir -p "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"
cd "${ROOT}"
cmd=(
    "${VENV_ROOT}/.venv/bin/torchrun" --standalone --nnodes 1
    --nproc_per_node "${TORCHRUN_NPROC}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}" --data_format sequence
    --history_frames 4 --future_frames 4 --sequence_anchors 3,5,8
    --out "${OUT}" --profile full --architecture object_memory_v1
    --training_stage readout --readout_scope "${SCOPE}"
    --representation_steps "${STEPS}" --joint_steps 0
    --batch "${BATCH_PER_GPU}" --grad_accum "${TRAINER_GRAD_ACCUM}"
    --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
    --lr "${CORE_LR}" --lr_floor "$(awk "BEGIN {print ${CORE_LR} * 0.1}")"
    --core_lr "${CORE_LR}" --action_lr "${ACTION_LR}" --readout_lr "${READOUT_LR}"
    --current_readout_weight "${CURRENT_READOUT_WEIGHT}"
    --readout_regularization_weight "${READOUT_REGULARIZATION_WEIGHT}"
    --carrier_support_weight 0 --carrier_compact_weight 0 --gaussian_children 1
    --basis_gate_report "" --carrier_preflight_report ""
    --dense_preflight_report "${DENSE_PREFLIGHT_REPORT}"
    --readout_gate_report "${READOUT_GATE_REPORT}"
    --warmup_steps 0 --warmup_fraction 0.05 --weight_decay 1e-4
    --save_every "${SAVE_EVERY}" --recovery_every "${RECOVERY_EVERY}"
    --log_every "${LOG_EVERY}" --seed "${SEED}" --amp bf16
    --language_condition off --rgb_supervision off
    --language_effect_weight 0 --zero_action_margin_weight 0
    --gate_report "${GATE_REPORT}"
    --wandb_mode "${WANDB_MODE}" --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}" --wandb_name "${WANDB_NAME}"
    --wandb_group "${WANDB_GROUP}" --wandb_tags "${WANDB_TAGS}"
    --wandb_run_id "${WANDB_RUN_ID}" --wandb_dir "${WANDB_DIR}"
    "${sidecar_args[@]}" "${init_args[@]}"
)
echo "[dense-v30] commit=$(git rev-parse HEAD) scope=${SCOPE} nproc=${TORCHRUN_NPROC}"
printf '[dense-v30] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/train.log") 2>&1
exec "${cmd[@]}"
