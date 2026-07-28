#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
SCOPE="${SCOPE:-isolated}"
GATE_REPORT="${GATE_REPORT:-}"
BASIS_GATE_REPORT="${BASIS_GATE_REPORT:-}"
CARRIER_PREFLIGHT_REPORT="${CARRIER_PREFLIGHT_REPORT:-}"
READOUT_GATE_REPORT="${READOUT_GATE_REPORT:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"

if [[ "${SCOPE}" != "isolated" && "${SCOPE}" != "joint" ]]; then
    echo "[carrier-v29] SCOPE must be isolated or joint" >&2
    exit 2
fi
for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[carrier-v29] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
if [[ ! -x "${VENV_ROOT}/.venv/bin/python" ]] \
    || [[ ! -x "${VENV_ROOT}/.venv/bin/torchrun" ]]; then
    echo "[carrier-v29] VENV_ROOT has no usable .venv: ${VENV_ROOT}" >&2
    exit 2
fi
if [[ -z "${GATE_REPORT}" || "${GATE_REPORT}" != /* \
    || ! -f "${GATE_REPORT}" ]]; then
    echo "[carrier-v29] GATE_REPORT is missing" >&2
    exit 2
fi
if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "[carrier-v29] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi
if [[ -z "${RESUME}" && ( -z "${INIT_FROM}" || ! -f "${INIT_FROM}" ) ]]; then
    echo "[carrier-v29] initial launch requires INIT_FROM" >&2
    exit 2
fi
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "[carrier-v29] repository is not clean" >&2
    git -C "${ROOT}" status --short >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[carrier-v29] episode manifest verification failed" >&2
    exit 2
fi

SEED="${SEED:-17}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
RECOVERY_EVERY="${RECOVERY_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-20}"
READOUT_LR="${READOUT_LR:-2e-4}"
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    TORCHRUN_NPROC="gpu"
elif [[ "${NPROC_PER_NODE}" == "gpu" ]] \
    || [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    TORCHRUN_NPROC="${NPROC_PER_NODE}"
else
    echo "[carrier-v29] NPROC_PER_NODE must be auto, gpu, or positive" >&2
    exit 2
fi
if [[ "${GRAD_ACCUM}" == "auto" ]]; then
    TRAINER_GRAD_ACCUM=0
elif [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    TRAINER_GRAD_ACCUM="${GRAD_ACCUM}"
else
    echo "[carrier-v29] GRAD_ACCUM must be auto or positive" >&2
    exit 2
fi

if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[carrier-v29] RESUME checkpoint is missing" >&2
        exit 2
    fi
    mapfile -t SAVED < <(
        "${VENV_ROOT}/.venv/bin/python" - "${RESUME}" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False, mmap=True
)
args = checkpoint["args"]
values = (
    args["readout_scope"],
    args["representation_steps"],
    args["batch"],
    args["grad_accum"],
    args["target_global_batch"],
    args["core_lr"],
    args["action_lr"],
    args["readout_lr"],
    args["lr_floor"],
    args["current_readout_weight"],
    args["readout_regularization_weight"],
    args["carrier_support_weight"],
    args["carrier_compact_weight"],
    args["gaussian_children"],
    args["seed"],
    args.get("basis_gate_report", ""),
    args.get("carrier_preflight_report", ""),
    args.get("teacher_sidecar", ""),
)
for value in values:
    print(value)
PY
    )
    if [[ "${SAVED[0]}" != "${SCOPE}" ]]; then
        echo "[carrier-v29] SCOPE differs from resume checkpoint" >&2
        exit 2
    fi
    STEPS="${SAVED[1]}"
    BATCH_PER_GPU="${SAVED[2]}"
    TRAINER_GRAD_ACCUM="${SAVED[3]}"
    TARGET_GLOBAL_BATCH="${SAVED[4]}"
    CORE_LR="${SAVED[5]}"
    ACTION_LR="${SAVED[6]}"
    READOUT_LR="${SAVED[7]}"
    LR_FLOOR="${SAVED[8]}"
    CURRENT_READOUT_WEIGHT="${SAVED[9]}"
    READOUT_REGULARIZATION_WEIGHT="${SAVED[10]}"
    CARRIER_SUPPORT_WEIGHT="${SAVED[11]}"
    CARRIER_COMPACT_WEIGHT="${SAVED[12]}"
    GAUSSIAN_CHILDREN="${SAVED[13]}"
    SEED="${SAVED[14]}"
    BASIS_GATE_REPORT="${SAVED[15]}"
    CARRIER_PREFLIGHT_REPORT="${SAVED[16]}"
    TEACHER_SIDECAR="${SAVED[17]}"
    READOUT_GATE_REPORT=""
else
    if [[ "${SCOPE}" == "isolated" ]]; then
        READOUT_GATE_REPORT=""
        for path in "${BASIS_GATE_REPORT}" "${CARRIER_PREFLIGHT_REPORT}"; do
            if [[ "${path}" != /* || ! -f "${path}" ]]; then
                echo "[carrier-v29] isolated launch is missing a gate: ${path}" >&2
                exit 2
            fi
        done
        GAUSSIAN_CHILDREN="$(
            "${VENV_ROOT}/.venv/bin/python" -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["selected_children"])' \
                "${BASIS_GATE_REPORT}"
        )"
        STEPS="${STEPS:-2000}"
        CORE_LR="${CORE_LR:-2e-5}"
        ACTION_LR="${ACTION_LR:-2e-4}"
        LR_FLOOR="${LR_FLOOR:-2e-6}"
        CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-1.0}"
        READOUT_REGULARIZATION_WEIGHT="${READOUT_REGULARIZATION_WEIGHT:-0.0}"
        CARRIER_SUPPORT_WEIGHT="${CARRIER_SUPPORT_WEIGHT:-0.2}"
        CARRIER_COMPACT_WEIGHT="${CARRIER_COMPACT_WEIGHT:-0.001}"
    else
        BASIS_GATE_REPORT=""
        CARRIER_PREFLIGHT_REPORT=""
        if [[ "${READOUT_GATE_REPORT}" != /* || ! -f "${READOUT_GATE_REPORT}" ]]; then
            echo "[carrier-v29] joint launch requires READOUT_GATE_REPORT" >&2
            exit 2
        fi
        GAUSSIAN_CHILDREN="$(
            "${VENV_ROOT}/.venv/bin/python" -c \
                'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True); print(c["config"]["gaussian_children"])' \
                "${INIT_FROM}"
        )"
        STEPS="${STEPS:-5000}"
        CORE_LR="${CORE_LR:-2e-5}"
        ACTION_LR="${ACTION_LR:-2e-4}"
        LR_FLOOR="${LR_FLOOR:-2e-6}"
        CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-0.25}"
        READOUT_REGULARIZATION_WEIGHT="${READOUT_REGULARIZATION_WEIGHT:-0.0}"
        CARRIER_SUPPORT_WEIGHT="${CARRIER_SUPPORT_WEIGHT:-0.05}"
        CARRIER_COMPACT_WEIGHT="${CARRIER_COMPACT_WEIGHT:-0.001}"
    fi
fi

RUN_NAME="${RUN_NAME:-hierarchical_gaussian_carrier_v29_${SCOPE}_seed${SEED}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-hierarchical-gaussian-carrier-v29}"
WANDB_TAGS="${WANDB_TAGS:-object-memory,jepa,no-language,no-rgb,v29,carrier,${SCOPE}}"
for path in "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "[carrier-v29] runtime path must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ "${WANDB_MODE}" != "online" ]]; then
    echo "[carrier-v29] W&B must run online" >&2
    exit 2
fi
if ! "${VENV_ROOT}/.venv/bin/python" -c 'import wandb' >/dev/null 2>&1; then
    echo "[carrier-v29] wandb is not installed" >&2
    exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[carrier-v29] W&B credentials are missing" >&2
    exit 2
fi

if [[ -n "${RESUME}" ]]; then
    if [[ "$(dirname "$(readlink -f "${RESUME}")")" \
        != "$(readlink -m "${OUT}")" ]]; then
        echo "[carrier-v29] RESUME must belong to OUT" >&2
        exit 2
    fi
    if [[ -z "${WANDB_RUN_ID}" && ! -s "${OUT}/wandb_run_id.txt" ]]; then
        echo "[carrier-v29] resume requires the original W&B run id" >&2
        exit 2
    fi
    init_args=(--resume "$(readlink -f "${RESUME}")")
else
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[carrier-v29] refusing to overwrite ${OUT}" >&2
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
    "${VENV_ROOT}/.venv/bin/torchrun"
    --standalone
    --nnodes 1
    --nproc_per_node "${TORCHRUN_NPROC}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}"
    --data_format sequence
    --history_frames 4
    --future_frames 4
    --sequence_anchors 3,5,8
    --out "${OUT}"
    --profile full
    --architecture object_memory_v1
    --training_stage readout
    --readout_scope "${SCOPE}"
    --representation_steps "${STEPS}"
    --joint_steps 0
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${TRAINER_GRAD_ACCUM}"
    --target_global_batch "${TARGET_GLOBAL_BATCH}"
    --workers "${WORKERS_PER_RANK}"
    --lr "${CORE_LR}"
    --lr_floor "${LR_FLOOR}"
    --core_lr "${CORE_LR}"
    --action_lr "${ACTION_LR}"
    --readout_lr "${READOUT_LR}"
    --current_readout_weight "${CURRENT_READOUT_WEIGHT}"
    --readout_regularization_weight "${READOUT_REGULARIZATION_WEIGHT}"
    --carrier_support_weight "${CARRIER_SUPPORT_WEIGHT}"
    --carrier_compact_weight "${CARRIER_COMPACT_WEIGHT}"
    --gaussian_children "${GAUSSIAN_CHILDREN}"
    --basis_gate_report "${BASIS_GATE_REPORT}"
    --carrier_preflight_report "${CARRIER_PREFLIGHT_REPORT}"
    --readout_gate_report "${READOUT_GATE_REPORT}"
    --warmup_steps 0
    --warmup_fraction 0.05
    --weight_decay 1e-4
    --save_every "${SAVE_EVERY}"
    --recovery_every "${RECOVERY_EVERY}"
    --log_every "${LOG_EVERY}"
    --seed "${SEED}"
    --amp bf16
    --language_condition off
    --rgb_supervision off
    --language_effect_weight 0.0
    --zero_action_margin_weight 0.0
    --gate_report "${GATE_REPORT}"
    --wandb_mode "${WANDB_MODE}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_name "${WANDB_NAME}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_tags "${WANDB_TAGS}"
    --wandb_run_id "${WANDB_RUN_ID}"
    --wandb_dir "${WANDB_DIR}"
    "${sidecar_args[@]}"
    "${init_args[@]}"
)
current_commit="$(git rev-parse HEAD)"
echo "[carrier-v29] commit=${current_commit} scope=${SCOPE} children=${GAUSSIAN_CHILDREN} nproc=${TORCHRUN_NPROC}"
printf '[carrier-v29] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/train.log") 2>&1
exec "${cmd[@]}"
