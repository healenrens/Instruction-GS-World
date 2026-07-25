#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
required=(STAGE GATE_REPORT)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[object-memory-v28] missing environment variable: ${name}" >&2
        exit 2
    fi
done
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[object-memory-v28] NNODES must be a positive node count" >&2
    exit 2
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] \
    || [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
    echo "[object-memory-v28] invalid node rank: ${NODE_RANK}" >&2
    exit 2
fi
if ! [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] \
    || [[ "${MASTER_PORT}" -lt 1 || "${MASTER_PORT}" -gt 65535 ]]; then
    echo "[object-memory-v28] MASTER_PORT must be in 1..65535" >&2
    exit 2
fi
if [[ "${NNODES}" -gt 1 ]] \
    && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "[object-memory-v28] MASTER_ADDR must be reachable from node 1" >&2
    exit 2
fi
if [[ "${STAGE}" != "representation" && "${STAGE}" != "posterior" ]]; then
    echo "[object-memory-v28] STAGE must be representation or posterior" >&2
    exit 2
fi

DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    TORCHRUN_NPROC="gpu"
elif [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    TORCHRUN_NPROC="${NPROC_PER_NODE}"
else
    echo "[object-memory-v28] NPROC_PER_NODE must be auto or a positive integer" >&2
    exit 2
fi
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"
SEED="${SEED:-17}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
if [[ "${GRAD_ACCUM}" == "auto" ]]; then
    TRAINER_GRAD_ACCUM=0
elif [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    TRAINER_GRAD_ACCUM="${GRAD_ACCUM}"
else
    echo "[object-memory-v28] GRAD_ACCUM must be auto or a positive integer" >&2
    exit 2
fi
if ! [[ "${TARGET_GLOBAL_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[object-memory-v28] TARGET_GLOBAL_BATCH must be positive" >&2
    exit 2
fi
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-20}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
CONTRACT_WAIT_SECONDS="${CONTRACT_WAIT_SECONDS:-300}"

if [[ "${STAGE}" == "representation" ]]; then
    STEPS="${STEPS:-100000}"
    CORE_LR="${CORE_LR:-2e-4}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    LR_FLOOR="${LR_FLOOR:-2e-5}"
    RUN_NAME="${RUN_NAME:-object_memory_jepa_v28_representation_seed${SEED}}"
    phase_args=(--representation_steps "${STEPS}" --joint_steps 0)
else
    STEPS="${STEPS:-100000}"
    CORE_LR="${CORE_LR:-2e-5}"
    ACTION_LR="${ACTION_LR:-2e-4}"
    LR_FLOOR="${LR_FLOOR:-2e-6}"
    RUN_NAME="${RUN_NAME:-object_memory_jepa_v28_posterior_seed${SEED}}"
    phase_args=(--representation_steps 0 --joint_steps "${STEPS}")
fi
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_NAME}}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-object-memory-jepa-v28}"
WANDB_TAGS="${WANDB_TAGS:-object-memory,jepa,no-language,no-rgb,v28,ddp-auto,${STAGE}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"

for path in "${ROOT}" "${DATA}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[object-memory-v28] required absolute directory is missing: ${path}" >&2
        exit 2
    fi
done
for path in "${GATE_REPORT}"; do
    if [[ "${path}" != /* || ! -f "${path}" ]]; then
        echo "[object-memory-v28] required absolute file is missing: ${path}" >&2
        exit 2
    fi
done
if [[ -n "${TEACHER_SIDECAR}" ]] \
    && [[ "${TEACHER_SIDECAR}" != /* || ! -d "${TEACHER_SIDECAR}" ]]; then
    echo "[object-memory-v28] teacher sidecar directory is invalid" >&2
    exit 2
fi
if [[ "${OUT}" != /* || "${LOG_ROOT}" != /* || "${WANDB_DIR}" != /* ]]; then
    echo "[object-memory-v28] output, log, and W&B paths must be absolute" >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[object-memory-v28] verified episode manifest is missing or stale" >&2
    exit 2
fi
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "[object-memory-v28] repository is not clean" >&2
    git -C "${ROOT}" status --short >&2
    exit 2
fi

current_commit="$(git -C "${ROOT}" rev-parse HEAD)"
sidecar_manifest=""
if [[ -n "${TEACHER_SIDECAR}" ]]; then
    sidecar_manifest="${TEACHER_SIDECAR}/teacher_sidecar_manifest.json"
    if [[ ! -f "${sidecar_manifest}" ]]; then
        echo "[object-memory-v28] sidecar manifest is missing" >&2
        exit 2
    fi
fi
"${ROOT}/.venv/bin/python" - \
    "${GATE_REPORT}" "${current_commit}" \
    "${DATA}/episode_manifest.json" "${sidecar_manifest}" <<'PY'
import hashlib
import json
import sys

gate_path, commit, data_manifest, sidecar_manifest = sys.argv[1:]
with open(gate_path, encoding="utf-8") as handle:
    gate = json.load(handle)

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

expected = {
    "status": "passed",
    "architecture": "object_memory_v1",
    "checkpoint_version": 28,
    "git_commit": commit,
    "data_manifest_sha256": digest(data_manifest),
    "teacher_sidecar_sha256": digest(sidecar_manifest) if sidecar_manifest else "",
}
mismatch = {
    name: {"gate": gate.get(name), "current": value}
    for name, value in expected.items()
    if gate.get(name) != value
}
if mismatch:
    raise SystemExit(f"v28 gate mismatch: {mismatch}")
print(json.dumps(expected, sort_keys=True))
PY

if [[ "${WANDB_MODE}" != "online" ]]; then
    echo "[object-memory-v28] W&B must run online" >&2
    exit 2
fi
if ! "${ROOT}/.venv/bin/python" -c "import wandb" >/dev/null 2>&1; then
    echo "[object-memory-v28] wandb is not installed" >&2
    exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[object-memory-v28] online W&B credentials are missing" >&2
    exit 2
fi
if [[ "${CONTRACT_WAIT_SECONDS}" -le 0 ]]; then
    echo "[object-memory-v28] contract wait must be positive" >&2
    exit 2
fi

if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "[object-memory-v28] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi
if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[object-memory-v28] resume checkpoint is missing" >&2
        exit 2
    fi
    resume_dir="$(dirname "$(readlink -f "${RESUME}")")"
    if [[ "${resume_dir}" != "$(readlink -m "${OUT}")" ]]; then
        echo "[object-memory-v28] resume checkpoint must belong to OUT" >&2
        exit 2
    fi
    if [[ -z "${WANDB_RUN_ID}" && ! -s "${OUT}/wandb_run_id.txt" ]]; then
        echo "[object-memory-v28] resume requires the original W&B run id" >&2
        exit 2
    fi
    init_args=(--resume "$(readlink -f "${RESUME}")")
else
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[object-memory-v28] refusing to overwrite ${OUT}" >&2
        exit 2
    fi
    if [[ "${STAGE}" == "posterior" && -z "${INIT_FROM}" ]]; then
        echo "[object-memory-v28] posterior requires representation INIT_FROM" >&2
        exit 2
    fi
    init_args=()
    if [[ -n "${INIT_FROM}" ]]; then
        if [[ "${INIT_FROM}" != /* || ! -f "${INIT_FROM}" ]]; then
            echo "[object-memory-v28] initialization checkpoint is missing" >&2
            exit 2
        fi
        init_args=(--init_from "$(readlink -f "${INIT_FROM}")")
    fi
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

sidecar_args=()
if [[ -n "${TEACHER_SIDECAR}" ]]; then
    sidecar_args=(--teacher_sidecar "${TEACHER_SIDECAR}")
else
    echo "[object-memory-v28] sidecar=disabled; disparity/visibility teacher losses off"
fi
mkdir -p "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"
cd "${ROOT}"
cmd=(
    .venv/bin/torchrun
    --nproc_per_node "${TORCHRUN_NPROC}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}"
    --data_format sequence
    --history_frames 4
    --future_frames 4
    --sequence_anchors 3,5,8
    --out "${OUT}"
    --profile full
    --architecture object_memory_v1
    --training_stage "${STAGE}"
    "${phase_args[@]}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${TRAINER_GRAD_ACCUM}"
    --target_global_batch "${TARGET_GLOBAL_BATCH}"
    --workers "${WORKERS_PER_RANK}"
    --lr "${CORE_LR}"
    --lr_floor "${LR_FLOOR}"
    --core_lr "${CORE_LR}"
    --action_lr "${ACTION_LR}"
    --warmup_steps 0
    --warmup_fraction 0.05
    --weight_decay 1e-4
    --save_every "${SAVE_EVERY}"
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
echo "[object-memory-v28] commit=${current_commit} node=${NODE_RANK}/${NNODES} stage=${STAGE}"
echo "[object-memory-v28] nproc=${TORCHRUN_NPROC} grad_accum=${GRAD_ACCUM} target_global_batch=${TARGET_GLOBAL_BATCH} steps=${STEPS}"
printf '[object-memory-v28] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/node_${NODE_RANK}.log") 2>&1
exec "${cmd[@]}"
