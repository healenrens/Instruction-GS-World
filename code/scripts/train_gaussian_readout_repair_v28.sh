#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_no_language_v1}"
SCOPE="${SCOPE:-isolated}"
GATE_REPORT="${GATE_REPORT:-}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-}"
READOUT_GATE_REPORT="${READOUT_GATE_REPORT:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"

if [[ "${SCOPE}" != "isolated" && "${SCOPE}" != "joint" ]]; then
    echo "[readout-repair-v28] SCOPE must be isolated or joint" >&2
    exit 2
fi
for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[readout-repair-v28] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
if [[ ! -x "${VENV_ROOT}/.venv/bin/python" ]] \
    || [[ ! -x "${VENV_ROOT}/.venv/bin/torchrun" ]]; then
    echo "[readout-repair-v28] VENV_ROOT has no usable .venv: ${VENV_ROOT}" >&2
    exit 2
fi
if [[ -z "${GATE_REPORT}" || "${GATE_REPORT}" != /* || ! -f "${GATE_REPORT}" ]]; then
    echo "[readout-repair-v28] GATE_REPORT is missing" >&2
    exit 2
fi
if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "[readout-repair-v28] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi
if [[ -z "${RESUME}" && ( -z "${INIT_FROM}" || ! -f "${INIT_FROM}" ) ]]; then
    echo "[readout-repair-v28] initial launch requires INIT_FROM" >&2
    exit 2
fi
if [[ "${SCOPE}" == "isolated" && -z "${RESUME}" ]] \
    && [[ -z "${PREFLIGHT_REPORT}" || "${PREFLIGHT_REPORT}" != /* \
        || ! -f "${PREFLIGHT_REPORT}" ]]; then
    echo "[readout-repair-v28] isolated launch requires PREFLIGHT_REPORT" >&2
    exit 2
fi
if [[ "${SCOPE}" == "joint" ]] \
    && [[ -z "${READOUT_GATE_REPORT}" || "${READOUT_GATE_REPORT}" != /* \
        || ! -f "${READOUT_GATE_REPORT}" ]]; then
    echo "[readout-repair-v28] joint launch requires READOUT_GATE_REPORT" >&2
    exit 2
fi
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "[readout-repair-v28] repository is not clean" >&2
    git -C "${ROOT}" status --short >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[readout-repair-v28] episode manifest verification failed" >&2
    exit 2
fi

current_commit="$(git -C "${ROOT}" rev-parse HEAD)"
"${VENV_ROOT}/.venv/bin/python" - \
    "${GATE_REPORT}" "${current_commit}" "${DATA}/episode_manifest.json" <<'PY'
import hashlib
import json
import sys

gate_path, commit, manifest = sys.argv[1:]
with open(gate_path, encoding="utf-8") as handle:
    gate = json.load(handle)
digest = hashlib.sha256()
with open(manifest, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
expected = {
    "status": "passed",
    "architecture": "object_memory_v1",
    "checkpoint_version": 28,
    "checkpoint_contract": "rolling_recovery_v1",
    "git_commit": commit,
    "data_manifest_sha256": digest.hexdigest(),
}
mismatch = {
    name: {"gate": gate.get(name), "current": value}
    for name, value in expected.items()
    if gate.get(name) != value
}
if mismatch:
    raise SystemExit(f"base gate mismatch: {mismatch}")
PY

if [[ "${SCOPE}" == "isolated" && -z "${RESUME}" ]]; then
    "${VENV_ROOT}/.venv/bin/python" - \
        "${PREFLIGHT_REPORT}" "${current_commit}" "${INIT_FROM}" <<'PY'
import hashlib
import json
import sys

report_path, commit, checkpoint = sys.argv[1:]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
digest = hashlib.sha256()
with open(checkpoint, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
expected = {
    "status": "passed",
    "contract": "gaussian_readout_preflight_v1",
    "git_commit": commit,
    "init_checkpoint_sha256": digest.hexdigest(),
}
mismatch = {
    name: {"preflight": report.get(name), "current": value}
    for name, value in expected.items()
    if report.get(name) != value
}
if mismatch:
    raise SystemExit(f"readout preflight mismatch: {mismatch}")
PY
fi

SEED="${SEED:-17}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
RECOVERY_EVERY="${RECOVERY_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-20}"
READOUT_LR="${READOUT_LR:-2e-4}"
READOUT_REGULARIZATION_WEIGHT="${READOUT_REGULARIZATION_WEIGHT:-0.01}"
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    NPROC_PER_NODE="gpu"
elif [[ "${NPROC_PER_NODE}" != "gpu" ]] \
    && ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[readout-repair-v28] NPROC_PER_NODE must be auto, gpu, or positive" >&2
    exit 2
fi

if [[ "${SCOPE}" == "isolated" ]]; then
    STEPS="${STEPS:-2000}"
    CORE_LR="${CORE_LR:-2e-5}"
    CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-1.0}"
else
    STEPS="${STEPS:-5000}"
    CORE_LR="${CORE_LR:-2e-5}"
    CURRENT_READOUT_WEIGHT="${CURRENT_READOUT_WEIGHT:-0.25}"
fi
LR_FLOOR="${LR_FLOOR:-2e-6}"
RUN_NAME="${RUN_NAME:-gaussian_readout_v28_${SCOPE}_seed${SEED}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-gaussian-readout-repair-v28}"
WANDB_TAGS="${WANDB_TAGS:-object-memory,jepa,no-language,no-rgb,v28,readout,${SCOPE}}"
for path in "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "[readout-repair-v28] runtime paths must be absolute: ${path}" >&2
        exit 2
    fi
done

if [[ "${WANDB_MODE}" != "online" ]]; then
    echo "[readout-repair-v28] W&B must run online" >&2
    exit 2
fi
if ! "${VENV_ROOT}/.venv/bin/python" -c "import wandb" >/dev/null 2>&1; then
    echo "[readout-repair-v28] wandb is not installed" >&2
    exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[readout-repair-v28] W&B credentials are missing" >&2
    exit 2
fi

if [[ -n "${RESUME}" ]]; then
    if [[ ! -f "${RESUME}" ]]; then
        echo "[readout-repair-v28] RESUME checkpoint is missing" >&2
        exit 2
    fi
    if [[ "$(dirname "$(readlink -f "${RESUME}")")" != "$(readlink -m "${OUT}")" ]]; then
        echo "[readout-repair-v28] RESUME must belong to OUT" >&2
        exit 2
    fi
    if [[ -z "${WANDB_RUN_ID}" && ! -s "${OUT}/wandb_run_id.txt" ]]; then
        echo "[readout-repair-v28] resume requires the original W&B run id" >&2
        exit 2
    fi
    init_args=(--resume "$(readlink -f "${RESUME}")")
else
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[readout-repair-v28] refusing to overwrite ${OUT}" >&2
        exit 2
    fi
    init_args=(--init_from "$(readlink -f "${INIT_FROM}")")
fi

sidecar_args=()
if [[ -n "${TEACHER_SIDECAR}" ]]; then
    sidecar_args=(--teacher_sidecar "${TEACHER_SIDECAR}")
fi
readout_gate_args=()
if [[ "${SCOPE}" == "joint" ]]; then
    readout_gate_args=(--readout_gate_report "${READOUT_GATE_REPORT}")
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
    --nproc_per_node "${NPROC_PER_NODE}"
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
    --grad_accum 0
    --target_global_batch "${TARGET_GLOBAL_BATCH}"
    --workers "${WORKERS_PER_RANK}"
    --lr "${CORE_LR}"
    --lr_floor "${LR_FLOOR}"
    --core_lr "${CORE_LR}"
    --action_lr 2e-4
    --readout_lr "${READOUT_LR}"
    --current_readout_weight "${CURRENT_READOUT_WEIGHT}"
    --readout_regularization_weight "${READOUT_REGULARIZATION_WEIGHT}"
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
    "${readout_gate_args[@]}"
    "${sidecar_args[@]}"
    "${init_args[@]}"
)
echo "[readout-repair-v28] commit=${current_commit} scope=${SCOPE} nproc=${NPROC_PER_NODE} steps=${STEPS}"
printf '[readout-repair-v28] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/train.log") 2>&1
exec "${cmd[@]}"
