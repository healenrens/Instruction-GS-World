#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
VIDEO_VAE_MODEL="${VIDEO_VAE_MODEL:-/mnt/pfs/public/xuhaoming/models/Wan2.2-TI2V-5B-Diffusers}"
VIDEO_VAE_CONTRACT="${VIDEO_VAE_CONTRACT:-${VIDEO_VAE_MODEL}/vae_contract.json}"
VIDEO_VAE_PYTHONPATH="${VIDEO_VAE_PYTHONPATH:-${RUNTIME_ROOT}/v44_runtime/diffusers-0.35.2}"
VIDEO_VAE_BATCH="${VIDEO_VAE_BATCH:-1}"
GATE_REPORT="${GATE_REPORT:-}"
TEACHER_SIDECAR="${TEACHER_SIDECAR:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"
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
STEPS="${STEPS:-50000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
RECOVERY_EVERY="${RECOVERY_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
REQUESTED_WANDB_RUN_ID="${WANDB_RUN_ID:-}"
unset WANDB_RUN_ID

CORE_LR=2e-5
DINO_LR=2e-6
NEW_MODULE_LR=2e-4
ACTION_LR=2e-4
LR_FLOOR=2e-6

[[ "${AUTO_RESUME}" =~ ^[01]$ ]] || {
    echo "[object-region-v44] AUTO_RESUME must be 0 or 1" >&2
    exit 2
}
[[ "${STEPS}" == "50000" ]] || {
    echo "[object-region-v44] v44 requires STEPS=50000" >&2
    exit 2
}
for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}" "${VIDEO_VAE_MODEL}" \
    "${VIDEO_VAE_PYTHONPATH}"; do
    [[ "${path}" == /* && -d "${path}" ]] || {
        echo "[object-region-v44] missing absolute directory: ${path}" >&2
        exit 2
    }
done
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
[[ -x "${PY}" && -x "${TORCHRUN}" ]] || {
    echo "[object-region-v44] VENV_ROOT has no usable .venv" >&2
    exit 2
}
export PYTHONPATH="${VIDEO_VAE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" -c 'import diffusers; from diffusers import AutoencoderKLWan; assert diffusers.__version__ == "0.35.2"' || {
    echo "[object-region-v44] existing environment lacks AutoencoderKLWan" >&2
    exit 2
}
[[ "${VIDEO_VAE_CONTRACT}" == /* && -f "${VIDEO_VAE_CONTRACT}" ]] || {
    echo "[object-region-v44] VIDEO_VAE_CONTRACT is missing" >&2
    exit 2
}
PYTHONPATH="${ROOT}/code:${PYTHONPATH}" "${PY}" -c \
    'import sys; from igsw.adaptive_gaussian_wm.video_vae_contract import validate_video_vae_artifact; validate_video_vae_artifact(sys.argv[1], sys.argv[2], verify_hashes=True)' \
    "${VIDEO_VAE_MODEL}" "${VIDEO_VAE_CONTRACT}"
[[ "${GATE_REPORT}" == /* && -f "${GATE_REPORT}" ]] || {
    echo "[object-region-v44] GATE_REPORT is missing" >&2
    exit 2
}
if ! git -C "${ROOT}" diff --quiet || ! git -C "${ROOT}" diff --cached --quiet; then
    echo "[object-region-v44] tracked repository files are modified" >&2
    git -C "${ROOT}" status --short --untracked-files=no >&2
    exit 2
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! (cd "${DATA}" && sha256sum -c --status episode_manifest.verified.sha256); then
    echo "[object-region-v44] RGB episode manifest checksum failed" >&2
    exit 2
fi
if [[ -n "${TEACHER_SIDECAR}" ]] \
    && [[ "${TEACHER_SIDECAR}" != /* || ! -d "${TEACHER_SIDECAR}" ]]; then
    echo "[object-region-v44] TEACHER_SIDECAR is invalid" >&2
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
    echo "[object-region-v44] NPROC_PER_NODE must be auto or positive" >&2
    exit 2
fi
[[ "${RESUME_WORLD_SIZE}" -ge 1 && "${RESUME_WORLD_SIZE}" -le "${VISIBLE_GPU_COUNT}" ]] || {
    echo "[object-region-v44] visible/requested GPU count is invalid" >&2
    exit 2
}
if [[ "${GRAD_ACCUM}" == "auto" ]]; then
    TRAINER_GRAD_ACCUM=0
elif [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    TRAINER_GRAD_ACCUM="${GRAD_ACCUM}"
else
    echo "[object-region-v44] GRAD_ACCUM must be auto or positive" >&2
    exit 2
fi

short_commit="$(git -C "${ROOT}" rev-parse --short=7 HEAD)"
RUN_NAME="${RUN_NAME:-object_region_jepa_v44_seed${SEED}_${short_commit}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-object-region-jepa-v44}"
WANDB_TAGS="${WANDB_TAGS:-object-region,jepa,dino,wan-vae,dual-encoder,30hz,v44}"

if [[ "${AUTO_RESUME}" == "1" && -z "${RESUME}" ]]; then
    if [[ -f "${OUT}/checkpoint_manifest.json" ]]; then
        RESUME="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "${OUT}/checkpoint_manifest.json")"
    elif [[ -e "${OUT}/latest.pt" ]]; then
        RESUME="${OUT}/latest.pt"
    fi
fi
[[ -z "${RESUME}" || -z "${INIT_FROM}" ]] || {
    echo "[object-region-v44] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
}
[[ "${WANDB_MODE}" == "online" && -n "${WANDB_ENTITY}" ]] || {
    echo "[object-region-v44] online W&B and WANDB_ENTITY are required" >&2
    exit 2
}
if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[object-region-v44] W&B credentials are missing" >&2
    exit 2
fi

init_args=()
if [[ -n "${RESUME}" ]]; then
    [[ "${RESUME}" == /* && -f "${RESUME}" ]] || {
        echo "[object-region-v44] RESUME checkpoint is missing" >&2
        exit 2
    }
    RESUME="$(readlink -f "${RESUME}")"
    "${PY}" "${ROOT}/code/scripts/verify_resume_checkpoint_v44.py" \
        --out "${OUT}" --checkpoint "${RESUME}" \
        --git_commit "$(git -C "${ROOT}" rev-parse HEAD)" \
        --world_size "${RESUME_WORLD_SIZE}"
    init_args=(--resume "${RESUME}")
else
    if [[ -e "${OUT}/latest.pt" || -e "${OUT}/run_contract.json" \
        || -e "${OUT}/train.jsonl" || -e "${OUT}/wandb_run_id.txt" ]]; then
        echo "[object-region-v44] refusing to overwrite ${OUT}" >&2
        exit 2
    fi
    if [[ -n "${INIT_FROM}" ]]; then
        [[ "${INIT_FROM}" == /* && -f "${INIT_FROM}" ]] || {
            echo "[object-region-v44] INIT_FROM is missing" >&2
            exit 2
        }
        init_args=(--init_from "$(readlink -f "${INIT_FROM}")")
    fi
fi
sidecar_args=()
[[ -n "${TEACHER_SIDECAR}" ]] && sidecar_args=(--teacher_sidecar "${TEACHER_SIDECAR}")
wandb_id_args=()
[[ -n "${REQUESTED_WANDB_RUN_ID}" ]] && wandb_id_args=(--wandb_run_id "${REQUESTED_WANDB_RUN_ID}")

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"

mkdir -p "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"
exec 9>"${OUT}/.v44_train.lock"
flock -n 9 || { echo "[object-region-v44] another process owns ${OUT}" >&2; exit 2; }
cd "${ROOT}"
cmd=(
    "${TORCHRUN}" --standalone --nnodes 1 --nproc_per_node "${TORCHRUN_NPROC}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}" --data_format sequence --feature_source jit
    --jit_dino_batch "${JIT_DINO_BATCH}"
    --temporal_contract dynamic_dual_horizon_video_v2
    --video_vae_model "${VIDEO_VAE_MODEL}"
    --video_vae_contract "${VIDEO_VAE_CONTRACT}"
    --video_vae_pythonpath "${VIDEO_VAE_PYTHONPATH}"
    --video_vae_short_side 256 --video_vae_clip_frames 5
    --video_vae_batch "${VIDEO_VAE_BATCH}"
    --history_frames 4 --history_frames_min 1 --history_frames_max 4
    --history_span_frames "${HISTORY_SPAN_FRAMES}"
    --future_frames 2 --short_horizon_frames "${SHORT_HORIZON_FRAMES}"
    --goal_query_seconds "${GOAL_QUERY_SECONDS}"
    --goal_tail_guard_frames "${GOAL_TAIL_GUARD_FRAMES}"
    --goal_probe_frames "${GOAL_PROBE_FRAMES}"
    --goal_stability_threshold "${GOAL_STABILITY_THRESHOLD}"
    --goal_rollout_weight "${GOAL_ROLLOUT_WEIGHT}"
    --path_consistency_weight "${PATH_CONSISTENCY_WEIGHT}"
    --out "${OUT}" --profile full --architecture object_region_dual_encoder_v1
    --training_stage representation --readout_scope off
    --representation_steps "${STEPS}" --joint_steps 0
    --batch "${BATCH_PER_GPU}" --grad_accum "${TRAINER_GRAD_ACCUM}"
    --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
    --max_train_items "${MAX_TRAIN_ITEMS}"
    --lr "${CORE_LR}" --lr_floor "${LR_FLOOR}"
    --core_lr "${CORE_LR}" --dino_lr "${DINO_LR}"
    --new_module_lr "${NEW_MODULE_LR}" --action_lr "${ACTION_LR}"
    --readout_lr "${CORE_LR}" --current_readout_weight 0
    --readout_regularization_weight 0 --carrier_support_weight 0
    --carrier_compact_weight 0 --gaussian_children 1
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
    "${sidecar_args[@]}" "${wandb_id_args[@]}" "${init_args[@]}"
)
echo "[object-region-v44] commit=$(git rev-parse HEAD) nproc=${TORCHRUN_NPROC}"
printf '[object-region-v44] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec > >(tee -a "${LOG_ROOT}/train.log") 2>&1
exec "${cmd[@]}"
