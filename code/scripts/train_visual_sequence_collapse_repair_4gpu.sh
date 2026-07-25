#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
SOURCE="${SOURCE:-${ROOT}/outputs/visual_sequence_core_h4q4_nolang_core_1n8g_v26_steps100000_seed17_20260720/joint_0004000.pt}"
RESUME="${RESUME:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260723}"
JOINT_STEPS="${JOINT_STEPS:-12000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
LR="${LR:-5e-5}"
LR_FLOOR="${LR_FLOOR:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
RGB_LOSS_WEIGHT="${RGB_LOSS_WEIGHT:-0.5}"
RGB_CHANGE_LOSS_WEIGHT="${RGB_CHANGE_LOSS_WEIGHT:-1.0}"
RGB_CHANGE_THRESHOLD="${RGB_CHANGE_THRESHOLD:-0.04}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed${SEED}_${RUN_DATE}}"
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-visual-sequence-collapse-repair}"
WANDB_TAGS="${WANDB_TAGS:-rt2,no-language,posterior-gate,change-balanced,v27,ddp-4gpu}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"

if [[ "${DATA}" != /* || "${SOURCE}" != /* || "${OUT}" != /* ]]; then
    echo "[collapse-repair] data, source, and output paths must be absolute" >&2
    exit 2
fi
if [[ ! -d "${DATA}" ]] \
    || [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! sha256sum -c --status "${DATA}/episode_manifest.verified.sha256"; then
    echo "[collapse-repair] verified sequence data is missing: ${DATA}" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[collapse-repair] expected exactly four GPU ids" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" != "online" \
      && "${WANDB_MODE}" != "offline" \
      && "${WANDB_MODE}" != "disabled" ]]; then
    echo "[collapse-repair] WANDB_MODE must be online, offline, or disabled" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" != "disabled" ]] \
    && ! "${ROOT}/.venv/bin/python" -c "import wandb" >/dev/null 2>&1; then
    echo "[collapse-repair] wandb is not installed in ${ROOT}/.venv" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" == "online" ]] \
    && [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[collapse-repair] online W&B requires WANDB_API_KEY or ~/.netrc" >&2
    exit 2
fi

if [[ -n "${RESUME}" ]]; then
    if [[ ! -f "${RESUME}" ]]; then
        echo "[collapse-repair] resume checkpoint is missing: ${RESUME}" >&2
        exit 2
    fi
    init_args=(--resume "${RESUME}")
else
    if [[ ! -f "${SOURCE}" ]]; then
        echo "[collapse-repair] source checkpoint is missing: ${SOURCE}" >&2
        exit 2
    fi
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[collapse-repair] refusing to overwrite run: ${OUT}" >&2
        exit 2
    fi
    init_args=(--init_from "${SOURCE}")
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

cd "${ROOT}"
cmd=(
    .venv/bin/torchrun
    --standalone
    --nproc_per_node=4
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}"
    --data_format sequence
    --history_frames 4
    --future_frames 4
    --sequence_anchors 3,5,8
    --out "${OUT}"
    --profile full
    --representation_steps 0
    --joint_steps "${JOINT_STEPS}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${GRAD_ACCUM}"
    --workers "${WORKERS_PER_RANK}"
    --lr "${LR}"
    --lr_floor "${LR_FLOOR}"
    --warmup_steps "${WARMUP_STEPS}"
    --warmup_fraction 0.0
    --weight_decay 1e-4
    --save_every "${SAVE_EVERY}"
    --log_every "${LOG_EVERY}"
    --seed "${SEED}"
    --amp bf16
    --language_condition off
    --rgb_supervision on
    --rgb_short_side 256
    --rgb_pad_multiple 16
    --rgb_render_chunk 8192
    --rgb_loss_weight "${RGB_LOSS_WEIGHT}"
    --rgb_ssim_weight 0.2
    --rgb_change_loss_weight "${RGB_CHANGE_LOSS_WEIGHT}"
    --rgb_change_threshold "${RGB_CHANGE_THRESHOLD}"
    --language_effect_weight 0.0
    --zero_action_margin_weight 5.0
    --posterior_dynamics_gate
    --posterior_update_scope full
    --action_anchor object_slot
    --canonical_center_gate 1.0
    --canonical_activity_gate
    --canonical_activity_power 1.0
    --action_residual_dim 8
    --action_residual_gate 1.0
    --action_residual_dropout 0.0
    --semantic_action_basis rgb
    --wandb_mode "${WANDB_MODE}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_name "${WANDB_NAME}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_tags "${WANDB_TAGS}"
    --wandb_run_id "${WANDB_RUN_ID}"
    --wandb_dir "${WANDB_DIR}"
)
cmd+=("${init_args[@]}")
printf "[collapse-repair] command:"
printf " %q" "${cmd[@]}"
printf "\n"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
mkdir -p "${OUT}" "${WANDB_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}" \
    2>&1 | tee -a "${OUT}/train.console.log"
