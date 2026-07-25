#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
required=(WORLD_SIZE RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[visual-sequence] missing environment variable: ${name}" >&2
        exit 2
    fi
done
if [[ "${WORLD_SIZE}" -ne 2 || "${NPROC_PER_NODE}" -ne 8 ]]; then
    echo "[visual-sequence] expected two nodes with eight GPUs each" >&2
    exit 2
fi
if [[ "${RANK}" -lt 0 || "${RANK}" -ge "${WORLD_SIZE}" ]]; then
    echo "[visual-sequence] invalid node rank: ${RANK}" >&2
    exit 2
fi

DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
SOURCE="${SOURCE:-${ROOT}/outputs/canonical6_rgb_centergate0p1_activitypower1_v24_converted_seed17_20260720/joint_0000000.pt}"
RESUME="${RESUME:-}"
TRAIN_MODE="${TRAIN_MODE:-core}"
POSTERIOR_UPDATE_SCOPE="${POSTERIOR_UPDATE_SCOPE:-full}"
HISTORY_FRAMES="${HISTORY_FRAMES:-4}"
FUTURE_FRAMES="${FUTURE_FRAMES:-4}"
SEQUENCE_ANCHORS="${SEQUENCE_ANCHORS:-3,5,8}"
JOINT_STEPS="${JOINT_STEPS:-100000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
LR="${LR:-2e-4}"
LR_FLOOR="${LR_FLOOR:-2e-5}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.05}"
RGB_LOSS_WEIGHT="${RGB_LOSS_WEIGHT:-0.5}"
CANONICAL_CENTER_GATE="${CANONICAL_CENTER_GATE:-1.0}"
ACTION_RESIDUAL_DIM="${ACTION_RESIDUAL_DIM:-8}"
ACTION_RESIDUAL_GATE="${ACTION_RESIDUAL_GATE:-1.0}"
ACTION_RESIDUAL_DROPOUT="${ACTION_RESIDUAL_DROPOUT:-0.0}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
RUN_NAME="${RUN_NAME:-visual_sequence_core_h${HISTORY_FRAMES}q${FUTURE_FRAMES}_nolang_${TRAIN_MODE}_2n8g_v26_seed${SEED}_${RUN_DATE}}"
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-visual-sequence-core-2n8g}"
WANDB_TAGS="${WANDB_TAGS:-rt2,core,no-language,ddp-2n8g}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"

if [[ ! -d "${DATA}" ]]; then
    echo "[visual-sequence] data not found: ${DATA}" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" != "online" \
      && "${WANDB_MODE}" != "offline" \
      && "${WANDB_MODE}" != "disabled" ]]; then
    echo "[visual-sequence] WANDB_MODE must be online, offline, or disabled" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" != "disabled" ]]; then
    if ! "${ROOT}/.venv/bin/python" -c "import wandb" >/dev/null 2>&1; then
        echo "[visual-sequence] wandb is not installed in ${ROOT}/.venv" >&2
        exit 2
    fi
    if [[ "${WANDB_MODE}" == "online" ]] \
        && [[ -z "${WANDB_API_KEY:-}" ]] \
        && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
        echo "[visual-sequence] online W&B requires WANDB_API_KEY or ~/.netrc" >&2
        exit 2
    fi
fi
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! sha256sum -c --status "${DATA}/episode_manifest.verified.sha256"; then
    echo "[visual-sequence] verified dense episode manifest not found: ${DATA}" >&2
    exit 2
fi
if [[ -n "${RESUME}" ]]; then
    if [[ ! -f "${RESUME}" ]]; then
        echo "[visual-sequence] resume checkpoint not found: ${RESUME}" >&2
        exit 2
    fi
    init_args=(--resume "${RESUME}")
else
    if [[ ! -f "${SOURCE}" ]]; then
        echo "[visual-sequence] warm-start checkpoint not found: ${SOURCE}" >&2
        exit 2
    fi
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[visual-sequence] refusing to overwrite run: ${OUT}" >&2
        exit 2
    fi
    init_args=(--init_from "${SOURCE}")
fi
mode_args=()
if [[ "${TRAIN_MODE}" == "posterior" ]]; then
    mode_args+=(
        --posterior_dynamics_gate
        --posterior_update_scope "${POSTERIOR_UPDATE_SCOPE}"
    )
elif [[ "${TRAIN_MODE}" == "core" ]]; then
    mode_args+=(--posterior_core_training)
    if [[ "${POSTERIOR_UPDATE_SCOPE}" != "full" ]]; then
        echo "[visual-sequence] core mode requires full update scope" >&2
        exit 2
    fi
elif [[ "${TRAIN_MODE}" != "joint" ]]; then
    echo "[visual-sequence] TRAIN_MODE must be core, posterior, or joint" >&2
    exit 2
elif [[ "${POSTERIOR_UPDATE_SCOPE}" != "full" ]]; then
    echo "[visual-sequence] restricted update scope requires posterior mode" >&2
    exit 2
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

cd "${ROOT}"
mkdir -p "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"
cmd=(
    .venv/bin/torchrun
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${WORLD_SIZE}"
    --node_rank "${RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}"
    --data_format sequence
    --history_frames "${HISTORY_FRAMES}"
    --future_frames "${FUTURE_FRAMES}"
    --sequence_anchors "${SEQUENCE_ANCHORS}"
    --out "${OUT}"
    --profile full
    --representation_steps 0
    --joint_steps "${JOINT_STEPS}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${GRAD_ACCUM}"
    --workers "${WORKERS_PER_RANK}"
    --lr "${LR}"
    --lr_floor "${LR_FLOOR}"
    --warmup_fraction "${WARMUP_FRACTION}"
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
    --language_effect_weight 0.0
    --zero_action_margin_weight 5.0
    --action_anchor object_slot
    --canonical_center_gate "${CANONICAL_CENTER_GATE}"
    --canonical_activity_gate
    --canonical_activity_power 1.0
    --action_residual_dim "${ACTION_RESIDUAL_DIM}"
    --action_residual_gate "${ACTION_RESIDUAL_GATE}"
    --action_residual_dropout "${ACTION_RESIDUAL_DROPOUT}"
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
cmd+=("${init_args[@]}" "${mode_args[@]}")
global_batch=$((BATCH_PER_GPU * WORLD_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
echo "[visual-sequence] node=${RANK}/${WORLD_SIZE} global_batch=${global_batch}"
echo "[visual-sequence] mode=${TRAIN_MODE} H=${HISTORY_FRAMES} Q=${FUTURE_FRAMES}"
printf "[visual-sequence] command:"
printf " %q" "${cmd[@]}"
printf "\n"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
exec > >(tee -a "${LOG_ROOT}/node_${RANK}.log") 2>&1
exec "${cmd[@]}"
