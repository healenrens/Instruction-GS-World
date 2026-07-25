#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
SOURCE="${SOURCE:-}"
RESUME="${RESUME:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
STEPS="${STEPS:-100000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-0}"
HISTORY_FRAMES="${HISTORY_FRAMES:-4}"
FUTURE_FRAMES="${FUTURE_FRAMES:-4}"
SEQUENCE_ANCHORS="${SEQUENCE_ANCHORS:-3,5,8}"
LR="${LR:-2e-4}"
LR_FLOOR="${LR_FLOOR:-2e-5}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.05}"
EFFECT_WEIGHT="${EFFECT_WEIGHT:-0.5}"
GOAL_ANCHOR_WEIGHT="${GOAL_ANCHOR_WEIGHT:-0.5}"
GOAL_RANK_WEIGHT="${GOAL_RANK_WEIGHT:-0.5}"
GOAL_RELATIVE_MARGIN="${GOAL_RELATIVE_MARGIN:-0.05}"
ACTION_ACTIVITY_FLOOR="${ACTION_ACTIVITY_FLOOR:-0.25}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-20}"
RUN_NAME="${RUN_NAME:-visual_sequence_goal_prior_h${HISTORY_FRAMES}q${FUTURE_FRAMES}_4gpu_steps${STEPS}_seed${SEED}_${RUN_DATE}}"
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"

if [[ ! -d "${DATA}" || ! -f "${SOURCE}" ]] \
    || [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! sha256sum -c --status "${DATA}/episode_manifest.verified.sha256"; then
    echo "[goal-prior] missing data or base checkpoint" >&2
    exit 2
fi
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "[goal-prior] resume checkpoint not found: ${RESUME}" >&2
    exit 2
fi
if [[ -z "${RESUME}" && -e "${OUT}/latest.pt" ]]; then
    echo "[goal-prior] refusing to overwrite run: ${OUT}" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[goal-prior] expected exactly four GPU ids" >&2
    exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

cd "${ROOT}"
mkdir -p "${OUT}"
cmd=(
    .venv/bin/torchrun
    --standalone
    --nproc_per_node=4
    code/scripts/train_visual_sequence_goal_prior.py
    --checkpoint "${SOURCE}"
    --resume "${RESUME}"
    --data "${DATA}"
    --out "${OUT}"
    --steps "${STEPS}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${GRAD_ACCUM}"
    --workers "${WORKERS_PER_RANK}"
    --max_train_items "${MAX_TRAIN_ITEMS}"
    --history_frames "${HISTORY_FRAMES}"
    --future_frames "${FUTURE_FRAMES}"
    --sequence_anchors "${SEQUENCE_ANCHORS}"
    --lr "${LR}"
    --lr_floor "${LR_FLOOR}"
    --warmup_fraction "${WARMUP_FRACTION}"
    --weight_decay 1e-4
    --effect_weight "${EFFECT_WEIGHT}"
    --goal_anchor_weight "${GOAL_ANCHOR_WEIGHT}"
    --goal_rank_weight "${GOAL_RANK_WEIGHT}"
    --goal_relative_margin "${GOAL_RELATIVE_MARGIN}"
    --action_activity_floor "${ACTION_ACTIVITY_FLOOR}"
    --save_every "${SAVE_EVERY}"
    --log_every "${LOG_EVERY}"
    --seed "${SEED}"
    --amp bf16
)
echo "[goal-prior] effective_batch=$((BATCH_PER_GPU * 4 * GRAD_ACCUM))"
printf "[goal-prior] command:"
printf " %q" "${cmd[@]}"
printf "\n"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}" \
    2>&1 | tee "${OUT}/train.console.log"
