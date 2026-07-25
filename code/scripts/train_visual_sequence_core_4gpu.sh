#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
SOURCE="${SOURCE:-${ROOT}/outputs/canonical6_rgb_centergate0p1_activitypower1_v24_converted_seed17_20260720/joint_0000000.pt}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
RUN_DATE="${RUN_DATE:-20260720}"
JOINT_STEPS="${JOINT_STEPS:-100000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-0}"
TRAIN_MODE="${TRAIN_MODE:-core}"
POSTERIOR_UPDATE_SCOPE="${POSTERIOR_UPDATE_SCOPE:-full}"
HISTORY_FRAMES="${HISTORY_FRAMES:-4}"
FUTURE_FRAMES="${FUTURE_FRAMES:-4}"
SEQUENCE_ANCHORS="${SEQUENCE_ANCHORS:-3,5,8}"
LR="${LR:-2e-4}"
LR_FLOOR="${LR_FLOOR:-2e-5}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.05}"
RGB_LOSS_WEIGHT="${RGB_LOSS_WEIGHT:-0.5}"
ACTION_RESIDUAL_DIM="${ACTION_RESIDUAL_DIM:-8}"
ACTION_RESIDUAL_GATE="${ACTION_RESIDUAL_GATE:-1.0}"
ACTION_RESIDUAL_DROPOUT="${ACTION_RESIDUAL_DROPOUT:-0.0}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
OUT="${OUT:-${ROOT}/outputs/visual_sequence_core_h${HISTORY_FRAMES}q${FUTURE_FRAMES}_nolang_${TRAIN_MODE}_v26_steps${JOINT_STEPS}_seed${SEED}_${RUN_DATE}}"

if [[ ! -d "${DATA}" || ! -f "${SOURCE}" ]] \
    || [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! sha256sum -c --status "${DATA}/episode_manifest.verified.sha256"; then
    echo "[visual-sequence] missing data or warm-start checkpoint" >&2
    exit 2
fi
if [[ -e "${OUT}/latest.pt" ]]; then
    echo "[visual-sequence] refusing to overwrite run: ${OUT}" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[visual-sequence] expected exactly four GPU ids" >&2
    exit 2
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

cd "${ROOT}"
cmd=(
    .venv/bin/torchrun
    --standalone
    --nproc_per_node=4
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}"
    --data_format sequence
    --history_frames "${HISTORY_FRAMES}"
    --future_frames "${FUTURE_FRAMES}"
    --sequence_anchors "${SEQUENCE_ANCHORS}"
    --out "${OUT}"
    --init_from "${SOURCE}"
    --profile full
    --representation_steps 0
    --joint_steps "${JOINT_STEPS}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${GRAD_ACCUM}"
    --workers "${WORKERS_PER_RANK}"
    --max_train_items "${MAX_TRAIN_ITEMS}"
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
    --canonical_center_gate 1.0
    --canonical_activity_gate
    --canonical_activity_power 1.0
    --action_residual_dim "${ACTION_RESIDUAL_DIM}"
    --action_residual_gate "${ACTION_RESIDUAL_GATE}"
    --action_residual_dropout "${ACTION_RESIDUAL_DROPOUT}"
    --semantic_action_basis rgb
)
cmd+=("${mode_args[@]}")
printf "[visual-sequence] command:"
printf " %q" "${cmd[@]}"
printf "\n"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}" \
    2>&1 | tee "${OUT}/train.console.log"
