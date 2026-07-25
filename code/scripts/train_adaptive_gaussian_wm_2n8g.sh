#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
cd "$ROOT"

required_env=(
    WORLD_SIZE
    RANK
    MASTER_ADDR
    MASTER_PORT
    NPROC_PER_NODE
    DATA
    DINO
    CONDITION_CACHE
)
for name in "${required_env[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[launcher] missing required environment variable: $name" >&2
        exit 2
    fi
done

NNODES="$WORLD_SIZE"
NODE_RANK="$RANK"
if [[ "$NNODES" -ne 2 || "$NPROC_PER_NODE" -ne 8 ]]; then
    echo "[launcher] expected 2 nodes x 8 GPUs, got ${NNODES}x${NPROC_PER_NODE}" >&2
    exit 2
fi
if [[ "$NODE_RANK" -lt 0 || "$NODE_RANK" -ge "$NNODES" ]]; then
    echo "[launcher] invalid node rank: $NODE_RANK" >&2
    exit 2
fi
for path in "$DATA" "$DINO"; do
    if [[ ! -d "$path" ]]; then
        echo "[launcher] required directory not found: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$CONDITION_CACHE" ]]; then
    echo "[launcher] condition cache not found: $CONDITION_CACHE" >&2
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
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

RUN_NAME="${RUN_NAME:-adaptive_gaussian_object_jepa_rgb_lang_2n8g_v3}"
OUT="${OUT:-$ROOT/checkpoints/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$RUN_NAME}"
RESUME="${RESUME:-}"
INIT_FROM="${INIT_FROM:-}"
GATE_REPORT="${GATE_REPORT:-$ROOT/outputs/adaptive_gaussian_wm_feasibility/gate_summary.json}"
PROFILE="${PROFILE:-full}"
REPRESENTATION_STEPS="${REPRESENTATION_STEPS:-5000}"
JOINT_STEPS="${JOINT_STEPS:-40000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
SEED="${SEED:-17}"
LR="${LR:-2e-4}"
LR_FLOOR="${LR_FLOOR:-2e-5}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.05}"
LANGUAGE_CONDITION="${LANGUAGE_CONDITION:-on}"
RGB_SUPERVISION="${RGB_SUPERVISION:-on}"
RGB_SHORT_SIDE="${RGB_SHORT_SIDE:-256}"
RGB_LOSS_WEIGHT="${RGB_LOSS_WEIGHT:-0.5}"
LANGUAGE_EFFECT_WEIGHT="${LANGUAGE_EFFECT_WEIGHT:-0.0}"

if [[ ! -f "$GATE_REPORT" ]]; then
    echo "[launcher] gate report not found: $GATE_REPORT" >&2
    exit 2
fi
gate_ready=$(.venv/bin/python -c \
    'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["large_scale_ready"])))' \
    "$GATE_REPORT")
if [[ "$gate_ready" != 1 && "${ALLOW_FAILED_GATES:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
    echo "[launcher] large-scale gates have not passed; refusing to train" >&2
    exit 2
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
    echo "[launcher] resume checkpoint not found: $RESUME" >&2
    exit 2
fi
if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
    echo "[launcher] init checkpoint not found: $INIT_FROM" >&2
    exit 2
fi
if [[ -n "$RESUME" && -n "$INIT_FROM" ]]; then
    echo "[launcher] RESUME and INIT_FROM are mutually exclusive" >&2
    exit 2
fi
if [[ -z "$RESUME" && -e "$OUT/latest.pt" ]]; then
    echo "[launcher] refusing to overwrite existing run: $OUT" >&2
    exit 2
fi

mkdir -p "$OUT" "$LOG_ROOT"
cmd=(
    .venv/bin/torchrun
    --nproc_per_node "$NPROC_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    code/scripts/train_adaptive_gaussian_wm.py
    --data "$DATA"
    --dino "$DINO"
    --condition_cache "$CONDITION_CACHE"
    --out "$OUT"
    --resume "$RESUME"
    --init_from "$INIT_FROM"
    --profile "$PROFILE"
    --representation_steps "$REPRESENTATION_STEPS"
    --joint_steps "$JOINT_STEPS"
    --batch "$BATCH_PER_GPU"
    --grad_accum "$GRAD_ACCUM"
    --workers "$WORKERS_PER_RANK"
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"
    --seed "$SEED"
    --lr "$LR"
    --lr_floor "$LR_FLOOR"
    --warmup_fraction "$WARMUP_FRACTION"
    --language_condition "$LANGUAGE_CONDITION"
    --rgb_supervision "$RGB_SUPERVISION"
    --rgb_short_side "$RGB_SHORT_SIDE"
    --rgb_loss_weight "$RGB_LOSS_WEIGHT"
    --language_effect_weight "$LANGUAGE_EFFECT_WEIGHT"
    --amp bf16
)

GLOBAL_PROCESSES=$((NNODES * NPROC_PER_NODE))
EFFECTIVE_BATCH=$((BATCH_PER_GPU * GLOBAL_PROCESSES * GRAD_ACCUM))
echo "[launcher] DDP node=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
echo "[launcher] processes=$GLOBAL_PROCESSES effective_batch=$EFFECTIVE_BATCH"
echo "[launcher] language=$LANGUAGE_CONDITION rgb=$RGB_SUPERVISION"
echo "[launcher] output=$OUT checkpoint=v12_rank0_full_state_dict"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf "[launcher] command:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    exit 0
fi

visible_gpus=$(.venv/bin/python -c "import torch; print(torch.cuda.device_count())")
if [[ "$visible_gpus" -lt "$NPROC_PER_NODE" ]]; then
    echo "[launcher] only $visible_gpus GPUs visible, need $NPROC_PER_NODE" >&2
    exit 2
fi

exec > >(tee -a "$LOG_ROOT/node_${NODE_RANK}.log") 2>&1
exec "${cmd[@]}"
