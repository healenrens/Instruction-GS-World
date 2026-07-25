#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
cd "$ROOT"

required_env=(
    WORLD_SIZE RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
    DATA DINO PAIR_AUDIT DINO_AUDIT
)
for name in "${required_env[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[launcher] missing required environment variable: $name" >&2
        exit 2
    fi
done

# Platform WORLD_SIZE/RANK describe nodes. Torchrun replaces them in child
# processes with global process WORLD_SIZE/RANK and injects LOCAL_RANK.
NNODES="$WORLD_SIZE"
NODE_RANK="$RANK"
EXPECTED_NNODES="${EXPECTED_NNODES:-2}"
EXPECTED_NPROC_PER_NODE="${EXPECTED_NPROC_PER_NODE:-8}"
if [[ "$NNODES" -ne "$EXPECTED_NNODES" ||
      "$NPROC_PER_NODE" -ne "$EXPECTED_NPROC_PER_NODE" ]]; then
    echo "[launcher] expected ${EXPECTED_NNODES}x${EXPECTED_NPROC_PER_NODE}, got ${NNODES}x${NPROC_PER_NODE}" >&2
    exit 2
fi
if [[ "$NODE_RANK" -lt 0 || "$NODE_RANK" -ge "$NNODES" ]]; then
    echo "[launcher] invalid node rank: $NODE_RANK for $NNODES nodes" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

RUN_NAME="${RUN_NAME:-correlated_action_field_2n8g_v1}"
OUT="${OUT:-$ROOT/checkpoints/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$RUN_NAME}"
RESUME="${RESUME:-}"

POSTERIOR_STEPS="${POSTERIOR_STEPS:-8000}"
PRIOR_STEPS="${PRIOR_STEPS:-4000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-4}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-25}"
SEED="${SEED:-17}"
AMP="${AMP:-bf16}"

HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
ACTION_DIM="${ACTION_DIM:-16}"
CONTROL_ROWS="${CONTROL_ROWS:-16}"
CONTROL_COLS="${CONTROL_COLS:-16}"
ACTIVE_COUNT="${ACTIVE_COUNT:-192}"
DINO_DIM="${DINO_DIM:-32}"
FLOW_STEPS="${FLOW_STEPS:-16}"
RENDER_HEIGHT="${RENDER_HEIGHT:-98}"
RENDER_WIDTH="${RENDER_WIDTH:-130}"

POSTERIOR_LR="${POSTERIOR_LR:-3e-4}"
PRIOR_LR="${PRIOR_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

mkdir -p "$OUT" "$LOG_ROOT"
if [[ -z "$RESUME" ]] &&
   { compgen -G "$OUT/posterior_*.pt" > /dev/null ||
     compgen -G "$OUT/prior_*.pt" > /dev/null; }; then
    echo "[launcher] refusing to overwrite checkpoints in $OUT" >&2
    exit 2
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
    echo "[launcher] resume checkpoint not found: $RESUME" >&2
    exit 2
fi

GLOBAL_PROCESSES=$((NNODES * NPROC_PER_NODE))
PREFLIGHT_REPORT="$LOG_ROOT/preflight_node_${NODE_RANK}.json"
.venv/bin/python code/scripts/preflight_correlated_action_training.py \
    --data "$DATA" \
    --dino "$DINO" \
    --pair_audit "$PAIR_AUDIT" \
    --dino_audit "$DINO_AUDIT" \
    --out "$OUT" \
    --world_size "$GLOBAL_PROCESSES" \
    --batch "$BATCH_PER_GPU" \
    --grad_accum "$GRAD_ACCUM" \
    --dino_dim "$DINO_DIM" \
    --resume "$RESUME" \
    --report "$PREFLIGHT_REPORT"

cmd=(
    .venv/bin/torchrun
    --nproc_per_node "$NPROC_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    code/scripts/train_correlated_action_field.py
    --data "$DATA"
    --dino "$DINO"
    --out "$OUT"
    --resume "$RESUME"
    --posterior_steps "$POSTERIOR_STEPS"
    --prior_steps "$PRIOR_STEPS"
    --batch "$BATCH_PER_GPU"
    --grad_accum "$GRAD_ACCUM"
    --workers "$WORKERS_PER_RANK"
    --hidden_dim "$HIDDEN_DIM"
    --layers "$LAYERS"
    --heads "$HEADS"
    --action_dim "$ACTION_DIM"
    --control_rows "$CONTROL_ROWS"
    --control_cols "$CONTROL_COLS"
    --active_count "$ACTIVE_COUNT"
    --dino_dim "$DINO_DIM"
    --flow_steps "$FLOW_STEPS"
    --render_height "$RENDER_HEIGHT"
    --render_width "$RENDER_WIDTH"
    --posterior_lr "$POSTERIOR_LR"
    --prior_lr "$PRIOR_LR"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_steps "$WARMUP_STEPS"
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"
    --w_move 30
    --w_deterministic 0.25
    --w_appearance 0.1
    --w_visibility 0.1
    --w_rgb 0.2
    --w_dino 0.05
    --w_effect 0.5
    --w_action_alignment 1
    --w_usage 1
    --usage_margin 0.01
    --w_local 0.2
    --w_alpha 0.01
    --w_slot 0
    --seed "$SEED"
    --amp "$AMP"
)

EFFECTIVE_BATCH=$((BATCH_PER_GPU * GLOBAL_PROCESSES * GRAD_ACCUM))
echo "[launcher] node=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
echo "[launcher] global_processes=$GLOBAL_PROCESSES effective_batch=$EFFECTIVE_BATCH"
echo "[launcher] output=$OUT node_log=$LOG_ROOT/node_${NODE_RANK}.log"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf "[launcher] command:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    exit 0
fi

VISIBLE_GPUS=$(.venv/bin/python -c "import torch; print(torch.cuda.device_count())")
if [[ "$VISIBLE_GPUS" -lt "$NPROC_PER_NODE" ]]; then
    echo "[launcher] only $VISIBLE_GPUS GPUs visible, need $NPROC_PER_NODE" >&2
    exit 2
fi

if [[ "${PREWARM:-1}" == 1 ]]; then
    PREWARM_LOG="$LOG_ROOT/prewarm_node_${NODE_RANK}.log"
    CUDA_VISIBLE_DEVICES="${PREWARM_GPU:-0}" \
        flock -x /tmp/instruct_gs_world_gsplat_cuda.lock \
        .venv/bin/python code/scripts/test_correlated_action_field.py \
        --data "$DATA" \
        --dino "$DINO" \
        --render_height "$RENDER_HEIGHT" \
        --render_width "$RENDER_WIDTH" \
        > "$PREWARM_LOG" 2>&1
fi

exec > >(tee -a "$LOG_ROOT/node_${NODE_RANK}.log") 2>&1
exec "${cmd[@]}"
