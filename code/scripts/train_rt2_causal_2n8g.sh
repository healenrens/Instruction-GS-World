#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
cd "$ROOT"

required_env=(WORLD_SIZE RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE)
for name in "${required_env[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[launcher] missing platform environment variable: $name" >&2
        exit 2
    fi
done

# The platform variables describe nodes. Torchrun replaces RANK/WORLD_SIZE in
# child processes with global process rank/world size and injects LOCAL_RANK.
NNODES="$WORLD_SIZE"
NODE_RANK="$RANK"
if [[ "$NNODES" -ne 2 || "$NPROC_PER_NODE" -ne 8 ]]; then
    echo "[launcher] expected 2 nodes x 8 GPUs, got nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE" >&2
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
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

DATA="${DATA:-$ROOT/data/rt2_causal_v1}"
PREP_CACHE="${PREP_CACHE:-$ROOT/data/rt2_causal_v1_prepcache_wrist}"
NORM_STATS="${NORM_STATS:-$ROOT/data/rt2_act/norm_stats.pt}"
RUN_NAME="${RUN_NAME:-vla_causal_v1_2n8g}"
OUT="${OUT:-$ROOT/checkpoints/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$RUN_NAME}"
INIT_FROM="${INIT_FROM:-}"

STEPS="${STEPS:-50000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-18}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SEED="${SEED:-42}"

for path in "$DATA" "$PREP_CACHE"; do
    if [[ ! -d "$path" ]]; then
        echo "[launcher] required directory not found: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$NORM_STATS" ]]; then
    echo "[launcher] normalization stats not found: $NORM_STATS" >&2
    exit 2
fi
if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
    echo "[launcher] warm-start checkpoint not found: $INIT_FROM" >&2
    exit 2
fi

mkdir -p "$OUT" "$LOG_ROOT"
if compgen -G "$OUT/vla_*.pt" > /dev/null; then
    echo "[launcher] refusing to overwrite checkpoints in $OUT" >&2
    exit 2
fi

GLOBAL_PROCESSES=$((NNODES * NPROC_PER_NODE))
EFFECTIVE_BATCH=$((BATCH_PER_GPU * GLOBAL_PROCESSES * GRAD_ACCUM))
echo "[launcher] node=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
echo "[launcher] global_processes=$GLOBAL_PROCESSES effective_batch=$EFFECTIVE_BATCH"
echo "[launcher] output=$OUT node_log=$LOG_ROOT/node_${NODE_RANK}.log"

cmd=(
    .venv/bin/torchrun
    --nproc_per_node "$NPROC_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    code/scripts/train_vla.py
    --data "$DATA"
    --prep_cache "$PREP_CACHE"
    --norm_stats "$NORM_STATS"
    --geom_mode xyz
    --img_loss 1
    --w_depth 0.5
    --L 512
    --n_state_tokens 1
    --placement entropy
    --wrist 1
    --causal_geometry_version vggt_t1_grid48_v1
    --deepspeed 1
    --steps "$STEPS"
    --batch "$BATCH_PER_GPU"
    --accum "$GRAD_ACCUM"
    --prefetch_workers "$PREFETCH_WORKERS"
    --lr_peak 5e-5
    --lr_floor 1e-5
    --warmup_steps 1500
    --w_flow 1.0
    --w_act 1.0
    --weight_decay 0.01
    --save_every "$SAVE_EVERY"
    --seed "$SEED"
    --init_from "$INIT_FROM"
    --out "$OUT"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf "[launcher] command:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    exit 0
fi

exec > >(tee -a "$LOG_ROOT/node_${NODE_RANK}.log") 2>&1
exec "${cmd[@]}"
