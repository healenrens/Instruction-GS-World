#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
cd "$ROOT"

export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/mnt/pfs/public/xuhaoming/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export RT2_CAUSAL_CPU_THREADS=2

DATA=data/rt2_causal_v1
CACHE=data/rt2_causal_v1_prepcache_wrist
OUT=checkpoints/vla_causal_v1
LOG_ROOT=logs/causal_v1
VERSION=vggt_t1_grid48_v1
NSHARD=8

mkdir -p "$DATA" "$CACHE" "$OUT" "$LOG_ROOT"
if compgen -G "$OUT/vla_*.pt" > /dev/null; then
    echo "[pipeline] refusing to overwrite an existing causal-v1 training run in $OUT"
    exit 1
fi

echo "[pipeline] phase=data start $(date -Is) nshard=$NSHARD (2 workers/GPU)"
pids=()
for shard in $(seq 0 $((NSHARD - 1))); do
    gpu=$((shard % 4))
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python code/scripts/rt2_build_causal_dataset.py \
        --source data/rt2_joint_src --actions data/rt2_act --flow data/rt2_joint \
        --plan data/rt2_win/window_plan.json --out "$DATA" \
        --shard "$shard" --nshard "$NSHARD" \
        > "$LOG_ROOT/data_shard${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" != 0 ]]; then
    echo "[pipeline] at least one data shard failed; inspect $LOG_ROOT/data_shard*.log"
    exit 1
fi
echo "[pipeline] phase=data done $(date -Is)"

.venv/bin/python code/scripts/rt2_verify_causal_dataset.py \
    --data "$DATA" --source data/rt2_joint_src --report "$LOG_ROOT/dataset_verification.json" \
    > "$LOG_ROOT/dataset_verification.log" 2>&1
echo "[pipeline] phase=verify done $(date -Is)"

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --nproc_per_node=4 --master_port=29614 \
    code/scripts/train_vla.py --cache_prep --data "$DATA" --prep_cache "$CACHE" \
    --norm_stats data/rt2_act/norm_stats.pt --geom_mode xyz --img_loss 1 --L 512 \
    --n_state_tokens 1 --placement entropy --wrist 1 \
    --causal_geometry_version "$VERSION" \
    > "$LOG_ROOT/cache.log" 2>&1
echo "[pipeline] phase=cache done $(date -Is)"

source_count=$(find "$DATA" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')
cache_count=$(find "$CACHE" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')
if [[ "$source_count" != "$cache_count" ]]; then
    echo "[pipeline] cache count mismatch: data=$source_count cache=$cache_count"
    exit 1
fi

echo "[pipeline] phase=train start $(date -Is) data=$source_count cache=$cache_count"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --nproc_per_node=4 --master_port=29615 \
    code/scripts/train_vla.py --data "$DATA" --prep_cache "$CACHE" \
    --norm_stats data/rt2_act/norm_stats.pt --geom_mode xyz --img_loss 1 --w_depth 0.5 \
    --L 512 --n_state_tokens 1 --placement entropy --wrist 1 \
    --causal_geometry_version "$VERSION" \
    --deepspeed 1 --steps 50000 --batch 24 --accum 3 --lr 5e-05 \
    --w_flow 1.0 --w_act 1.0 --weight_decay 0.01 --init_from "" --out "$OUT" \
    > "$LOG_ROOT/train.log" 2>&1
