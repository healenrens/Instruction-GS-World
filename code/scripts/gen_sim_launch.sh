#!/usr/bin/env bash
# Parallel sim-clip generation across 4 GPUs x WORKERS_PER_GPU workers.
# ManiSkill is CPU-sim + GPU-render -> several workers per GPU overlap CPU sim with GPU render.
# Each worker gets a disjoint shard of the (env,seed) job list. setsid + nohup so it survives logout.
#
#   bash code/scripts/gen_sim_launch.sh <WS> <SEEDS> <WORKERS_PER_GPU> <LOGDIR>
set -u
WS="${1:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SEEDS="${2:-200}"
WPG="${3:-4}"
LOGDIR="${4:-$WS/logs/gensim}"
TASKS="PickCube-v1,PushCube-v1,StackCube-v1"
HELD_TASK="StackCube-v1"
NGPU=4
NSHARDS=$(( NGPU * WPG ))
mkdir -p "$LOGDIR"
cd "$WS" || exit 1
echo "[launch] nshards=$NSHARDS seeds/task=$SEEDS tasks=$TASKS held_task=$HELD_TASK logdir=$LOGDIR"
for ((s=0; s<NSHARDS; s++)); do
  gpu=$(( s % NGPU ))
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup ./.venv/bin/python code/scripts/gen_sim_dataset.py \
      --out "$WS/data/maniskill" --shard "$s" --nshards "$NSHARDS" \
      --tasks "$TASKS" --seeds "$SEEDS" --held_task "$HELD_TASK" \
      --held_seed_frac 0.15 --K 16 --cam 512 --min_val_psnr 16 \
      > "$LOGDIR/gen_shard${s}.log" 2>&1 < /dev/null &
  echo "  shard $s -> GPU $gpu (log gen_shard${s}.log)"
  sleep 0.3
done
echo "[launch] all $NSHARDS workers started"
