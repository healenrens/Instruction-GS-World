#!/usr/bin/env bash
# Launch 4-GPU DDP sim generalization training (agent.md §39).
# Resumes dynamics from stream11c (strict=False), spatial_ground on, direct 3D traj loss + InfoNCE.
#   bash code/scripts/train_sim_launch.sh <WS> <OUT> <LOG>
set -u
WS="${1:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
OUT="${2:-$WS/checkpoints/sim_gen}"
LOG="${3:-$WS/logs/train_sim.log}"
cd "$WS" || exit 1
echo "[launch] sim DDP training -> OUT=$OUT LOG=$LOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup ./.venv/bin/torchrun --nproc_per_node=4 --master-port 29531 \
    code/scripts/train_sim.py \
    --data "$WS/data/maniskill" --out "$OUT" \
    --resume "$WS/checkpoints/stream11c_infonce/ckpt_0006000.pt" \
    --spatial_ground 1 --vlm_image 1 \
    --M 2048 --K 16 --dim 1536 --layers 28 --heads 16 --n_query 16 \
    --lr 3e-4 --lr_sg 1e-3 --warmup 300 --total_steps 60000 --epochs 400 \
    --w_traj_pos 1.0 --w_traj_vel 1.0 --w_traj_rot 0.2 \
    --w_lang_contrast 0.5 --lang_tau 0.07 --queue_size 256 \
    --w_render 0.1 --w_scale_anchor 0.2 --obj_focus 0 \
    --checkpoint_every 1 --workers 4 --prefetch 2 \
    --log_every 20 --ckpt_every 500 \
    > "$LOG" 2>&1 < /dev/null &
echo "[launch] started (pid group $!)"
