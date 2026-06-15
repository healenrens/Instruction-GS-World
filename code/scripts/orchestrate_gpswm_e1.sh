#!/bin/bash
# PLAN E1 (4-GPU DDP): geometry prediction space A/B — xyz vs flowd. Each arm DDP across 2 GPUs:
# xyz=GPU0,1  flowd=GPU2,3 (all 4 cards). steps=1250 @ 2 clips/step ≈ 2500 single-GPU clip-passes.
# Run: setsid bash code/scripts/orchestrate_gpswm_e1.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [gpswm-e1] START $(date) ===================="
COMMON="--data data/mix_v15 --L 512 --steps 1250 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 625"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --master_port=29551 \
  code/scripts/train_gpstoken_wm.py $COMMON --out checkpoints/gpswm_xyz --geom_mode xyz \
  > logs/gpswm_xyz.log 2>&1 &
PID_XYZ=$!
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun --nproc_per_node=2 --master_port=29552 \
  code/scripts/train_gpstoken_wm.py $COMMON --out checkpoints/gpswm_flowd --geom_mode flowd \
  > logs/gpswm_flowd.log 2>&1 &
PID_FLOWD=$!
echo "[gpswm-e1] xyz pid=$PID_XYZ (GPU0,1)  flowd pid=$PID_FLOWD (GPU2,3)"
wait $PID_XYZ; echo "[gpswm-e1] xyz exit=$? $(date)"
wait $PID_FLOWD; echo "[gpswm-e1] flowd exit=$? $(date)"
echo "==================== [gpswm-e1] DONE $(date) ===================="
