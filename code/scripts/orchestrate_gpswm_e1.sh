#!/bin/bash
# PLAN E1: geometry prediction space A/B — xyz (direct 3D displacement) vs flowd (2D flow + depth change),
# parallel on GPU0/GPU1. Same data (mix_v15 dual-track sim+real), same everything else. L=512 (user: 256
# may be too sparse). Full v1 objective (geom + JEPA + SIGReg + grounding).
# Run: setsid bash code/scripts/orchestrate_gpswm_e1.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [gpswm-e1] START $(date) ===================="

CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/train_gpstoken_wm.py \
  --data data/mix_v15 --out checkpoints/gpswm_xyz --geom_mode xyz --L 512 \
  --steps 2500 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 1250 \
  > logs/gpswm_xyz.log 2>&1 &
PID_XYZ=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python code/scripts/train_gpstoken_wm.py \
  --data data/mix_v15 --out checkpoints/gpswm_flowd --geom_mode flowd --L 512 \
  --steps 2500 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 1250 \
  > logs/gpswm_flowd.log 2>&1 &
PID_FLOWD=$!
echo "[gpswm-e1] xyz pid=$PID_XYZ (GPU0)  flowd pid=$PID_FLOWD (GPU1)"
wait $PID_XYZ; echo "[gpswm-e1] xyz exit=$? $(date)"
wait $PID_FLOWD; echo "[gpswm-e1] flowd exit=$? $(date)"
echo "==================== [gpswm-e1] DONE $(date) ===================="
