#!/bin/bash
# CONSOLIDATION 2x2 (token-count × magnitude), all SINGLE-GPU (no DDP confound), STABILIZED mover_magnitude.
# c512/c1024 = no-mag ; c512m/c1024m = stable-mag 0.1. 2500 steps. Answers E2 (L512 vs L1024) AND the
# magnitude question cleanly on the known-good single-GPU setting. Run: setsid bash code/scripts/orchestrate_gpswm_consol.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [consol] START $(date) ===================="
COMMON="--data data/mix_v15 --geom_mode xyz --steps 2500 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 2500"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON --L 512  --w_mag 0.0 --out checkpoints/gpswm_c512   > logs/gpswm_c512.log   2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON --L 512  --w_mag 0.1 --out checkpoints/gpswm_c512m  > logs/gpswm_c512m.log  2>&1 &
CUDA_VISIBLE_DEVICES=2 .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON --L 1024 --w_mag 0.0 --out checkpoints/gpswm_c1024  > logs/gpswm_c1024.log  2>&1 &
CUDA_VISIBLE_DEVICES=3 .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON --L 1024 --w_mag 0.1 --out checkpoints/gpswm_c1024m > logs/gpswm_c1024m.log 2>&1 &
echo "[consol] launched 4 single-GPU: c512 c512m c1024 c1024m"
wait
echo "==================== [consol] DONE $(date) ===================="
