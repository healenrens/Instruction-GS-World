#!/bin/bash
# Direction A: token VISUAL feature source A/B — Qwen VLM patches (baseline) vs frozen DINOv2 dense.
# Same config (xyz L1024 no-mag, 1800 steps; model converges by ~1500). GPU0=qwen GPU1=dino. Does
# stronger dense spatial features lift the moderate ~0.6 dcos? Run: setsid bash code/scripts/orchestrate_gpswm_feat.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [gpswm-feat] START $(date) ===================="
COMMON="--data data/mix_v15 --geom_mode xyz --L 1024 --steps 1800 --w_mag 0 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 1800"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON --feat_source qwen --out checkpoints/gpswm_fqwen > logs/gpswm_fqwen.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON --feat_source dino --out checkpoints/gpswm_fdino > logs/gpswm_fdino.log 2>&1 &
echo "[gpswm-feat] qwen=GPU0 dino=GPU1"
wait
echo "==================== [gpswm-feat] DONE $(date) ===================="
