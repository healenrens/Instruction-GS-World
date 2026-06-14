#!/bin/bash
# Clean magnitude test on the E1+E2 winner config (xyz, L1024), PROPERLY trained (2000 DDP steps):
# w_mag=0 (baseline) GPU0,1  vs  w_mag=0.1 (gentle) GPU2,3. Isolates whether a light magnitude nudge
# fixes mag-ratio 0.32 WITHOUT the dcos collapse that w_mag=0.5 caused.
# Run: setsid bash code/scripts/orchestrate_gpswm_mag.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [gpswm-mag] START $(date) ===================="
COMMON="--data data/mix_v15 --geom_mode xyz --L 1024 --steps 2000 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 1000"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --master_port=29571 \
  code/scripts/train_gpstoken_wm.py $COMMON --w_mag 0.0 --out checkpoints/gpswm_m0 \
  > logs/gpswm_m0.log 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun --nproc_per_node=2 --master_port=29572 \
  code/scripts/train_gpstoken_wm.py $COMMON --w_mag 0.1 --out checkpoints/gpswm_m01 \
  > logs/gpswm_m01.log 2>&1 &
PID_B=$!
echo "[gpswm-mag] w_mag0 pid=$PID_A (GPU0,1)  w_mag0.1 pid=$PID_B (GPU2,3)"
wait $PID_A; echo "[gpswm-mag] m0 exit=$? $(date)"
wait $PID_B; echo "[gpswm-mag] m01 exit=$? $(date)"
echo "==================== [gpswm-mag] DONE $(date) ===================="
