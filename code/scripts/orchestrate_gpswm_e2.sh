#!/bin/bash
# PLAN next: magnitude fix (--w_mag 0.5, treats the mag-ratio 0.32 under-prediction) + E2 token-count
# sweep, on the E1-winning geom_mode=xyz. 4 GPUs: L512 DDP on GPU0,1 ; L1024 DDP on GPU2,3.
# Run: setsid bash code/scripts/orchestrate_gpswm_e2.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [gpswm-e2] START $(date) ===================="
COMMON="--data data/mix_v15 --geom_mode xyz --steps 1250 --w_mag 0.5 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 625"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --master_port=29561 \
  code/scripts/train_gpstoken_wm.py $COMMON --L 512 --out checkpoints/gpswm_L512mag \
  > logs/gpswm_L512.log 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun --nproc_per_node=2 --master_port=29562 \
  code/scripts/train_gpstoken_wm.py $COMMON --L 1024 --out checkpoints/gpswm_L1024mag \
  > logs/gpswm_L1024.log 2>&1 &
PID_B=$!
echo "[gpswm-e2] L512 pid=$PID_A (GPU0,1)  L1024 pid=$PID_B (GPU2,3)"
wait $PID_A; echo "[gpswm-e2] L512 exit=$? $(date)"
wait $PID_B; echo "[gpswm-e2] L1024 exit=$? $(date)"
echo "==================== [gpswm-e2] DONE $(date) ===================="
