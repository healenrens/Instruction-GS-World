#!/bin/bash
# Direction-preserving magnitude fix: motion-weighted geom loss (up-weight high-motion tokens in the
# POSITION smooth-L1, so the model serves big movers without the dead-end relative term). Sweep w_motion
# {0,1,3,5} at the locked v1 config (xyz L1024 no-mag), single-GPU each, 2500 steps. Goal: lift mag-ratio
# 0.24->~0.8 while KEEPING dcos. Run: setsid bash code/scripts/orchestrate_gpswm_motion.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [motion] START $(date) ===================="
COMMON="--data data/mix_v15 --geom_mode xyz --L 1024 --steps 2500 --w_mag 0.0 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 2500"
i=0
for WM in 0 1 3 5; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON \
    --w_motion $WM --out checkpoints/gpswm_mw$WM > logs/gpswm_mw$WM.log 2>&1 &
  echo "[motion] w_motion=$WM -> GPU$i pid=$!"
  i=$((i+1))
done
wait
echo "==================== [motion] DONE $(date) ===================="
