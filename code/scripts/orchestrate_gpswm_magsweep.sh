#!/bin/bash
# CLEAN controlled w_mag sweep — removes the DDP/L/step confounds. SINGLE-GPU each (the known-good E1
# setting: L512, 2500 steps, lr3e-4), ONLY w_mag varies: {0, 0.05, 0.1, 0.2} on GPU0/1/2/3.
# w_mag=0 == reproduce E1 (sanity that 7.2/0.74 is real). Goal: find w_mag that lifts mag-ratio 0.32->~0.9
# WITHOUT dropping dcos<0.7. Run: setsid bash code/scripts/orchestrate_gpswm_magsweep.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1
echo "==================== [magsweep] START $(date) ===================="
COMMON="--data data/mix_v15 --geom_mode xyz --L 512 --steps 2500 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --save_every 2500"
i=0
for WM in 0.0 0.05 0.1 0.2; do
  tag=$(echo $WM | tr -d .)
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python code/scripts/train_gpstoken_wm.py $COMMON \
    --w_mag $WM --out checkpoints/gpswm_sw$tag > logs/gpswm_sw$tag.log 2>&1 &
  echo "[magsweep] w_mag=$WM -> GPU$i pid=$! (sw$tag)"
  i=$((i+1))
done
wait
echo "==================== [magsweep] DONE $(date) ===================="
