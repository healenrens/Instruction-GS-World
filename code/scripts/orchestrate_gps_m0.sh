#!/bin/bash
# §93 GPSToken M0: keypoint-selection A/B at MATCHED control budget (M=256). The ONLY variable is the
# control INDEX source: ARM-A = random mover-biased (sample_controls); ARM-B = GPSToken entropy-partition
# + GT-mover saliency (beta=30). Everything else (model, losses, LBS, per-control translation field) is
# byte-identical. Warm-start v12mix. Decisive metric = train_corr / dcos / ratio (localization) + dir-cos
# must NOT regress (the §89 kill metric). Run: setsid bash code/scripts/orchestrate_gps_m0.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/gps_m0.log
exec >> "$LOG" 2>&1
echo "==================== [gps-m0] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

COMMON="--data data/mix_v15 --resume checkpoints/libero_v12mix/ckpt_last.pt \
  --K 12 --M 256 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 600 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5 --w_traj_rot 0.3"

echo "-------------------- ARM-B GPSToken (beta=30, GT-mover saliency) --------------------"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29561 code/scripts/train_sim.py $COMMON \
  --out checkpoints/gps_m0_gpstok --use_gpstoken 1 --gps_motion_beta 30
echo "[gps-m0] ARM-B exit=$? $(date)"

echo "-------------------- ARM-A random (matched M=256) --------------------"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29562 code/scripts/train_sim.py $COMMON \
  --out checkpoints/gps_m0_rand --use_gpstoken 0
echo "[gps-m0] ARM-A exit=$? $(date)"
echo "==================== [gps-m0] DONE $(date) ===================="
