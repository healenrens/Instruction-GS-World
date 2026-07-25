#!/bin/bash
# §72 R1: retrain v9-lang + mover_magnitude_loss (--w_mag) to open the dyn-gate on true movers
# (the magnitude-collapse fix). Train RAW (rigid_agg off; it's an inference projection), then eval_3d
# 3-split in the production config (--force_rigid_agg 1). v9-lang recipe otherwise, settle 1500 steps.
# Run detached: setsid bash code/scripts/orchestrate_r1.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_r1.log
exec >> "$LOG" 2>&1
echo "==================== [orch-r1] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

echo "[orch] R1 train: v9-lang + w_mag 0.5, settle 1500 $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29535 code/scripts/train_sim.py \
  --data data/libero_pi3_v2 --out checkpoints/libero_v11mag \
  --resume checkpoints/libero_v9lang/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 1500 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v11mag/ckpt_last.pt
for SP in train heldseed heldtask; do
  echo "----- v11mag 3D $SP (production config: +rigid_agg) -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
    --ckpt "$CK" --data data/libero_pi3_v2 --split "$SP" --force_rigid_agg 1 2>&1 \
    | grep -iE "mag-ratio|EPE3D|5°5cm|Acc3D|coherence|SUMMARY"
done
echo "==================== [orch-r1] DONE $(date) ===================="
