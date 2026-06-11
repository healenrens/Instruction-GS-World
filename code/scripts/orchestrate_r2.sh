#!/bin/bash
# §74 R2: train v11rigid = resume v11mag + rigid_agg ON IN-LOOP (omega-mean rotation, now trainable —
# no SVD-backward). Makes the RAW votes rigid (the plan's "原始投票即刚性") and lets omega's gradient
# flow through the projection. Keep w_mag (magnitude). Then eval_3d 3-split + langswap guard.
# Run detached: setsid bash code/scripts/orchestrate_r2.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_r2.log
exec >> "$LOG" 2>&1
echo "==================== [orch-r2] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

echo "[orch] R2 train: v11mag + rigid_agg in-loop (omega-mean), settle 1200 $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29537 code/scripts/train_sim.py \
  --data data/libero_pi3_v2 --out checkpoints/libero_v11rigid \
  --resume checkpoints/libero_v11mag/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 2e-4 --total_steps 1200 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 1 --w_mag 0.5
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v11rigid/ckpt_last.pt
for SP in train heldseed heldtask; do
  echo "----- v11rigid 3D $SP (rigid_agg baked in training) -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
    --ckpt "$CK" --data data/libero_pi3_v2 --split "$SP" 2>&1 \
    | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D|coherence"
done
echo "----- v11rigid langswap guard (heldseed) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --n_swap 4 2>&1 \
  | grep -iE "SELECTION ACCURACY|DIRECTION cos|COHERENCE"
echo "==================== [orch-r2] DONE $(date) ===================="
