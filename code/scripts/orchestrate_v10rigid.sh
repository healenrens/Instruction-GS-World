#!/bin/bash
# §67 V3+V4: train v10-rigid (resume v9-lang, --rigid_agg 1 baked in, w_rigid auto-zeroed) then the
# 3-split langswap eval WITH coherence metrics. v9-lang recipe otherwise (data/libero_pi3_v2 GT-mask).
# Run detached: setsid bash code/scripts/orchestrate_v10rigid.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v10rigid.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v10rigid] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

# ---- V3: train (resume v9-lang, rigid_agg on) -------------------------------------------------
echo "[orch] V3: train v10-rigid $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29533 code/scripts/train_sim.py \
  --data data/libero_pi3_v2 --out checkpoints/libero_v10rigid \
  --resume checkpoints/libero_v9lang/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 800 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 1
echo "[orch] train exit=$? $(date)"

# ---- V4: 3-split langswap eval (rigid_agg baked into ckpt -> no --force flag) ------------------
CK=checkpoints/libero_v10rigid/ckpt_last.pt
for SP in train heldseed heldtask; do
  echo "----- v10-rigid $SP -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
    --ckpt "$CK" --data data/libero_pi3_v2 --split "$SP" --n_swap 4 2>&1 \
    | grep -iE "rigid_agg=|SELECTION ACCURACY|DIRECTION cos|COHERENCE"
done
echo "==================== [orch-v10rigid] DONE $(date) ===================="
