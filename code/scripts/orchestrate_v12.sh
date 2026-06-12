#!/bin/bash
# §84 R4 entry: v12 SIM+REAL co-train (LIBERO sim + AgiBot StV2 real clips) + dual eval.
# Recipe = v11mag (R1 magnitude fix) resumed, 2500-step settle on the mix.
# Eval: (a) REAL heldreal eval_3d (the R3/R4 headline — does co-training beat the zero-shot baseline?)
#       (b) sim heldseed eval_3d + langswap (regression guard).
# Run detached: setsid bash code/scripts/orchestrate_v12.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v12.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v12] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

.venv/bin/python code/scripts/prep_mix_v12.py
NTR=$(ls data/mix_v12/*_train.pt 2>/dev/null | wc -l)
echo "[orch] mix ready: $NTR train clips"
if [ "$NTR" -lt 45 ]; then echo "[orch] ABORT: mix too small"; exit 1; fi

echo "[orch] v12 train $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29537 code/scripts/train_sim.py \
  --data data/mix_v12 --out checkpoints/libero_v12mix \
  --resume checkpoints/libero_v11mag/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 2500 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v12mix/ckpt_last.pt
echo "----- v12 REAL heldreal (3D, production +rigid_agg) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
  --ckpt "$CK" --data data/mix_v12 --split heldreal --force_rigid_agg 1 2>&1 \
  | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D|SUMMARY|epi9"
echo "----- v12 sim heldseed (3D regression guard) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --force_rigid_agg 1 2>&1 \
  | grep -iE "mag-ratio|EPE3D|5°5cm|Acc3D|SUMMARY"
echo "----- v12 sim heldseed langswap (language guard) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --n_swap 4 --force_rigid_agg 1 2>&1 \
  | grep -iE "SELECTION ACCURACY|DIRECTION cos|COHERENCE"
echo "==================== [orch-v12] DONE $(date) ===================="
