#!/bin/bash
# §87 R4 vocab: v13 co-train (sim + AgiBot real + libero_90 BOOK) testing UNSEEN-NOUN generalization.
# Resume v12mix (already real-capable), settle on the book-expanded mix. Eval the held-out book split
# (heldtask was 0 = vocab ceiling — does the model now move a NEW object it generalizes to?) + guards.
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v13.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v13] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

.venv/bin/python code/scripts/prep_mix_v13.py
NTR=$(ls data/mix_v13/*_train.pt 2>/dev/null | wc -l)
NBH=$(ls data/mix_v13/*_held90.pt 2>/dev/null | wc -l)
echo "[orch] mix_v13: $NTR train, $NBH held90"
if [ "$NTR" -lt 60 ] || [ "$NBH" -lt 3 ]; then echo "[orch] ABORT: mix too small"; exit 1; fi

echo "[orch] v13 train $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29541 code/scripts/train_sim.py \
  --data data/mix_v13 --out checkpoints/libero_v13book \
  --resume checkpoints/libero_v12mix/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 2000 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v13book/ckpt_last.pt
echo "----- v13 HELD90 (unseen book instances — THE vocab test) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
  --ckpt "$CK" --data data/mix_v13 --split held90 --force_rigid_agg 1 2>&1 \
  | grep -iE "mag-ratio|EPE3D|Acc3D|SUMMARY|epi8"
echo "----- v13 sim heldseed (regression guard) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --force_rigid_agg 1 2>&1 \
  | grep -iE "mag-ratio|EPE3D|Acc3D|SUMMARY"
echo "----- v13 REAL heldreal (real-video guard) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
  --ckpt "$CK" --data data/mix_v13 --split heldreal --force_rigid_agg 1 2>&1 \
  | grep -iE "mag-ratio|EPE3D|Acc3D|SUMMARY"
echo "==================== [orch-v13] DONE $(date) ===================="
