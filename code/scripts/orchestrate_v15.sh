#!/bin/bash
# §90 R4 FINAL: v15 = v12 base + book96 (vocab, WIN=96 fixed) + libero_goal (rotation data, §89 lever).
# NO new architecture — the per-control field learns rotation as tangential motion IF the data has
# large rotations (libero_goal drawer/knob). resume v12mix, 2500 steps. Quadruple eval.
# Run detached: setsid bash code/scripts/orchestrate_v15.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v15.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v15] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

# wait for any prior training to free GPUs
while pgrep -f "train_sim.py.*v1[34]" > /dev/null; do sleep 60; done
.venv/bin/python code/scripts/prep_mix_v15.py
NTR=$(ls data/mix_v15/*_train.pt 2>/dev/null | wc -l)
echo "[orch] mix_v15: $NTR train clips"
[ "$NTR" -lt 70 ] && { echo "[orch] ABORT: mix too small ($NTR)"; exit 1; }

echo "[orch] v15 train $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29545 code/scripts/train_sim.py \
  --data data/mix_v15 --out checkpoints/libero_v15 --resume checkpoints/libero_v12mix/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 2500 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5 --w_traj_rot 0.3
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v15/ckpt_last.pt
for SP_DATA in "heldgoal:data/mix_v15" "held90:data/mix_v15" "heldreal:data/mix_v15" "heldseed:data/libero_pi3_v2"; do
  SP=${SP_DATA%%:*}; DD=${SP_DATA##*:}
  echo "----- v15 $SP (3D +rigid_agg) -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
    --ckpt "$CK" --data "$DD" --split "$SP" --force_rigid_agg 1 2>&1 \
    | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D"
done
echo "----- v15 sim heldseed langswap (language guard) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --n_swap 4 --force_rigid_agg 1 2>&1 \
  | grep -iE "SELECTION ACCURACY|DIRECTION cos|COHERENCE"
echo "==================== [orch-v15] DONE $(date) ===================="
