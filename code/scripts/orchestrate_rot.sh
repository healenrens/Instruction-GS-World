#!/bin/bash
# §86 R4: rotation attack — does STRONGER rotation supervision (w_traj_rot 0.2->1.0) reduce the
# rot-err (19 deg on clean sim GT, where pred error ~= GT rotation signal)? Clean sim test (analytic
# GT), resume v11mag, 1000-step settle. Eval_3d sim heldseed (rot-err / 5deg5cm / EPE3D / mag guard).
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_rot.log
exec >> "$LOG" 2>&1
echo "==================== [orch-rot] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

echo "[orch] rotation attack: w_traj_rot 1.0, resume v11mag $(date)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29539 code/scripts/train_sim.py \
  --data data/libero_pi3_v2 --out checkpoints/libero_v11rot \
  --resume checkpoints/libero_v11mag/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 1000 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5 --w_traj_rot 1.0
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v11rot/ckpt_last.pt
echo "----- v11rot sim heldseed (3D: rotation focus) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --force_rigid_agg 1 2>&1 \
  | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D|epi0"
echo "==================== [orch-rot] DONE $(date) ===================="
