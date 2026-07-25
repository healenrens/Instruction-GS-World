#!/bin/bash
# §91 bases-RESIDUAL (the low-rank motion bases' fair second chance, user-directed):
# clean A/B vs v15 — SAME base (v12mix), SAME data (mix_v15 incl. libero_goal rotation),
# the ONLY variable = --motion_bases 10 --bases_mode residual (bases ADD low-rank structure on top
# of the per-control field instead of replacing it; pure mode killed direction, §89).
# Auto-queues: waits for v15 to finish, then trains + the same quadruple eval.
# Run detached: setsid bash code/scripts/orchestrate_v16_basesres.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v16.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v16] QUEUED $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

# gate on v15: both its training process AND its DONE marker (evals finished)
while pgrep -f "train_sim.py.*libero_v1[5]" > /dev/null; do sleep 90; done
until grep -q "orch-v15] DONE" logs/orchestrate_v15.log 2>/dev/null; do sleep 90; done
echo "[orch] v15 finished -> starting bases-residual $(date)"

.venv/bin/torchrun --nproc_per_node=4 --master_port=29547 code/scripts/train_sim.py \
  --data data/mix_v15 --out checkpoints/libero_v16basesres \
  --resume checkpoints/libero_v12mix/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 2500 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5 --w_traj_rot 0.3 \
  --motion_bases 10 --bases_mode residual --w_coef_seg 0.1 --detach_state_rot 1
echo "[orch] train exit=$? $(date)"

CK=checkpoints/libero_v16basesres/ckpt_last.pt
for SP_DATA in "heldgoal:data/mix_v15" "held90:data/mix_v15" "heldreal:data/mix_v15" "heldseed:data/libero_pi3_v2"; do
  SP=${SP_DATA%%:*}; DD=${SP_DATA##*:}
  echo "----- v16basesres $SP (3D +rigid_agg) -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
    --ckpt "$CK" --data "$DD" --split "$SP" --force_rigid_agg 1 2>&1 \
    | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D"
done
echo "----- v16basesres langswap heldseed (direction guard — pure mode died here at -0.14) -----"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
  --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --n_swap 4 --force_rigid_agg 1 2>&1 \
  | grep -iE "SELECTION ACCURACY|DIRECTION cos|COHERENCE"
echo "==================== [orch-v16] DONE $(date) ===================="
