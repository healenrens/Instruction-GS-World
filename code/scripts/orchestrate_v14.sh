#!/bin/bash
# §87 the rotation-architecture A/B: two competing redesigns trained IN PARALLEL (2 GPUs each).
#   v14erot  (A+B+C): entity-level 6D rotation head (attention readout) + position-space rot loss
#                     + detached rotation state chain. Translation path untouched. 1500 steps.
#   v14bases (SoM):   B=10 low-rank SE(3) motion bases, PURE mode (bases REPLACE per-control motion;
#                     translation must be re-learned through the bases) -> longer, 3000 steps.
# Both resume v13mix (sim+real+book trunk), train on data/mix_v13, then eval_3d (rotation focus).
# Run detached: setsid bash code/scripts/orchestrate_v14.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v14.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v14] START $(date) ===================="
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 PYTHONDONTWRITEBYTECODE=1

# wait for v13 to release the GPUs
while pgrep -f "train_sim.py.*v13" > /dev/null; do sleep 60; done
echo "[orch] GPUs free, launching both trainings $(date)"
RESUME=checkpoints/libero_v13mix/ckpt_last.pt
[ -f "$RESUME" ] || RESUME=checkpoints/libero_v12mix/ckpt_last.pt
echo "[orch] resume=$RESUME"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --master_port=29541 \
  code/scripts/train_sim.py \
  --data data/mix_v13 --out checkpoints/libero_v14erot --resume "$RESUME" \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 1500 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5 \
  --entity_rot 1 --w_erot 0.5 --detach_state_rot 1 > logs/train_v14erot.log 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun --nproc_per_node=2 --master_port=29542 \
  code/scripts/train_sim.py \
  --data data/mix_v13 --out checkpoints/libero_v14bases --resume "$RESUME" \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 3000 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --gate_entity_pool 1 --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 \
  --entity_head 0 --w_resid 0 --rigid_agg 0 --w_mag 0.5 \
  --motion_bases 10 --bases_mode pure --w_coef_seg 0.1 --detach_state_rot 1 \
  > logs/train_v14bases.log 2>&1 &
PID_B=$!
echo "[orch] erot pid=$PID_A bases pid=$PID_B"
wait $PID_A; echo "[orch] v14erot train done exit=$? $(date)"
wait $PID_B; echo "[orch] v14bases train done exit=$? $(date)"

for V in v14erot v14bases; do
  CK=checkpoints/libero_$V/ckpt_last.pt
  for SP in heldseed; do
    echo "----- $V sim $SP (3D, RAW — did the head itself learn rotation?) -----"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
      --ckpt "$CK" --data data/libero_pi3_v2 --split "$SP" 2>&1 \
      | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D"
  done
  echo "----- $V REAL heldreal (3D RAW) -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_3d.py \
    --ckpt "$CK" --data data/mix_v12 --split heldreal 2>&1 \
    | grep -iE "mag-ratio|EPE3D|5°5cm|GT-rot|Acc3D"
  echo "----- $V sim heldseed langswap (language guard) -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
    --ckpt "$CK" --data data/libero_pi3_v2 --split heldseed --n_swap 4 2>&1 \
    | grep -iE "SELECTION ACCURACY|DIRECTION cos|COHERENCE"
done
echo "==================== [orch-v14] DONE $(date) ===================="
