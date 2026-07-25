#!/bin/bash
# §61 B2: detached orchestrator — open-vocab data gen -> train v9-lang-ov (same recipe as v9-lang,
# resume v7_pi3, 800 steps) -> langswap 3-split eval. The honesty test: can the model trained on the
# pipeline's OWN open-vocab masks match the GT-mask v9-lang? Run: setsid bash orchestrate_v9ov.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
LOG=logs/orchestrate_v9ov.log
exec >> "$LOG" 2>&1
echo "==================== [orch-v9ov] START $(date) ===================="

TRAIN="0 10 30 50 70 100 110 130 150 160 180 200 210 250 260 300 310 340 350 370"
HELDTASK="410 430 450 470"
HELDSEED="40 140 240 330"

# ---- Stage 1: open-vocab data generation ------------------------------------------------------
echo "[orch] Stage 1: open-vocab data gen $(date)"
bash code/scripts/gen_libero_pi3_v2_ov.sh "$TRAIN" "$HELDTASK" "$HELDSEED"
NTRAIN=$(ls data/libero_pi3_v2_ov/*_train.pt 2>/dev/null | wc -l)
NALL=$(ls data/libero_pi3_v2_ov/*.pt 2>/dev/null | wc -l)
echo "[orch] gen done: $NALL clips ($NTRAIN train)"
if [ "$NTRAIN" -lt 30 ]; then
  echo "[orch] ABORT: too few train clips ($NTRAIN < 30) — open-vocab gen failed"; exit 1
fi

# ---- Stage 2: train v9-lang-ov (identical recipe to v9-lang; resume v7_pi3) --------------------
echo "[orch] Stage 2: train v9-lang-ov $(date)"
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000
.venv/bin/torchrun --nproc_per_node=4 --master_port=29531 code/scripts/train_sim.py \
  --data data/libero_pi3_v2_ov --out checkpoints/libero_v9lang_ov \
  --resume checkpoints/libero_v7_pi3/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 800 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --w_rigid 0.5 --gate_entity_pool 1 \
  --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 --entity_head 0 --w_resid 0
echo "[orch] train exit=$? $(date)"

# ---- Stage 3: langswap 3-split eval (compare to v9-lang GT-mask numbers) -----------------------
echo "[orch] Stage 3: langswap 3-split eval $(date)"
CK=checkpoints/libero_v9lang_ov/ckpt_last.pt
for SP in train heldseed heldtask; do
  echo "----- v9lang-OV $SP -----"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/eval_langswap.py \
    --ckpt "$CK" --data data/libero_pi3_v2_ov --split "$SP" --n_swap 4 2>&1 \
    | grep -iE "SELECTION ACCURACY|DIRECTION cos|endpoint"
done
echo "==================== [orch-v9ov] DONE $(date) ===================="
