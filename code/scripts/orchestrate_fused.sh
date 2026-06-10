#!/bin/bash
# Auto-handoff for the WHOLE-VIDEO FUSION verification (§42): wait for the fused regen, then warm-start
# train on the fused data (3 GPUs — GPU0 has a stuck process), logging host RAM to catch the §41 OOM mode.
# Run detached: setsid bash code/scripts/orchestrate_fused.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=logs/orchestrate_fused.log
echo "[orch] waiting for fused regen to FULLY finish (avoid GPU conflict) $(date)" >> "$L"
while [ "$(ps aux | grep -c '[g]en_sim_dataset')" -gt 0 ]; do sleep 30; done
NTRAIN=$(ls data/maniskill_fused/*_train.pt 2>/dev/null | wc -l)
NALL=$(ls data/maniskill_fused/*.pt 2>/dev/null | wc -l)
echo "[orch] regen gate passed: $NTRAIN train / $NALL total clips $(date)" >> "$L"
if [ "$NTRAIN" -lt 50 ]; then echo "[orch] ABORT <50 train clips" >> "$L"; exit 1; fi
echo "[orch] launching warm-started fused train on GPUs 1,2,3 $(date)" >> "$L"
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 code/scripts/train_sim.py \
  --data data/maniskill_fused --out checkpoints/sim_fused \
  --resume checkpoints/sim_gen/ckpt_last.pt \
  --spatial_ground 1 --vlm_image 1 --workers 2 \
  --total_steps 3000 --max_steps 3000 --ckpt_every 500 --log_every 20 \
  >> logs/train_fused.log 2>&1 &
TPID=$!
echo "[orch] train pid=$TPID $(date)" >> "$L"
while kill -0 $TPID 2>/dev/null; do
  echo "[ram $(date +%H:%M)] $(free -g | awk '/Mem:/{print $3"/"$2"GB"}') last=$(ls checkpoints/sim_fused/ckpt_0*.pt 2>/dev/null | tail -1)" >> "$L"
  sleep 120
done
echo "[orch] train exited ($?) $(date)" >> "$L"
