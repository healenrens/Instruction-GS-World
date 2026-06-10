#!/bin/bash
# Auto-handoff for the "random-start 4s-window" experiment: wait for the parallel
# clean-GT regen (gen_sim_dataset workers) to finish, then launch the 10k warm-started
# DDP training on the new data. Run detached: setsid bash code/scripts/orchestrate_4s.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=logs/orchestrate_4s.log
echo "[orch] waiting for regen workers at $(date)" >> "$L"
while [ "$(ps aux | grep -c '[g]en_sim_dataset')" -gt 0 ]; do sleep 30; done
NCLIP=$(ls data/maniskill_4s/*.pt 2>/dev/null | wc -l)
echo "[orch] regen done at $(date); clips=$NCLIP" >> "$L"
if [ "$NCLIP" -lt 100 ]; then
  echo "[orch] ABORT: only $NCLIP clips (<100) -> NOT launching train" >> "$L"
  exit 1
fi
echo "[orch] launching 10k warm-started train (resume sim_gen/ckpt_last) at $(date)" >> "$L"
torchrun --nproc_per_node=4 code/scripts/train_sim.py \
  --data data/maniskill_4s --out checkpoints/sim_4s \
  --resume checkpoints/sim_gen/ckpt_last.pt \
  --spatial_ground 1 --vlm_image 1 \
  --total_steps 10000 --max_steps 10000 --ckpt_every 500 --log_every 20 \
  >> logs/train_4s.log 2>&1
echo "[orch] train exited ($?) at $(date)" >> "$L"
