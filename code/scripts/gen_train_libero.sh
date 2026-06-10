#!/bin/bash
# Autonomous LIBERO PURE-VIDEO gen → train. Generates clips via video_gt.py (St4RTrack RGB→3DGS+motion,
# NO GT depth/camera used to generate), task-diverse (every 20th ep spans the 10 libero_object tasks;
# last 2 tasks = heldtask), 4 GPUs, then launches the dyngate7 recipe. Run: setsid bash code/scripts/gen_train_libero.sh &
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
L=logs/gen_train_libero.log
mkdir -p data/libero_video logs
echo "[gen] start $(date)" >> "$L"
EPIS="0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460"
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache
i=0
for epi in $EPIS; do
  gpu=$((i % 4)); split=train; [ "$epi" -ge 400 ] && split=heldtask
  out=data/libero_video/epi$(printf %06d "$epi")_${split}.pt
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python code/scripts/video_gt.py --epi "$epi" --K 12 \
      --out "$out" --split "$split" >> logs/vgen.log 2>&1 < /dev/null &
  fi
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done
wait
NALL=$(ls data/libero_video/*.pt 2>/dev/null | wc -l); NTRAIN=$(ls data/libero_video/*_train.pt 2>/dev/null | wc -l)
echo "[gen] done: $NALL clips ($NTRAIN train) $(date)" >> "$L"
if [ "$NTRAIN" -lt 4 ]; then echo "[ABORT] <4 train clips" >> "$L"; exit 1; fi
echo "[train] launch dyngate7 recipe on LIBERO pure-video $(date)" >> "$L"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --nproc_per_node=4 --master-port 29540 code/scripts/train_sim.py \
  --data data/libero_video --out checkpoints/libero_v1 \
  --resume checkpoints/sim_gen/ckpt_last.pt --spatial_ground 1 --vlm_image 1 \
  --dyn_gate 1 --w_dyn 1.0 --sem_dim 16 --w_seg 0.3 --gate_uses_sem 1 --obj_focus 1.5 --w_mag 0.5 \
  --lr 5e-5 --lr_sg 5e-4 --warmup 30 --total_steps 1000 --max_steps 1000 --workers 2 --ckpt_every 200 --log_every 20 \
  >> logs/train_libero.log 2>&1
echo "[train] exited ($?) $(date)" >> "$L"
