#!/bin/bash
# §50 LIBERO pure-video data v3: per-entity rigid motion (object/arm/gripper) + size-depth cue +
# hole fill + MOVER-FILTERED episodes (failed/approach-only demos excluded by the scan).
# Usage: bash gen_libero_v3.sh "<train epis>" "<heldtask epis>"
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
L=logs/gen_libero_v3.log
mkdir -p data/libero_video_v3 logs
rm -f data/libero_video_v3/*.pt
echo "[gen-v3] start $(date)" > "$L"
TRAIN_EPIS="$1"; HELD_EPIS="$2"
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache
i=0
for epi in $TRAIN_EPIS; do
  gpu=$((i % 4))
  out=data/libero_video_v3/epi$(printf %06d "$epi")_train.pt
  CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python code/scripts/video_gt.py --epi "$epi" --K 12 \
    --out "$out" --split train >> logs/vgen_v3.log 2>&1 < /dev/null &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done
wait
for epi in $HELD_EPIS; do
  gpu=$((i % 4))
  out=data/libero_video_v3/epi$(printf %06d "$epi")_heldtask.pt
  CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python code/scripts/video_gt.py --epi "$epi" --K 12 \
    --out "$out" --split heldtask >> logs/vgen_v3.log 2>&1 < /dev/null &
  i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done
wait
NALL=$(ls data/libero_video_v3/*.pt 2>/dev/null | wc -l); NTRAIN=$(ls data/libero_video_v3/*_train.pt 2>/dev/null | wc -l)
echo "[gen-v3] done: $NALL clips ($NTRAIN train) $(date)" >> "$L"
grep -hE "\[dbg\] (manipulated|ent|hole)" logs/vgen_v3.log | tail -120 >> "$L"
