#!/bin/bash
# §84 R3 batch: generate real-AgiBot training clips (StV2 backend, clip-builder v1.2) for episodes
# 0..N-1 of task_327, sequential on one GPU (~2.5min each). Per-episode failures tolerated (logged).
# Usage: bash gen_agibot_batch.sh [N=20] [GPU=0]
N=${1:-20}
GPU=${2:-0}
cd /mnt/pfs/public/xuhaoming/SpaTrackerV2
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000
export CUDA_VISIBLE_DEVICES=$GPU GD_REPO=IDEA-Research/grounding-dino-base PYTHONDONTWRITEBYTECODE=1
WS=/mnt/pfs/public/xuhaoming/instruct_gs_world
L=$WS/logs/gen_agibot_batch.log
echo "==================== [agibot-batch] START N=$N $(date) ====================" >> "$L"
for ep in $(seq 0 $((N - 1))); do
  echo "----- [agibot-batch] ep$ep $(date) -----" >> "$L"
  timeout 600 $WS/.venv/bin/python $WS/code/scripts/agibot_clip_stv2.py "$ep" >> "$L" 2>&1 \
    || echo "[agibot-batch] ep$ep FAILED exit=$?" >> "$L"
done
OKN=$(ls $WS/data/_agibot/clip_stv2_ep*.pt 2>/dev/null | wc -l)
echo "==================== [agibot-batch] DONE: $OKN clips $(date) ====================" >> "$L"
