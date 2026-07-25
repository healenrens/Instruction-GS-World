#!/bin/bash
# Robust detached launcher for the prep-cache precompute. Redirects its OWN output to the log so the
# launching ssh can send the setsid fds to /dev/null and return immediately.
cd /mnt/pfs/public/xuhaoming/instruct_gs_world || exit 1
exec > logs/cache_prep.log 2>&1
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       XDG_CACHE_HOME=/mnt/pfs/public/xuhaoming/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       CUDA_VISIBLE_DEVICES=0,1,2,3
exec .venv/bin/torchrun --nproc_per_node=4 --master_port=29564 code/scripts/train_vla.py \
  --cache_prep --prep_cache data/rt2_joint_prepcache --data data/rt2_joint \
  --norm_stats data/rt2_act/norm_stats.pt --geom_mode xyz --img_loss 1 --L 512 --n_state_tokens 1
