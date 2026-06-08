#!/usr/bin/env bash
# Launch the 4×A100 DDP training of the language-conditioned Gaussian dynamics model.
# Prereq: clips cached under ./data/clips_v1 (see cache_clips.py).
set -euo pipefail
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

RUN=${1:-run2}
# IMPORTANT: use the venv's torchrun (conda's torchrun lacks gsplat/transformers).
# v2: 1.77B model — Qwen3-VL(+LoRA) in-loop, per-layer interaction, 28-block dynamics.
./.venv/bin/torchrun --nproc_per_node=4 code/scripts/train.py \
  --data ./data/clips_v1 \
  --out ./checkpoints/${RUN} \
  --epochs 60 --K 8 --M 2048 --dim 1536 --layers 28 --heads 16 --lora_r 16 \
  --lr 2e-4 --lr_lora 1e-4 --warmup 800 --noise_frac 0.02 \
  --log_every 20 --ckpt_every 500
  # resume: append  --resume ./checkpoints/${RUN}/ckpt_last.pt
