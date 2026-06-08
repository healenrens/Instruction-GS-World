# Instruct-GS-World — Handoff / Run Guide

Language-conditioned 3D-Gaussian dynamics (world) model: given a 3DGS scene + a
natural-language instruction, predict per-Gaussian transforms and roll them out
autoregressively for ≥10 s. Full rationale & decision log: `agent.md`.

## Where things are (server `ssh -p 8600 root@106.13.104.32`)
- Workspace: `/mnt/pfs/public/xuhaoming/instruct_gs_world/`
- venv: `./.venv` (torch 2.8+cu126, gsplat, transformers 5.x). Proxy for any download:
  `export http_proxy=http://10.66.65.186:18000 https_proxy=$http_proxy`
- Backbone: `model_zoo/Cosmos-Reason2-2B` (Qwen3-VL-2B). Lifter ckpt: `checkpoints/Pi3/model.safetensors`.
- Code: `code/igsw/` (package) + `code/scripts/` (entrypoints). Mirrored locally at
  `/Users/hela/Instruct-GS-World/code` (rsync code only).

## ★ Recommended: streaming training (full corpus, zero clip storage, DIRECT 3D loss)
No pre-caching — decode→lift→track→train→discard over ALL 137,768 episodes / 183 tasks:
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup ./.venv/bin/torchrun --nproc_per_node=4 code/scripts/train_stream.py \
  --out ./checkpoints/stream2 --K 8 --M 2048 --dim 1536 --layers 28 --n_query 16 \
  --w_traj_pos 1.0 --w_traj_vel 1.0 --w_render 0.1 --checkpoint_every 2 \
  --grad_accum 2 --workers 6 --prefetch 4 --render_steps 2 \
  --lr 2e-4 --warmup 1000 --total_steps 300000 --ckpt_every 1000 \
  > logs/train_stream2.log 2>&1 &
```
- **PRIMARY loss = direct 3D per-Gaussian motion** (position + velocity), via CoTracker3
  correspondence + Pi3 point-map sampling (GT trajectory in the prediction gauge).
  **Render/PSNR is auxiliary** (`--w_render 0.1`, only `--render_steps` rendered).
- Qwen3-VL FROZEN; learnable special-token head aggregates all 28 layers.
- `--checkpoint_every 2` (half grad-ckpt) → ~58–65 GB/GPU, ~0.38 it/s; `--grad_accum` = effective batch.
- Resume: `--resume ./checkpoints/stream2/ckpt_last.pt`.
- Eval: `code/scripts/eval_longhorizon.py --ckpt checkpoints/stream2/ckpt_last.pt --N 100`.
- Speed knobs: `--checkpoint_every 0` (no ckpt, ~80 GB, fastest/risky), `--grad_accum N`, `--workers/--prefetch`.

## (Legacy) offline-cache pipeline
1. **Cache clips** (data prep, the heavy step; shard across GPUs):
   ```
   for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s nohup ./.venv/bin/python code/scripts/cache_clips.py \
     --out_dir ./data/clips_v1 --n_tasks 40 --eps_per_task 15 --clips_per_ep 6 \
     --K 12 --stride 3 --shard $s --num_shards 4 > logs/cache_shard$s.log 2>&1 & done
   ```
   ~14 MB/clip, ~1.6 s/clip. Resumable (skips existing). Scale up `--n_tasks/--eps_per_task/--clips_per_ep`.
2. **Train (4×A100 DDP) — v2, 1.77B**:
   ```
   bash run_train.sh run2     # venv torchrun; Qwen3-VL(+LoRA) in-loop, per-layer interaction, 28-block dynamics
   ```
   Log `logs/train_run2.log`; ckpts `checkpoints/run2/ckpt_*.pt` + `ckpt_last.pt` (every 500 steps);
   TensorBoard `checkpoints/run2/tb`. Resume: append `--resume ./checkpoints/run2/ckpt_last.pt`.
   ~40 GB/GPU, ~0.4 it/s (×4) ≈ 37 min/epoch. Trainable = 1.768B (dyn 1.66B + proj 91M + LoRA 17.4M).
   Headroom (40/80 GB) → can raise `--dim 2048` or `--K 12`, or move to FSDP for multi-B.
3. **Long-horizon eval (≥10 s demo)**:
   ```
   ./.venv/bin/python code/scripts/eval_longhorizon.py --ckpt checkpoints/run1/ckpt_last.pt \
     --ep 0 --f0 60 --N 100 --stride 3
   ```
   Writes `outputs/longhorizon/rollout.mp4` (GT | static-baseline | predicted) + PSNR-vs-time curve.
   Test language control by passing a different `--instruction`.

## Architecture (v2, ≥1B)
`igsw/model_full.py::InstructGSWorldModel`:
- **Qwen3-VL-2B in-loop + LoRA** (`igsw/dynamics/conditioning.py`): base frozen (bf16),
  LoRA adapters (fp32, trainable) → language/vision features adapt. Returns ALL 28 layer
  hidden states.
- **Per-layer interaction**: 28 Linear(2048→1536) projections; **dynamics block j cross-attends
  to Qwen3-VL layer j** (`igsw/dynamics/{model,transformer}.py`). AdaLN cond from pooled last layer.
- **~1.66B dynamics** (d1536/28L/16H), on-manifold deltas, identity-at-init.
- **SC-GS** (`igsw/gaussians/deform.py`): dynamics on M=2048 control gaussians → LBS to ~120k dense.
- One DDP forward = VLM-encode → K-step SC-GS rollout (dynamics called K× inside; `static_graph=True`,
  `broadcast_buffers=False`).

## Feasibility status (verified)
Render-supervised conditioned rollout beats the static baseline (+1.7–2.5 dB on
future frames) on real AgiBot data (v1 80M proof). v2 (1.77B) now training. Absolute
render quality is bottlenecked by the naive Pi3 lift (~10 PSNR); next quality levers:
per-frame 3DGS optimization of G0 and/or St4RTrack trajectory supervision (`agent.md` §9).
Note: `scripts/overfit_clip*.py` are v1-legacy feasibility demos (pre-redesign).

## Known knobs / gotchas
- bf16 is required for attention (fp32 SDPA OOMs at 16k tokens); gsplat runs fp32.
- AgiBot videos are AV1 → decoded via PyAV/libdav1d.
- Pi3/VGGT weights are non-commercial (research only).
- Increase `--K` for longer training-time rollout (more memory; grad-checkpointing is on).
