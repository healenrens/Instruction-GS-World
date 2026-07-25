# Wrist-camera conditioning — code+data READY (not yet trained)

Goal: feed **head + left + right** wrist views to the frozen Qwen3-VL (richer context for the hard
mid-manipulation frames where the head cam can't see the gripper/object), while the **3D/GPSToken stays
head-only**. Verified: encode smoke-test `cos=1.000` → head grid byte-identical regardless of wrist images
(Qwen is causal; head=image #0 can't see the later wrist images; only context/text tokens attend to all).

## What changed (all synced to the server)
- `igsw/dynamics/conditioning.py`: `build_inputs(text, image)` accepts a **list** of images (head first);
  `image_grid_features` slices the **first image's** tokens.
- `igsw/gpstoken_wm/wm_model.py`: `_grid_from_hidden`, `_grid_from_row` slice image #0; `encode_cond_batch`
  indexes each clip's HEAD thw by cumulative image offset.
- `scripts/train_vla.py`: `--wrist 1`; `_imgs_with_wrist()` feeds `[head,left,right]` in build_batch/_single.
- `scripts/rt2_add_wrist.py`: **patch script — DONE** (all ~17.6k rt2_joint clips now carry `left_rgb`/
  `right_rgb` uint8 [240,320,3]; 0 errors).
- `scripts/rt2_policy_server.py` + `rt2_rollout_client.py`: `--wrist 1` (server uses left/right; client
  sends them from the sim obs). Gated → the current no-wrist ckpt is unaffected.

## To TRAIN (when ready) — 4×A100, env from _run_cache_prep.sh
```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       XDG_CACHE_HOME=/mnt/pfs/public/xuhaoming/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       CUDA_VISIBLE_DEVICES=0,1,2,3

# 1) rebuild the prep cache WITH wrist (encode is now 3 images) -> ~1-2h
setsid .venv/bin/torchrun --nproc_per_node=4 --master_port=29564 code/scripts/train_vla.py \
  --cache_prep --prep_cache data/rt2_joint_prepcache_wrist --data data/rt2_joint \
  --norm_stats data/rt2_act/norm_stats.pt --geom_mode xyz --img_loss 1 --L 512 --n_state_tokens 1 \
  --placement entropy --wrist 1  > logs/cache_wrist.log 2>&1 < /dev/null &

# 2) train (same config as vla_50k_v2 + --wrist; warm-start from vla_040000 for fast convergence,
#    or --init_from "" for from-scratch like the original)
setsid .venv/bin/torchrun --nproc_per_node=4 --master_port=29565 code/scripts/train_vla.py \
  --data data/rt2_joint --prep_cache data/rt2_joint_prepcache_wrist --norm_stats data/rt2_act/norm_stats.pt \
  --geom_mode xyz --img_loss 1 --L 512 --n_state_tokens 1 --placement entropy --wrist 1 \
  --deepspeed 1 --steps 50000 --batch 24 --accum 3 --lr 5e-05 --w_flow 1.0 --w_act 1.0 --weight_decay 0.01 \
  --init_from checkpoints/vla_50k_v2/vla_040000.pt --out checkpoints/vla_wrist \
  > logs/train_wrist.log 2>&1 < /dev/null &
```

## To DEPLOY (after training)
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/rt2_policy_server.py \
  --ckpt checkpoints/vla_wrist/vla_0XXXXX.pt --port 9010 --wrist 1            # server
# rollout client: add  --wrist 1   to the rt2_rollout_client.py command
```

## Why this should help (from the §debug)
Data/normalization are correct; predicted dq **magnitude** is correct (deploy p95 ~0.05–0.09 = GT). The
failures are **wrong DIRECTION on mid-manipulation frames** (held dir-cos 0.4 vs 0.9 on clean window-start
frames). Those are exactly the frames where the head cam can't see the gripper-object contact → wrist views
are the missing signal.
