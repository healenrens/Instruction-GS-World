#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
STAGING_ROOT="${STAGING_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_preflight_20260720_v2}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/rt2_visual_episode_cache_20260720}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pfs/public/xuhaoming/Cosmos-3-Finetune/data/RoboTwin2}"
printf "%s\n" "$$" >"${LOG_ROOT}/promotion_v2.pid"

old_pid="$(cat "${LOG_ROOT}/launcher.pid")"
while kill -0 "${old_pid}" 2>/dev/null; do
    sleep 60
done

files=(
    code/igsw/adaptive_gaussian_wm/checkpointing.py
    code/igsw/adaptive_gaussian_wm/episode_cache_contract.py
    code/igsw/adaptive_gaussian_wm/episode_cache_encoding.py
    code/igsw/adaptive_gaussian_wm/episode_sequence_dataset.py
    code/igsw/adaptive_gaussian_wm/experiment_tracking.py
    code/igsw/adaptive_gaussian_wm/goal_prior_checkpointing.py
    code/igsw/adaptive_gaussian_wm/group_balanced_sampler.py
    code/igsw/adaptive_gaussian_wm/sequence_contract.py
    code/igsw/adaptive_gaussian_wm/training_loop.py
    code/scripts/cache_rt2_visual_episodes.py
    code/scripts/cache_rt2_visual_episodes_4gpu.sh
    code/scripts/test_adaptive_checkpoint_rng.py
    code/scripts/test_group_balanced_sampler.py
    code/scripts/test_visual_episode_sequence.py
    code/scripts/train_adaptive_gaussian_wm.py
    code/scripts/train_visual_sequence_core_2n8g.sh
    code/scripts/train_visual_sequence_core_4gpu.sh
    code/scripts/train_visual_sequence_goal_prior.py
    code/scripts/train_visual_sequence_goal_prior_2n8g.sh
    code/scripts/train_visual_sequence_goal_prior_4gpu.sh
    code/scripts/verify_rt2_visual_episode_cache.py
)
for relative in "${files[@]}"; do
    cp "${STAGING_ROOT}/${relative}" "${ROOT}/${relative}"
done

exec env \
    ROOT="${ROOT}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    OUT="${DATA}" \
    LOG_ROOT="${LOG_ROOT}" \
    GPU_IDS=0,1,2,3 \
    JOBS_PER_GPU=1 \
    FRAME_BATCH=52 \
    CPU_THREADS=16 \
    JPEG_WORKERS=4 \
    WINDOW_LENGTHS=25,50,75,100 \
    SAMPLE_STRIDE=1 \
    LIMIT=0 \
    OVERWRITE=0 \
    bash "${ROOT}/code/scripts/cache_rt2_visual_episodes_4gpu.sh"
