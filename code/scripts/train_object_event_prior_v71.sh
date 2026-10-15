#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUN_NAME="${RUN_NAME:-object_event_prior_v71_seed17}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-32}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export ENCODER_FRAME_BATCH="${ENCODER_FRAME_BATCH:-64}"
exec bash "${SOURCE_ROOT}/code/scripts/train_object_event_world_model_v71.sh" \
  --stage prior --base-checkpoint "${BASE_CHECKPOINT:-}" \
  --text-cache "${TEXT_CACHE:-${WORK_ROOT}/metadata/object_event_text_v71}" \
  --language-dropout "${LANGUAGE_DROPOUT:-0.1}" "$@"
