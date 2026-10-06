#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
export VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
export SOURCE_REVISION="$(cat "${ROOT}/SOURCE_REVISION")"
export NNODES="${NNODES:-${WORLD_SIZE:-4}}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Baige supplies WORLD_SIZE (node count), RANK (node index), MASTER_ADDR/PORT.
# The launcher also accepts explicit NNODES/NODE_RANK from other schedulers.
# The default name is shared across nodes; do not derive it from local clocks.
export RUN_NAME="${RUN_NAME:-language_object_effect_v70_flow_32gpu_from500_${SOURCE_REVISION:0:7}}"
export OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-4}" GRAD_ACCUM="${GRAD_ACCUM:-2}"
export STEPS="${STEPS:-40000}" WORKERS_PER_RANK="${WORKERS_PER_RANK:-4}"
export DINO_FRAME_BATCH="${DINO_FRAME_BATCH:-8}"
export LR_TEXT="${LR_TEXT:-1e-5}" LR_EXPERT="${LR_EXPERT:-2e-5}"
export WARMUP_FRACTION="${WARMUP_FRACTION:-0.05}" MODE=flow
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-instruct-gs-world}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
if [ -n "${RESUME:-}" ]; then
  unset INIT_FROM
else
  export INIT_FROM="${INIT_FROM-${RUNTIME_ROOT}/outputs/language_object_effect_v70_flow_teacher8750_seed17_20261006_042956/step_0000500}"
fi
exec "${ROOT}/code/scripts/run_language_object_effect_v70.sh" train
