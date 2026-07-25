#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
CKPT=${CKPT:-"$ROOT/checkpoints/vla_causal_v1_from20k_2n8g/vla_050000.pt"}
EPISODES=${EPISODES:-5}
TASKS=${TASKS:-}
EXPECTED_TASKS=${EXPECTED_TASKS:-50}
START_SEED=${START_SEED:-100000}
SEED_TRIES=${SEED_TRIES:-100}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-seen}
CLEAN_GPUS=${CLEAN_GPUS:-0,1}
RANDOMIZED_GPUS=${RANDOMIZED_GPUS:-2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
CLEAN_BASE_PORT=${CLEAN_BASE_PORT:-19010}
RANDOMIZED_BASE_PORT=${RANDOMIZED_BASE_PORT:-19020}
MAX_INITIAL_GPU_MEMORY_MIB=${MAX_INITIAL_GPU_MEMORY_MIB:-1024}
WAIT_FOR_GPUS_SECONDS=${WAIT_FOR_GPUS_SECONDS:-0}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-30}
POLL_SECONDS=${POLL_SECONDS:-30}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/mnt/pfs/xuhaoming/xr-2/RoboTwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/mnt/pfs/xuhaoming/xr-2/.venv/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

CKPT_NAME=$(basename "$CKPT" .pt)
OUT=${OUT:-"$ROOT/eval/robotwin2/dual_${CKPT_NAME}_clean_randomized_${EPISODES}_${STAMP}"}

test -f "$CKPT"
test -x "$ROOT/.venv/bin/python"
test -x "$ROBOTWIN_PYTHON"
test -d "$ROBOTWIN_ROOT/task_config"
mkdir -p "$OUT"

set -- "$ROOT/.venv/bin/python" "$ROOT/code/scripts/rt2_eval_dual_config.py" \
    --out "$OUT" --checkpoint "$CKPT" --episodes "$EPISODES" \
    --expected-tasks "$EXPECTED_TASKS" --start-seed "$START_SEED" \
    --seed-tries "$SEED_TRIES" --instruction-type "$INSTRUCTION_TYPE" \
    --clean-gpus "$CLEAN_GPUS" --randomized-gpus "$RANDOMIZED_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --clean-base-port "$CLEAN_BASE_PORT" --randomized-base-port "$RANDOMIZED_BASE_PORT" \
    --robotwin-root "$ROBOTWIN_ROOT" --robotwin-python "$ROBOTWIN_PYTHON" \
    --max-initial-gpu-memory-mib "$MAX_INITIAL_GPU_MEMORY_MIB" \
    --wait-for-gpus-seconds "$WAIT_FOR_GPUS_SECONDS" \
    --gpu-poll-seconds "$GPU_POLL_SECONDS" --poll-seconds "$POLL_SECONDS"

if [ -n "$TASKS" ]; then
    set -- "$@" --tasks "$TASKS"
fi
if [ "$PREFLIGHT_ONLY" = "1" ]; then
    set -- "$@" --preflight-only
fi

"$@"
