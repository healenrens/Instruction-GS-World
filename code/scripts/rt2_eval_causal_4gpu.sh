#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

CKPT=${CKPT:-"$ROOT/checkpoints/vla_causal_v1_from20k_2n8g/vla_050000.pt"}
CFG=${CFG:-demo_clean}
EPISODES=${EPISODES:-5}
TASKS=${TASKS:-}
EXPECTED_TASKS=${EXPECTED_TASKS:-50}
START_SEED=${START_SEED:-100000}
SEED_TRIES=${SEED_TRIES:-100}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-seen}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/mnt/pfs/xuhaoming/xr-2/.venv/bin/python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/mnt/pfs/xuhaoming/xr-2/RoboTwin}
ROBOTWIN_PLANNER_BACKEND=${ROBOTWIN_PLANNER_BACKEND:-curobo}
GPUS=${GPUS:-0,1,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
BASE_PORT=${BASE_PORT:-19010}
MAX_INITIAL_GPU_MEMORY_MIB=${MAX_INITIAL_GPU_MEMORY_MIB:-1024}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

CKPT_NAME=$(basename "$CKPT" .pt)
OUT=${OUT:-"$ROOT/eval/robotwin2/causal_${CKPT_NAME}_${CFG}_${EPISODES}_${STAMP}"}

test -f "$CKPT"
test -x "$ROOT/.venv/bin/python"
test -x "$ROBOTWIN_PYTHON"
test -d "$ROBOTWIN_ROOT/envs"
test "$ROBOTWIN_PLANNER_BACKEND" = curobo
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
mkdir -p "$OUT"

set -- "$ROOT/.venv/bin/python" "$ROOT/code/scripts/rt2_eval_parallel.py" \
    --out "$OUT" --checkpoint "$CKPT" --cfg "$CFG" --episodes "$EPISODES" \
    --expected-tasks "$EXPECTED_TASKS" --start-seed "$START_SEED" \
    --seed-tries "$SEED_TRIES" --instruction-type "$INSTRUCTION_TYPE" \
    --python "$ROBOTWIN_PYTHON" --robotwin-root "$ROBOTWIN_ROOT" \
    --planner-backend "$ROBOTWIN_PLANNER_BACKEND" --gpus "$GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" --base-port "$BASE_PORT" \
    --max-initial-gpu-memory-mib "$MAX_INITIAL_GPU_MEMORY_MIB"

if [ -n "$TASKS" ]; then
    set -- "$@" --tasks "$TASKS"
fi
if [ "$PREFLIGHT_ONLY" = "1" ]; then
    set -- "$@" --preflight-only
fi

"$@"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    printf '%s\n' "[eval] preflight complete: $OUT/preflight.json"
    exit 0
fi

VERIFY_EXPECTED_TASKS=$EXPECTED_TASKS
if [ -n "$TASKS" ]; then
    VERIFY_EXPECTED_TASKS=$(printf '%s\n' "$TASKS" | awk -F, '{print NF}')
fi

"$ROOT/.venv/bin/python" "$ROOT/code/scripts/rt2_eval_verify.py" \
    --out "$OUT" --checkpoint "$CKPT" --expected-tasks "$VERIFY_EXPECTED_TASKS" \
    --episodes "$EPISODES" --require-complete --report "$OUT/verification.json"

"$ROOT/.venv/bin/python" "$ROOT/code/scripts/rt2_eval_report.py" \
    --out "$OUT" --require-complete

printf '%s\n' "[eval] complete: $OUT"
