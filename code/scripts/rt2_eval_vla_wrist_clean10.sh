#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
CKPT=${CKPT:-"$ROOT/checkpoints/vla_causal_v1_from20k_2n8g/vla_050000.pt"}
PORT=${PORT:-19010}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-"$ROOT/eval/robotwin2/causal_vla_050000_demo_clean_10_${STAMP}"}
SERVER="http://127.0.0.1:${PORT}"
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/mnt/pfs/xuhaoming/xr-2/.venv/bin/python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/mnt/pfs/xuhaoming/xr-2/RoboTwin}
ROBOTWIN_PLANNER_BACKEND=${ROBOTWIN_PLANNER_BACKEND:-curobo}

health() {
    "$ROOT/.venv/bin/python" -c \
        'import sys, urllib.request; sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1], timeout=5).read())' \
        "$SERVER/health"
}

mkdir -p "$OUT"
test -f "$CKPT"
test -x "$ROOT/.venv/bin/python"
test -x "$ROBOTWIN_PYTHON"
test -d "$ROBOTWIN_ROOT/envs"
test "$ROBOTWIN_PLANNER_BACKEND" = curobo

if health >/dev/null 2>&1; then
    echo "ERROR: port ${PORT} already has a policy server" >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$ROOT/.venv/bin/python" \
    "$ROOT/code/scripts/rt2_policy_server.py" \
    --ckpt "$CKPT" --port "$PORT" --wrist 1 --placement entropy \
    >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

attempt=0
while ! health >"$OUT/health.json" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: policy server exited during startup" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 300 ]; then
        echo "ERROR: policy server did not become healthy in 600 seconds" >&2
        exit 1
    fi
    sleep 2
done

"$ROOT/.venv/bin/python" "$ROOT/code/scripts/rt2_eval_batch.py" \
    --out "$OUT" --server "$SERVER" --checkpoint "$CKPT" \
    --cfg demo_clean --episodes 10 --expected_tasks 50 --start_seed 100000 \
    --instruction_type seen --python "$ROBOTWIN_PYTHON" \
    --robotwin-root "$ROBOTWIN_ROOT" --planner-backend "$ROBOTWIN_PLANNER_BACKEND" \
    >"$OUT/batch.log" 2>&1

echo "[launcher] complete: $OUT"
