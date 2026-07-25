#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
cd "$ROOT"

required_env=(WORLD_SIZE RANK NPROC_PER_NODE SESSION)
for name in "${required_env[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[prepare] missing required environment variable: $name" >&2
        exit 2
    fi
done

NNODES="$WORLD_SIZE"
NODE_RANK="$RANK"
EXPECTED_NNODES="${EXPECTED_NNODES:-2}"
EXPECTED_NPROC_PER_NODE="${EXPECTED_NPROC_PER_NODE:-8}"
if [[ "$NNODES" -ne "$EXPECTED_NNODES" ||
      "$NPROC_PER_NODE" -ne "$EXPECTED_NPROC_PER_NODE" ]]; then
    echo "[prepare] expected ${EXPECTED_NNODES}x${EXPECTED_NPROC_PER_NODE}, got ${NNODES}x${NPROC_PER_NODE}" >&2
    exit 2
fi
if [[ "$NODE_RANK" -lt 0 || "$NODE_RANK" -ge "$NNODES" ]]; then
    echo "[prepare] invalid node rank: $NODE_RANK for $NNODES nodes" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export RT2_PAIR_CPU_THREADS="${RT2_PAIR_CPU_THREADS:-2}"

SOURCE="${SOURCE:-$ROOT/data/rt2_joint_src}"
TRACKED="${TRACKED:-$ROOT/data/rt2_joint}"
PAIRS_OUT="${PAIRS_OUT:-$ROOT/data/rt2_causal_pairs_full_20260718_v1}"
DINO_OUT="${DINO_OUT:-$ROOT/data/rt2_causal_pairs_full_20260718_v1_dino32}"
AUDIT_ROOT="${AUDIT_ROOT:-$ROOT/outputs/rt2_causal_pairs_full_20260718_v1}"
LOG_ROOT="${LOG_ROOT:-$AUDIT_ROOT/logs/$SESSION}"
SYNC_ROOT="${SYNC_ROOT:-$AUDIT_ROOT/sync/$SESSION}"

WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
PAIRS_PER_CLIP="${PAIRS_PER_CLIP:-6}"
HORIZONS="${HORIZONS:-1,2,3,4,6,8,10,12}"
SPLIT_LIMITS="${SPLIT_LIMITS:-}"
SYNC_TIMEOUT_MINUTES="${SYNC_TIMEOUT_MINUTES:-1440}"
LOCAL_WORKERS=$((NPROC_PER_NODE * WORKERS_PER_GPU))
GLOBAL_WORKERS=$((NNODES * LOCAL_WORKERS))

for path in "$SOURCE" "$TRACKED"; do
    if [[ ! -d "$path" ]]; then
        echo "[prepare] required source directory not found: $path" >&2
        exit 2
    fi
done
mkdir -p "$PAIRS_OUT" "$DINO_OUT" "$AUDIT_ROOT" "$LOG_ROOT" "$SYNC_ROOT"

write_marker() {
    local marker="$1"
    local temporary="${marker}.tmp.$$"
    printf '%s\n' "$(date -Is)" > "$temporary"
    mv "$temporary" "$marker"
}

wait_for_nodes() {
    local phase="$1"
    local deadline=$((SECONDS + SYNC_TIMEOUT_MINUTES * 60))
    while true; do
        if compgen -G "$SYNC_ROOT/${phase}_node_*.failed" > /dev/null; then
            echo "[prepare] $phase failed on at least one node" >&2
            return 1
        fi
        local complete=0
        local rank
        for ((rank = 0; rank < NNODES; rank++)); do
            if [[ -f "$SYNC_ROOT/${phase}_node_${rank}.done" ]]; then
                complete=$((complete + 1))
            fi
        done
        if [[ "$complete" -eq "$NNODES" ]]; then
            return 0
        fi
        if [[ "$SECONDS" -ge "$deadline" ]]; then
            echo "[prepare] timed out waiting for $phase nodes" >&2
            return 1
        fi
        sleep 15
    done
}

wait_for_audit() {
    local phase="$1"
    local deadline=$((SECONDS + SYNC_TIMEOUT_MINUTES * 60))
    while [[ ! -f "$SYNC_ROOT/${phase}_audit.done" ]]; do
        if [[ -f "$SYNC_ROOT/${phase}_audit.failed" ]]; then
            echo "[prepare] $phase audit failed" >&2
            return 1
        fi
        if [[ "$SECONDS" -ge "$deadline" ]]; then
            echo "[prepare] timed out waiting for $phase audit" >&2
            return 1
        fi
        sleep 15
    done
}

run_pair_workers() {
    local pids=()
    local worker
    for ((worker = 0; worker < LOCAL_WORKERS; worker++)); do
        local gpu=$((worker % NPROC_PER_NODE))
        local shard=$((NODE_RANK * LOCAL_WORKERS + worker))
        local cmd=(
            .venv/bin/python code/scripts/build_rt2_causal_pairs.py
            --source "$SOURCE"
            --tracked "$TRACKED"
            --out "$PAIRS_OUT"
            --horizons "$HORIZONS"
            --pairs_per_clip "$PAIRS_PER_CLIP"
            --shard "$shard"
            --nshard "$GLOBAL_WORKERS"
        )
        if [[ -n "$SPLIT_LIMITS" ]]; then
            cmd+=(--split_limits "$SPLIT_LIMITS")
        fi
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            printf '[prepare] pair shard=%s gpu=%s command:' "$shard" "$gpu"
            printf ' %q' env "CUDA_VISIBLE_DEVICES=$gpu" "${cmd[@]}"
            printf '\n'
        else
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" \
                > "$LOG_ROOT/pairs_shard_${shard}.log" 2>&1 &
            pids+=("$!")
        fi
    done
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        return 0
    fi
    local status=0
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}

run_dino_workers() {
    local pids=()
    local worker
    for ((worker = 0; worker < LOCAL_WORKERS; worker++)); do
        local gpu=$((worker % NPROC_PER_NODE))
        local shard=$((NODE_RANK * LOCAL_WORKERS + worker))
        local cmd=(
            .venv/bin/python code/scripts/cache_rt2_pair_dino.py
            --data "$PAIRS_OUT"
            --out "$DINO_OUT"
            --feature_dim 32
            --shard "$shard"
            --nshard "$GLOBAL_WORKERS"
        )
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            printf '[prepare] dino shard=%s gpu=%s command:' "$shard" "$gpu"
            printf ' %q' env "CUDA_VISIBLE_DEVICES=$gpu" "${cmd[@]}"
            printf '\n'
        else
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" \
                > "$LOG_ROOT/dino_shard_${shard}.log" 2>&1 &
            pids+=("$!")
        fi
    done
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        return 0
    fi
    local status=0
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}

echo "[prepare] node=$NODE_RANK/$NNODES local_workers=$LOCAL_WORKERS global_workers=$GLOBAL_WORKERS"
echo "[prepare] pairs=$PAIRS_OUT dino=$DINO_OUT audit=$AUDIT_ROOT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    run_pair_workers
    run_dino_workers
    exit 0
fi

rm -f "$SYNC_ROOT/pairs_node_${NODE_RANK}.done" "$SYNC_ROOT/pairs_node_${NODE_RANK}.failed"
if run_pair_workers; then
    write_marker "$SYNC_ROOT/pairs_node_${NODE_RANK}.done"
else
    write_marker "$SYNC_ROOT/pairs_node_${NODE_RANK}.failed"
    exit 1
fi
wait_for_nodes pairs

if [[ "$NODE_RANK" -eq 0 ]]; then
    rm -f "$SYNC_ROOT/pairs_audit.done" "$SYNC_ROOT/pairs_audit.failed"
    if .venv/bin/python code/scripts/verify_rt2_causal_pairs.py \
        --data "$PAIRS_OUT" \
        --source "$SOURCE" \
        --report "$AUDIT_ROOT/verification.json" \
        > "$LOG_ROOT/pairs_audit.log" 2>&1; then
        write_marker "$SYNC_ROOT/pairs_audit.done"
    else
        write_marker "$SYNC_ROOT/pairs_audit.failed"
        exit 1
    fi
fi
wait_for_audit pairs

rm -f "$SYNC_ROOT/dino_node_${NODE_RANK}.done" "$SYNC_ROOT/dino_node_${NODE_RANK}.failed"
if run_dino_workers; then
    write_marker "$SYNC_ROOT/dino_node_${NODE_RANK}.done"
else
    write_marker "$SYNC_ROOT/dino_node_${NODE_RANK}.failed"
    exit 1
fi
wait_for_nodes dino

if [[ "$NODE_RANK" -eq 0 ]]; then
    rm -f "$SYNC_ROOT/dino_audit.done" "$SYNC_ROOT/dino_audit.failed"
    if .venv/bin/python code/scripts/verify_rt2_pair_dino.py \
        --pairs "$PAIRS_OUT" \
        --dino "$DINO_OUT" \
        --report "$AUDIT_ROOT/dino_verification.json" \
        > "$LOG_ROOT/dino_audit.log" 2>&1; then
        write_marker "$SYNC_ROOT/dino_audit.done"
    else
        write_marker "$SYNC_ROOT/dino_audit.failed"
        exit 1
    fi
fi
wait_for_audit dino
echo "[prepare] DONE node=$NODE_RANK pair_audit=$AUDIT_ROOT/verification.json dino_audit=$AUDIT_ROOT/dino_verification.json"
