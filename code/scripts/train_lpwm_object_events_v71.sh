#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${WORK_ROOT}/environments/lpwm-v71/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
RUN_NAME="${RUN_NAME:-lpwm_cliport_ordered_v71_seed17}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="${WORK_ROOT}/cache/lpwm/huggingface" TORCH_HOME="${WORK_ROOT}/cache/lpwm/torch"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/lpwm" TMPDIR="${WORK_ROOT}/tmp/lpwm-v71"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/lpwm-pycache"
mkdir -p "${TMPDIR}"
launch=(--nproc_per_node "${NPROC_PER_NODE}")
if [ "${NNODES}" -eq 1 ]; then
  launch+=(--standalone)
else
  launch+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}"
           --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}")
fi
args=(--assets "${ASSETS:-${WORK_ROOT}/metadata/lpwm_assets_v71.json}"
      --manifest "${MANIFEST:-${WORK_ROOT}/metadata/cliport_event_v71.json}"
      --text-cache "${TEXT_CACHE:-${WORK_ROOT}/metadata/lpwm_cliport_text_v71}"
      --out "${OUT:-${WORK_ROOT}/outputs/${RUN_NAME}}"
      --batch "${BATCH_PER_GPU:-3}" --accum "${GRAD_ACCUM:-1}"
      --workers "${WORKERS_PER_RANK:-2}" --epochs "${EPOCHS:-150}"
      --source-revision "${SOURCE_REVISION:-}")
if [ -n "${STEPS:-}" ]; then args+=(--steps "${STEPS}"); fi
if [ -n "${RESUME:-}" ]; then args+=(--resume "${RESUME}"); fi
if [ -n "${SWANLAB_PROJECT:-}" ]; then args+=(--swanlab-project "${SWANLAB_PROJECT}"); fi
if [ -n "${SWANLAB_WORKSPACE:-}" ]; then args+=(--swanlab-workspace "${SWANLAB_WORKSPACE}"); fi
exec "${PYTHON}" -m torch.distributed.run "${launch[@]}" \
  "${ROOT}/code/scripts/train_lpwm_object_events_v71.py" "${args[@]}" "$@"
