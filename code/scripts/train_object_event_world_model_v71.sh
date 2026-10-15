#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${WORK_ROOT}/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
RUN_NAME="${RUN_NAME:-object_event_v71_object_step_seed17}"
OUT="${OUT:-${WORK_ROOT}/outputs/${RUN_NAME}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="${WORK_ROOT}/cache/huggingface" TORCH_HOME="${WORK_ROOT}/cache/torch"
export XDG_CACHE_HOME="${WORK_ROOT}/cache" TMPDIR="${WORK_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/pycache"
mkdir -p "${TMPDIR}" "${WORK_ROOT}/logs"

launch=(--nproc_per_node "${NPROC_PER_NODE}")
if [ "${NNODES}" -eq 1 ]; then
  launch+=(--standalone)
else
  launch+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}"
           --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}")
fi
args=(--manifest "${MANIFEST:-${WORK_ROOT}/metadata/object_event_v71.json}"
      --assets "${ASSETS:-${WORK_ROOT}/metadata/rae_assets_512.json}"
      --out "${OUT}" --effect-mode "${EFFECT_MODE:-object_step}"
      --steps "${STEPS:-10000}" --batch "${BATCH_PER_GPU:-4}" --accum "${GRAD_ACCUM:-8}"
      --workers "${WORKERS_PER_RANK:-2}" --frame-batch "${ENCODER_FRAME_BATCH:-4}"
      --decode-batch "${DECODER_FRAME_BATCH:-2}" --source-revision "${SOURCE_REVISION:-}")
if [ -n "${RESUME:-}" ]; then args+=(--resume "${RESUME}"); fi
if [ -n "${INIT_FROM:-}" ]; then args+=(--init-from "${INIT_FROM}"); fi
if [ -n "${SWANLAB_PROJECT:-}" ]; then args+=(--swanlab-project "${SWANLAB_PROJECT}"); fi
if [ -n "${SWANLAB_WORKSPACE:-}" ]; then args+=(--swanlab-workspace "${SWANLAB_WORKSPACE}"); fi
exec "${PYTHON}" -m torch.distributed.run "${launch[@]}" \
  "${SOURCE_ROOT}/code/scripts/train_object_event_world_model_v71.py" "${args[@]}" "$@"
