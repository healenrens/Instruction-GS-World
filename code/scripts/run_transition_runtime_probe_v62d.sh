#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
PY="${VENV_ROOT}/.venv/bin/python"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
RUN_ID="${RUN_ID:-v62d_e1_runtime_$(date +%Y%m%d_%H%M%S)_$$}"
OUT_ROOT="${OUT_ROOT:-${RUNTIME_ROOT}/outputs/v62_parallel/d_e1_runtime/${RUN_ID}}"
REPORT="${REPORT:-${OUT_ROOT}/transition_runtime_ddp.json}"
E0_CHECKPOINT="${E0_CHECKPOINT:-${RUNTIME_ROOT}/outputs/object_transition_v62_e0_seed17_f00082d/v62_e0_0002500.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$(${PY} -c 'import torch; print(torch.cuda.device_count())')"
fi

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

echo "[v62d] run_id=${RUN_ID} world=${NPROC_PER_NODE} checkpoint=${E0_CHECKPOINT} report=${REPORT}"
cd "${ROOT}"
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/verify_transition_runtime_ddp_v62d.py" \
  --codec_checkpoint "${E0_CHECKPOINT}" \
  --data_index "${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}" \
  --dino_checkpoint "${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${REPORT}" \
  --held_group_stride "${HELD_GROUP_STRIDE:-20}" \
  --batch "${BATCH_PER_RANK:-4}" \
  --steps "${STEPS:-3}" \
  --lr "${LR:-2e-4}" \
  --seed "${SEED:-17}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-32}" \
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-32}" \
  --amp "${AMP:-bf16}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_ID}}" \
  --wandb_group "${WANDB_GROUP:-object-transition-v62d-ddp-runtime}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
