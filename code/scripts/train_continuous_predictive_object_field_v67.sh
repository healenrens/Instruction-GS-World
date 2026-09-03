#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
STAGE="${STAGE:-predictive_state}"
if [ -f "${ROOT}/SOURCE_REVISION" ]; then
  read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"
else
  DEPLOYED_REVISION=local-unversioned
fi
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP_CHECKPOINT="${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
RUN_NAME="${RUN_NAME:-continuous_predictive_object_field_v67_${STAGE}_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$(${PY} -c 'import torch; print(torch.cuda.device_count())')"
fi
MIN_GPU_MEMORY_MIB="$(${PY} -c 'import torch; print(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) // 2**20)')"
BATCH_PER_GPU="${BATCH_PER_GPU:-auto}"
if [ "${BATCH_PER_GPU}" = auto ]; then
  if [ "${MIN_GPU_MEMORY_MIB}" -ge 76000 ]; then BATCH_PER_GPU=16
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 45000 ]; then BATCH_PER_GPU=8
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 22000 ]; then BATCH_PER_GPU=4
  else BATCH_PER_GPU=2
  fi
fi
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
LOCAL_BATCH=$((BATCH_PER_GPU * NPROC_PER_NODE))
GRAD_ACCUM="${GRAD_ACCUM:-$((TARGET_GLOBAL_BATCH / LOCAL_BATCH))}"
STEPS="${STEPS:-30000}"
WARMUP_STEPS="${WARMUP_STEPS:-$((STEPS / 20))}"

ARGS=(
  --stage "${STAGE}"
  --data_index "${DATA_INDEX}"
  --out "${OUT}"
  --source_revision "${SOURCE_REVISION}"
  --dino_checkpoint "${DINO_CHECKPOINT}"
  --siglip_checkpoint "${SIGLIP_CHECKPOINT}"
  --tracker_checkpoint "${TRACKER_CHECKPOINT}"
  --chunk_lengths 8
  --temporal_step_ms 100,200,400
  --held_group_stride "${HELD_GROUP_STRIDE:-20}"
  --batch "${BATCH_PER_GPU}"
  --grad_accum "${GRAD_ACCUM}"
  --target_global_batch "${TARGET_GLOBAL_BATCH}"
  --workers "${WORKERS_PER_RANK:-4}"
  --prefetch_factor "${PREFETCH_FACTOR:-2}"
  --dino_frame_batch "${DINO_FRAME_BATCH:-192}"
  --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-192}"
  --steps "${STEPS}"
  --lr "${LR:-2e-4}"
  --lr_floor_ratio "${LR_FLOOR_RATIO:-0.1}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}"
  --warmup_steps "${WARMUP_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM:-5.0}"
  --save_every "${SAVE_EVERY:-2500}"
  --recovery_every "${RECOVERY_EVERY:-250}"
  --log_every "${LOG_EVERY:-20}"
  --seed "${SEED:-17}"
  --amp "${AMP:-bf16}"
  --wandb_mode "${WANDB_MODE:-online}"
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}"
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}"
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}"
  --wandb_group "${WANDB_GROUP:-continuous-predictive-object-field-v67}"
  --wandb_tags "${WANDB_TAGS:-v67,${STAGE},continuous-field,predictive-rate-distortion}"
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
)
if [ "${STAGE}" = posterior_dynamics ]; then
  ARGS+=(--state_checkpoint "${STATE_CHECKPOINT:-}")
fi
if [ -n "${RESUME:-}" ]; then ARGS+=(--resume "${RESUME}"); fi
if [ -n "${RESUME_COMPATIBLE_SOURCE_REVISION:-}" ]; then
  ARGS+=(--resume_compatible_source_revision "${RESUME_COMPATIBLE_SOURCE_REVISION}")
fi
if [ -n "${WANDB_RUN_ID:-}" ]; then ARGS+=(--wandb_run_id "${WANDB_RUN_ID}"); fi
if [ -n "${MAX_TRAIN_ITEMS:-}" ]; then ARGS+=(--max_train_items "${MAX_TRAIN_ITEMS}"); fi

echo "[continuous-object-field-v67] stage=${STAGE} revision=${SOURCE_REVISION}"
echo "[continuous-object-field-v67] world=${NPROC_PER_NODE} batch_per_gpu=${BATCH_PER_GPU} grad_accum=${GRAD_ACCUM} effective_batch=${TARGET_GLOBAL_BATCH}"
echo "[continuous-object-field-v67] out=${OUT} dino_frame_batch=${DINO_FRAME_BATCH:-192} siglip_frame_batch=${SIGLIP_FRAME_BATCH:-192}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_continuous_predictive_object_field_v67.py" "${ARGS[@]}"
