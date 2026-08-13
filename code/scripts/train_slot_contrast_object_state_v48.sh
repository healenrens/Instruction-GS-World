#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/slot_contrast_object_state_v48_seed17}"
GATE_REPORT="${GATE_REPORT:-}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"

if [ -n "${INIT_FROM:-}" ]; then
  echo "[slot-contrast-v48] INIT_FROM is forbidden; v48 trains from scratch"
  exit 2
fi
if [ -z "${GATE_REPORT}" ] || [ ! -f "${GATE_REPORT}" ]; then
  echo "[slot-contrast-v48] GATE_REPORT is missing: ${GATE_REPORT}"
  exit 2
fi
if [ -z "${SOURCE_REVISION}" ]; then
  echo "[slot-contrast-v48] SOURCE_REVISION is missing"
  exit 2
fi
if [ ! -f "${DINO_CHECKPOINT}" ]; then
  echo "[slot-contrast-v48] local DINO checkpoint is missing: ${DINO_CHECKPOINT}"
  exit 2
fi
if [ ! -x "${PY}" ] || [ ! -x "${TORCHRUN}" ]; then
  echo "[slot-contrast-v48] runtime is missing under ${VENV_ROOT}/.venv"
  exit 2
fi
if [ ! -f "${DATA}/episode_manifest.json" ]; then
  echo "[slot-contrast-v48] RGB manifest is missing under ${DATA}"
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "[slot-contrast-v48] no visible CUDA devices"
  exit 2
fi

TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
BATCH_PER_GPU="${BATCH_PER_GPU:-auto}"
MIN_GPU_MEMORY_MIB="$("${PY}" -c 'import torch; print(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) // 2**20)')"
if [ "${BATCH_PER_GPU}" = auto ]; then
  if [ "${MIN_GPU_MEMORY_MIB}" -ge 76000 ]; then MAX_BATCH_PER_GPU=32
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 45000 ]; then MAX_BATCH_PER_GPU=16
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 22000 ]; then MAX_BATCH_PER_GPU=8
  else MAX_BATCH_PER_GPU=4
  fi
  BATCH_PER_GPU=0
  for CANDIDATE in 32 16 8 4 2 1; do
    LOCAL_CANDIDATE=$((CANDIDATE * NPROC_PER_NODE))
    if [ "${CANDIDATE}" -le "${MAX_BATCH_PER_GPU}" ] && \
       [ "${LOCAL_CANDIDATE}" -le "${TARGET_GLOBAL_BATCH}" ] && \
       [ $((TARGET_GLOBAL_BATCH % LOCAL_CANDIDATE)) -eq 0 ]; then
      BATCH_PER_GPU="${CANDIDATE}"
      break
    fi
  done
fi

DINO_FRAME_BATCH="${DINO_FRAME_BATCH:-auto}"
if [ "${DINO_FRAME_BATCH}" = auto ]; then
  if [ "${MIN_GPU_MEMORY_MIB}" -ge 76000 ]; then DINO_FRAME_BATCH=128
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 45000 ]; then DINO_FRAME_BATCH=64
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 22000 ]; then DINO_FRAME_BATCH=32
  else DINO_FRAME_BATCH=16
  fi
fi

WORKERS_PER_RANK="${WORKERS_PER_RANK:-auto}"
if [ "${WORKERS_PER_RANK}" = auto ]; then
  WORKERS_PER_RANK="$("${PY}" -c "import os; print(max(2, min(8, (os.cpu_count() or 8) // (2 * ${NPROC_PER_NODE}))))")"
fi
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
LOCAL_BATCH=$((BATCH_PER_GPU * NPROC_PER_NODE))
if [ "${GRAD_ACCUM}" = auto ]; then
  if [ $((TARGET_GLOBAL_BATCH % LOCAL_BATCH)) -ne 0 ]; then
    echo "[slot-contrast-v48] target batch is not divisible by local batch"
    exit 2
  fi
  GRAD_ACCUM=$((TARGET_GLOBAL_BATCH / LOCAL_BATCH))
fi
if [ $((BATCH_PER_GPU * NPROC_PER_NODE * GRAD_ACCUM)) -ne "${TARGET_GLOBAL_BATCH}" ]; then
  echo "[slot-contrast-v48] effective batch contract differs"
  exit 2
fi

mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
ARGS=(
  --data "${DATA}" --out "${OUT}" --gate_report "${GATE_REPORT}"
  --source_revision "${SOURCE_REVISION}" --dino_checkpoint "${DINO_CHECKPOINT}"
  --chunk_lengths "${CHUNK_LENGTHS:-8,16,24,32}"
  --temporal_strides "${TEMPORAL_STRIDES:-1,2,3,4}"
  --observation_mask_probability "${OBSERVATION_MASK_PROBABILITY:-0.20}"
  --batch "${BATCH_PER_GPU}" --grad_accum "${GRAD_ACCUM}"
  --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
  --prefetch_factor "${PREFETCH_FACTOR:-2}" --dino_frame_batch "${DINO_FRAME_BATCH}"
  --steps "${STEPS:-50000}" --lr "${LR:-2e-4}" --lr_floor "${LR_FLOOR:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}" --warmup_steps "${WARMUP_STEPS:-2500}"
  --max_grad_norm "${MAX_GRAD_NORM:-5.0}" --save_every "${SAVE_EVERY:-2500}"
  --recovery_every "${RECOVERY_EVERY:-250}" --log_every "${LOG_EVERY:-20}"
  --seed "${SEED:-17}" --amp "${AMP:-bf16}"
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}"
  --wandb_entity "${WANDB_ENTITY:-}" --wandb_name "${WANDB_NAME:-${RUN_NAME:-slot_contrast_object_state_v48_seed17}}"
  --wandb_group "${WANDB_GROUP:-slot-contrast-object-state-v48}"
  --wandb_tags "${WANDB_TAGS:-v48,pure-video,slot-attention,slot-contrast}"
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
)
if [ -n "${RESUME:-}" ]; then ARGS+=(--resume "${RESUME}"); fi
if [ -n "${WANDB_RUN_ID:-}" ]; then ARGS+=(--wandb_run_id "${WANDB_RUN_ID}"); fi
if [ -n "${MAX_TRAIN_ITEMS:-}" ]; then ARGS+=(--max_train_items "${MAX_TRAIN_ITEMS}"); fi

echo "[slot-contrast-v48] root=${ROOT} data=${DATA} out=${OUT}"
echo "[slot-contrast-v48] world=${NPROC_PER_NODE} gpu_memory_mib=${MIN_GPU_MEMORY_MIB} batch_per_gpu=${BATCH_PER_GPU} grad_accum=${GRAD_ACCUM} effective_batch=${TARGET_GLOBAL_BATCH}"
echo "[slot-contrast-v48] dino_frame_batch=${DINO_FRAME_BATCH} workers_per_rank=${WORKERS_PER_RANK}"
echo "[slot-contrast-v48] source_revision=${SOURCE_REVISION} dino_checkpoint=${DINO_CHECKPOINT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "${ROOT}" || exit 2
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_slot_contrast_object_state_v48.py" "${ARGS[@]}"
