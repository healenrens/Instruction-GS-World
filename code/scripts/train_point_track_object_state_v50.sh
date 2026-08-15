#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
STAGE="${STAGE:-object_state}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
RUN_NAME="${RUN_NAME:-point_track_object_state_v50_${STAGE}_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
GATE_REPORT="${GATE_REPORT:-}"
STATE_GATE_REPORT="${STATE_GATE_REPORT:-}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"

if [ "${STAGE}" != object_state ] && [ "${STAGE}" != latent_effect ]; then
  echo "[point-track-v50] STAGE must be object_state or latent_effect"
  exit 2
fi
if [ -z "${GATE_REPORT}" ] || [ ! -f "${GATE_REPORT}" ]; then
  echo "[point-track-v50] GATE_REPORT is missing: ${GATE_REPORT}"
  exit 2
fi
if [ -z "${SOURCE_REVISION}" ]; then
  echo "[point-track-v50] SOURCE_REVISION is missing"
  exit 2
fi
if [ "${STAGE}" = object_state ] && [ -n "${INIT_FROM:-}" ]; then
  echo "[point-track-v50] Object State trains from scratch; INIT_FROM is forbidden"
  exit 2
fi
if [ "${STAGE}" = latent_effect ] && [ -z "${RESUME:-}" ]; then
  if [ -z "${INIT_FROM:-}" ] || [ ! -f "${INIT_FROM}" ]; then
    echo "[point-track-v50] latent_effect requires an Object State INIT_FROM"
    exit 2
  fi
  if [ -z "${STATE_GATE_REPORT}" ] || [ ! -f "${STATE_GATE_REPORT}" ]; then
    echo "[point-track-v50] latent_effect requires STATE_GATE_REPORT"
    exit 2
  fi
fi
if [ ! -f "${DINO_CHECKPOINT}" ]; then
  echo "[point-track-v50] frozen DINO checkpoint is missing: ${DINO_CHECKPOINT}"
  exit 2
fi
if [ "${STAGE}" = object_state ] && [ ! -f "${TRACKER_CHECKPOINT}" ]; then
  echo "[point-track-v50] frozen tracker checkpoint is missing: ${TRACKER_CHECKPOINT}"
  exit 2
fi
if [ ! -x "${PY}" ] || [ ! -x "${TORCHRUN}" ]; then
  echo "[point-track-v50] runtime is missing under ${VENV_ROOT}/.venv"
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "[point-track-v50] no visible CUDA devices"
  exit 2
fi

TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
MIN_GPU_MEMORY_MIB="$("${PY}" -c 'import torch; print(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) // 2**20)')"
BATCH_PER_GPU="${BATCH_PER_GPU:-auto}"
if [ "${BATCH_PER_GPU}" = auto ]; then
  if [ "${MIN_GPU_MEMORY_MIB}" -ge 76000 ]; then MAX_BATCH_PER_GPU=8
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 45000 ]; then MAX_BATCH_PER_GPU=4
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 22000 ]; then MAX_BATCH_PER_GPU=2
  else MAX_BATCH_PER_GPU=1
  fi
  BATCH_PER_GPU=0
  for candidate in 8 4 2 1; do
    local_candidate=$((candidate * NPROC_PER_NODE))
    if [ "${candidate}" -le "${MAX_BATCH_PER_GPU}" ] && \
       [ "${local_candidate}" -le "${TARGET_GLOBAL_BATCH}" ] && \
       [ $((TARGET_GLOBAL_BATCH % local_candidate)) -eq 0 ]; then
      BATCH_PER_GPU="${candidate}"
      break
    fi
  done
fi
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
LOCAL_BATCH=$((BATCH_PER_GPU * NPROC_PER_NODE))
if [ "${GRAD_ACCUM}" = auto ]; then
  if [ $((TARGET_GLOBAL_BATCH % LOCAL_BATCH)) -ne 0 ]; then
    echo "[point-track-v50] target batch is not divisible by local batch"
    exit 2
  fi
  GRAD_ACCUM=$((TARGET_GLOBAL_BATCH / LOCAL_BATCH))
fi
if [ $((BATCH_PER_GPU * NPROC_PER_NODE * GRAD_ACCUM)) -ne "${TARGET_GLOBAL_BATCH}" ]; then
  echo "[point-track-v50] effective batch contract differs"
  exit 2
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
TRACKER_SEQUENCE_BATCH="${TRACKER_SEQUENCE_BATCH:-1}"

if [ "${STAGE}" = object_state ]; then
  STEPS="${STEPS:-50000}"
  LR="${LR:-2e-4}"
  LR_FLOOR="${LR_FLOOR:-2e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-2500}"
else
  STEPS="${STEPS:-30000}"
  LR="${LR:-2e-4}"
  LR_FLOOR="${LR_FLOOR:-2e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-1500}"
fi

mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
ARGS=(
  --stage "${STAGE}" --data "${DATA}" --out "${OUT}"
  --gate_report "${GATE_REPORT}" --source_revision "${SOURCE_REVISION}"
  --dino_checkpoint "${DINO_CHECKPOINT}" --tracker_checkpoint "${TRACKER_CHECKPOINT}"
  --chunk_lengths "${CHUNK_LENGTHS:-8,16,24,32}"
  --temporal_strides "${TEMPORAL_STRIDES:-1,2,3,4}"
  --batch "${BATCH_PER_GPU}" --grad_accum "${GRAD_ACCUM}"
  --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
  --prefetch_factor "${PREFETCH_FACTOR:-2}" --dino_frame_batch "${DINO_FRAME_BATCH}"
  --tracker_sequence_batch "${TRACKER_SEQUENCE_BATCH}"
  --steps "${STEPS}" --lr "${LR}" --lr_floor "${LR_FLOOR}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}" --warmup_steps "${WARMUP_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM:-5.0}" --save_every "${SAVE_EVERY:-2500}"
  --recovery_every "${RECOVERY_EVERY:-250}" --log_every "${LOG_EVERY:-20}"
  --seed "${SEED:-17}" --amp "${AMP:-bf16}"
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}"
  --wandb_entity "${WANDB_ENTITY:-}" --wandb_name "${WANDB_NAME:-${RUN_NAME}}"
  --wandb_group "${WANDB_GROUP:-point-track-object-state-v50}"
  --wandb_tags "${WANDB_TAGS:-v50,pure-video,point-track,object-state,latent-effect}"
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
)
if [ -n "${STATE_GATE_REPORT}" ]; then ARGS+=(--state_gate_report "${STATE_GATE_REPORT}"); fi
if [ -n "${INIT_FROM:-}" ]; then ARGS+=(--init_from "${INIT_FROM}"); fi
if [ -n "${RESUME:-}" ]; then ARGS+=(--resume "${RESUME}"); fi
if [ -n "${WANDB_RUN_ID:-}" ]; then ARGS+=(--wandb_run_id "${WANDB_RUN_ID}"); fi
if [ -n "${MAX_TRAIN_ITEMS:-}" ]; then ARGS+=(--max_train_items "${MAX_TRAIN_ITEMS}"); fi

echo "[point-track-v50] root=${ROOT} data=${DATA} out=${OUT} stage=${STAGE}"
echo "[point-track-v50] world=${NPROC_PER_NODE} gpu_memory_mib=${MIN_GPU_MEMORY_MIB} batch_per_gpu=${BATCH_PER_GPU} grad_accum=${GRAD_ACCUM} effective_batch=${TARGET_GLOBAL_BATCH}"
echo "[point-track-v50] dino_frame_batch=${DINO_FRAME_BATCH} tracker_sequence_batch=${TRACKER_SEQUENCE_BATCH} workers_per_rank=${WORKERS_PER_RANK}"
echo "[point-track-v50] source_revision=${SOURCE_REVISION}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
cd "${ROOT}" || exit 2
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_point_track_object_state_v50.py" "${ARGS[@]}"
