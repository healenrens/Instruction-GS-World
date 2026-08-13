#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/observation_complete_object_state_v46_seed17}"
GATE_REPORT="${GATE_REPORT:-}"
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"

if [ -n "${INIT_FROM:-}" ]; then
  echo "[object-state-v46] INIT_FROM is forbidden; v46 trains a new state model"
  exit 2
fi
if [ -z "${GATE_REPORT}" ] || [ ! -f "${GATE_REPORT}" ]; then
  echo "[object-state-v46] GATE_REPORT is missing: ${GATE_REPORT}"
  exit 2
fi
if [ ! -x "${PY}" ] || [ ! -x "${TORCHRUN}" ]; then
  echo "[object-state-v46] runtime is missing under ${VENV_ROOT}/.venv"
  exit 2
fi
if [ ! -f "${DATA}/episode_manifest.json" ]; then
  echo "[object-state-v46] RGB manifest is missing under ${DATA}"
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "[object-state-v46] no visible CUDA devices"
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
    echo "[object-state-v46] target batch is not divisible by local batch"
    exit 2
  fi
  GRAD_ACCUM=$((TARGET_GLOBAL_BATCH / LOCAL_BATCH))
fi
if [ $((BATCH_PER_GPU * NPROC_PER_NODE * GRAD_ACCUM)) -ne "${TARGET_GLOBAL_BATCH}" ]; then
  echo "[object-state-v46] effective batch contract differs"
  exit 2
fi

mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
ARGS=(
  --data "${DATA}" --out "${OUT}" --gate_report "${GATE_REPORT}"
  --chunk_lengths "${CHUNK_LENGTHS:-8,16,24,32}"
  --temporal_strides "${TEMPORAL_STRIDES:-1,2,3,4}"
  --observation_mask_probability "${OBSERVATION_MASK_PROBABILITY:-0.20}"
  --batch "${BATCH_PER_GPU}" --grad_accum "${GRAD_ACCUM}"
  --target_global_batch "${TARGET_GLOBAL_BATCH}" --workers "${WORKERS_PER_RANK}"
  --prefetch_factor "${PREFETCH_FACTOR:-2}" --dino_frame_batch "${DINO_FRAME_BATCH}"
  --steps "${STEPS:-50000}" --state_phase_steps "${STATE_PHASE_STEPS:-15000}"
  --goal_phase_steps "${GOAL_PHASE_STEPS:-30000}"
  --curriculum_ramp_steps "${CURRICULUM_RAMP_STEPS:-2000}"
  --lr "${LR:-2e-4}" --lr_floor "${LR_FLOOR:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}" --warmup_steps "${WARMUP_STEPS:-2500}"
  --max_grad_norm "${MAX_GRAD_NORM:-5.0}" --save_every "${SAVE_EVERY:-2500}"
  --recovery_every "${RECOVERY_EVERY:-250}" --log_every "${LOG_EVERY:-20}"
  --seed "${SEED:-17}" --amp "${AMP:-bf16}"
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}"
  --wandb_entity "${WANDB_ENTITY:-}" --wandb_name "${WANDB_NAME:-${RUN_NAME:-observation_complete_object_state_v46_seed17}}"
  --wandb_group "${WANDB_GROUP:-observation-complete-object-state-v46}"
  --wandb_tags "${WANDB_TAGS:-v46,pure-video,object-state,latent-effect}"
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
)
if [ -n "${RESUME:-}" ]; then ARGS+=(--resume "${RESUME}"); fi
if [ -n "${WANDB_RUN_ID:-}" ]; then ARGS+=(--wandb_run_id "${WANDB_RUN_ID}"); fi
if [ -n "${MAX_TRAIN_ITEMS:-}" ]; then ARGS+=(--max_train_items "${MAX_TRAIN_ITEMS}"); fi

echo "[object-state-v46] root=${ROOT} data=${DATA} out=${OUT}"
echo "[object-state-v46] world=${NPROC_PER_NODE} gpu_memory_mib=${MIN_GPU_MEMORY_MIB} batch_per_gpu=${BATCH_PER_GPU} grad_accum=${GRAD_ACCUM} effective_batch=${TARGET_GLOBAL_BATCH}"
echo "[object-state-v46] dino_frame_batch=${DINO_FRAME_BATCH} workers_per_rank=${WORKERS_PER_RANK}"
cd "${ROOT}" || exit 2
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_observation_complete_object_state_v46.py" "${ARGS[@]}"
