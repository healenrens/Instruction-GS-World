#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DECODE_REPORT="${DECODE_REPORT:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/decode_frontier.json}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
GATE_REPORT="${GATE_REPORT:-}"
INIT_FROM="${INIT_FROM:-}"
RUN_NAME="${RUN_NAME:-object_transition_objective_v59_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"

if [ -z "${SOURCE_REVISION}" ]; then
  echo "[object-transition-v59] SOURCE_REVISION is missing"
  exit 2
fi
if [ -z "${GATE_REPORT}" ] || [ ! -f "${GATE_REPORT}" ]; then
  echo "[object-transition-v59] GATE_REPORT is missing: ${GATE_REPORT}"
  exit 2
fi
if [ -z "${INIT_FROM}" ] || [ ! -f "${INIT_FROM}" ]; then
  echo "[object-transition-v59] INIT_FROM v58 checkpoint is missing: ${INIT_FROM}"
  exit 2
fi
for path in "${DATA_INDEX}" "${DECODE_REPORT}" "${DINO_CHECKPOINT}" "${TRACKER_CHECKPOINT}"; do
  if [ ! -f "${path}" ]; then
    echo "[object-transition-v59] required runtime file is missing: ${path}"
    exit 2
  fi
done
if [ ! -x "${PY}" ] || [ ! -x "${TORCHRUN}" ]; then
  echo "[object-transition-v59] runtime is missing under ${VENV_ROOT}/.venv"
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$("${PY}" -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "[object-transition-v59] no visible CUDA devices"
  exit 2
fi
MIN_GPU_MEMORY_MIB="$("${PY}" -c 'import torch; print(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) // 2**20)')"
BATCH_PER_GPU="${BATCH_PER_GPU:-auto}"
if [ "${BATCH_PER_GPU}" = auto ]; then
  if [ "${MIN_GPU_MEMORY_MIB}" -ge 76000 ]; then BATCH_PER_GPU=32
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 45000 ]; then BATCH_PER_GPU=16
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 22000 ]; then BATCH_PER_GPU=8
  else BATCH_PER_GPU=4
  fi
fi
LOCAL_BATCH=$((BATCH_PER_GPU * NPROC_PER_NODE))
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
if [ "${GRAD_ACCUM}" = auto ]; then
  if [ $((TARGET_GLOBAL_BATCH % LOCAL_BATCH)) -ne 0 ]; then
    echo "[object-transition-v59] target batch is not divisible by local batch"
    exit 2
  fi
  GRAD_ACCUM=$((TARGET_GLOBAL_BATCH / LOCAL_BATCH))
fi
if [ $((LOCAL_BATCH * GRAD_ACCUM)) -ne "${TARGET_GLOBAL_BATCH}" ]; then
  echo "[object-transition-v59] effective batch contract differs"
  exit 2
fi
DINO_FRAME_BATCH="${DINO_FRAME_BATCH:-auto}"
if [ "${DINO_FRAME_BATCH}" = auto ]; then
  if [ "${MIN_GPU_MEMORY_MIB}" -ge 76000 ]; then DINO_FRAME_BATCH=192
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 45000 ]; then DINO_FRAME_BATCH=96
  elif [ "${MIN_GPU_MEMORY_MIB}" -ge 22000 ]; then DINO_FRAME_BATCH=48
  else DINO_FRAME_BATCH=16
  fi
fi
WORKERS_PER_RANK="${WORKERS_PER_RANK:-auto}"
if [ "${WORKERS_PER_RANK}" = auto ]; then
  WORKERS_PER_RANK="$("${PY}" -c "import os; print(max(2, min(8, (os.cpu_count() or 8) // (2 * ${NPROC_PER_NODE}))))")"
fi
STEPS="${STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-$((STEPS / 20))}"

mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
ARGS=(
  --data_index "${DATA_INDEX}" --out "${OUT}"
  --decode_report "${DECODE_REPORT}" --startup_gate "${GATE_REPORT}"
  --source_revision "${SOURCE_REVISION}"
  --dino_checkpoint "${DINO_CHECKPOINT}"
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" --init_from "${INIT_FROM}"
  --history_lengths "${HISTORY_LENGTHS:-1,2,3,4}"
  --teacher_future_frames "${TEACHER_FUTURE_FRAMES:-4}"
  --chunk_lengths "${CHUNK_LENGTHS:-5,6,7,8}"
  --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400,800}"
  --batch "${BATCH_PER_GPU}" --grad_accum "${GRAD_ACCUM}"
  --target_global_batch "${TARGET_GLOBAL_BATCH}"
  --workers "${WORKERS_PER_RANK}" --prefetch_factor "${PREFETCH_FACTOR:-2}"
  --dino_frame_batch "${DINO_FRAME_BATCH}" --steps "${STEPS}"
  --run_steps "${RUN_STEPS:-0}"
  --lr "${LR:-2e-4}" --lr_floor "${LR_FLOOR:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}" --warmup_steps "${WARMUP_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM:-5.0}"
  --save_every "${SAVE_EVERY:-1000}" --recovery_every "${RECOVERY_EVERY:-100}"
  --log_every "${LOG_EVERY:-20}" --seed "${SEED:-17}" --amp "${AMP:-bf16}"
  --wandb_mode "${WANDB_MODE:-online}"
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}"
  --wandb_entity "${WANDB_ENTITY:-}"
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}"
  --wandb_group "${WANDB_GROUP:-object-transition-objective-v59}"
  --wandb_tags "${WANDB_TAGS:-v59,multisource,pure-video,object-transition,latent-effect,intervention}"
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
)
if [ -n "${RESUME:-}" ]; then ARGS+=(--resume "${RESUME}"); fi
if [ -n "${WANDB_RUN_ID:-}" ]; then ARGS+=(--wandb_run_id "${WANDB_RUN_ID}"); fi
if [ -n "${MAX_TRAIN_ITEMS:-}" ]; then ARGS+=(--max_train_items "${MAX_TRAIN_ITEMS}"); fi

echo "[object-transition-v59] root=${ROOT} data_index=${DATA_INDEX} out=${OUT}"
echo "[object-transition-v59] init_from=${INIT_FROM}"
echo "[object-transition-v59] world=${NPROC_PER_NODE} batch_per_gpu=${BATCH_PER_GPU} grad_accum=${GRAD_ACCUM} effective_batch=${TARGET_GLOBAL_BATCH}"
echo "[object-transition-v59] dino_frame_batch=${DINO_FRAME_BATCH} workers_per_rank=${WORKERS_PER_RANK} steps=${STEPS}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
cd "${ROOT}" || exit 2
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_object_transition_objective_v59.py" "${ARGS[@]}"
