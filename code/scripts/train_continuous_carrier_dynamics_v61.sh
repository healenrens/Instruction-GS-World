#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
TORCHRUN="${VENV_ROOT}/.venv/bin/torchrun"
VARIANT="${VARIANT:-siglip2_dino_object}"
EFFECT_CAPACITY="${EFFECT_CAPACITY:-8x64_bound}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
DECODE_REPORT="${DECODE_REPORT:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/decode_frontier.json}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
SIGLIP2_CHECKPOINT="${SIGLIP2_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
STATE_RUN_NAME="${STATE_RUN_NAME:-continuous_carrier_v61_${VARIANT}_seed17_${SOURCE_REVISION:0:7}}"
STATE_CHECKPOINT="${STATE_CHECKPOINT:-${RUNTIME_ROOT}/outputs/${STATE_RUN_NAME}/latest.pt}"
GATE_REPORT="${GATE_REPORT:-${RUNTIME_ROOT}/outputs/v61_dynamics_gates/${SOURCE_REVISION}_${EFFECT_CAPACITY}.json}"
RUN_NAME="${RUN_NAME:-continuous_carrier_dynamics_v61_${EFFECT_CAPACITY}_seed17_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
if [ "${NPROC_PER_NODE}" = auto ]; then
  NPROC_PER_NODE="$(${PY} -c 'import torch; print(torch.cuda.device_count())')"
fi
BATCH_PER_GPU="${BATCH_PER_GPU:-16}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
LOCAL_BATCH=$((BATCH_PER_GPU * NPROC_PER_NODE))
GRAD_ACCUM="${GRAD_ACCUM:-$((TARGET_GLOBAL_BATCH / LOCAL_BATCH))}"
STEPS="${STEPS:-20000}"
WARMUP_STEPS="${WARMUP_STEPS:-$((STEPS / 20))}"

ARGS=(
  --variant "${VARIANT}"
  --effect_capacity "${EFFECT_CAPACITY}"
  --state_checkpoint "${STATE_CHECKPOINT}"
  --data_index "${DATA_INDEX}"
  --out "${OUT}"
  --gate_report "${GATE_REPORT}"
  --decode_report "${DECODE_REPORT}"
  --source_revision "${SOURCE_REVISION}"
  --dino_checkpoint "${DINO_CHECKPOINT}"
  --siglip2_checkpoint "${SIGLIP2_CHECKPOINT}"
  --tracker_checkpoint "${TRACKER_CHECKPOINT}"
  --chunk_lengths "${CHUNK_LENGTHS:-4,6,8}"
  --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400}"
  --held_group_stride "${HELD_GROUP_STRIDE:-20}"
  --batch "${BATCH_PER_GPU}"
  --grad_accum "${GRAD_ACCUM}"
  --target_global_batch "${TARGET_GLOBAL_BATCH}"
  --workers "${WORKERS_PER_RANK:-4}"
  --prefetch_factor "${PREFETCH_FACTOR:-2}"
  --student_frame_batch "${STUDENT_FRAME_BATCH:-64}"
  --dino_frame_batch "${DINO_FRAME_BATCH:-128}"
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
  --wandb_entity "${WANDB_ENTITY:-}"
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}"
  --wandb_group "${WANDB_GROUP:-continuous-carrier-dynamics-v61}"
  --wandb_tags "${WANDB_TAGS:-v61,posterior-dynamics,${EFFECT_CAPACITY}}"
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
)
if [ -n "${RESUME:-}" ]; then ARGS+=(--resume "${RESUME}"); fi
if [ -n "${WANDB_RUN_ID:-}" ]; then ARGS+=(--wandb_run_id "${WANDB_RUN_ID}"); fi
if [ -n "${MAX_TRAIN_ITEMS:-}" ]; then ARGS+=(--max_train_items "${MAX_TRAIN_ITEMS}"); fi

echo "[continuous-carrier-dynamics-v61] capacity=${EFFECT_CAPACITY} revision=${SOURCE_REVISION}"
echo "[continuous-carrier-dynamics-v61] world=${NPROC_PER_NODE} batch=${BATCH_PER_GPU} accum=${GRAD_ACCUM} effective=${TARGET_GLOBAL_BATCH}"
echo "[continuous-carrier-dynamics-v61] state=${STATE_CHECKPOINT} out=${OUT}"

export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/.cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${OUT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"
exec "${TORCHRUN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_continuous_carrier_dynamics_v61.py" "${ARGS[@]}"
