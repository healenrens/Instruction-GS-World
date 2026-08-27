#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/gated_residual_object_transition_v60_seed17_c2930c6/v60_transition_0010000.pt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
EVALUATOR_REVISION="${EVALUATOR_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
OUTPUT="${OUTPUT:-${RUNTIME_ROOT}/outputs/v60_evaluations/step10000_${EVALUATOR_REVISION}_observation_reconstruction.json}"

GPU_MEMORY_MIB="$(${PY} -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory // 2**20)')"
EVAL_BATCH="${EVAL_BATCH:-auto}"
if [ "${EVAL_BATCH}" = auto ]; then
  if [ "${GPU_MEMORY_MIB}" -ge 76000 ]; then EVAL_BATCH=32
  elif [ "${GPU_MEMORY_MIB}" -ge 45000 ]; then EVAL_BATCH=16
  elif [ "${GPU_MEMORY_MIB}" -ge 22000 ]; then EVAL_BATCH=8
  else EVAL_BATCH=4
  fi
fi
EVAL_DINO_FRAME_BATCH="${EVAL_DINO_FRAME_BATCH:-auto}"
if [ "${EVAL_DINO_FRAME_BATCH}" = auto ]; then
  if [ "${GPU_MEMORY_MIB}" -ge 76000 ]; then EVAL_DINO_FRAME_BATCH=192
  elif [ "${GPU_MEMORY_MIB}" -ge 45000 ]; then EVAL_DINO_FRAME_BATCH=96
  elif [ "${GPU_MEMORY_MIB}" -ge 22000 ]; then EVAL_DINO_FRAME_BATCH=48
  else EVAL_DINO_FRAME_BATCH=16
  fi
fi

mkdir -p "$(dirname "${OUTPUT}")" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"

echo "[v60-observation-reeval] revision=${EVALUATOR_REVISION}"
echo "[v60-observation-reeval] checkpoint=${CHECKPOINT}"
echo "[v60-observation-reeval] batch=${EVAL_BATCH} dino_frame_batch=${EVAL_DINO_FRAME_BATCH}"
echo "[v60-observation-reeval] output=${OUTPUT}"

exec "${PY}" "${ROOT}/code/scripts/evaluate_object_observation_reconstruction_v60.py" \
  --data_index "${DATA_INDEX}" \
  --checkpoint "${CHECKPOINT}" \
  --dino_checkpoint "${DINO_CHECKPOINT}" \
  --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --evaluator_revision "${EVALUATOR_REVISION}" \
  --expected_checkpoint_step "${EXPECTED_CHECKPOINT_STEP:-10000}" \
  --history_lengths "${HISTORY_LENGTHS:-1,2,3,4}" \
  --teacher_future_frames "${TEACHER_FUTURE_FRAMES:-4}" \
  --temporal_step_ms "${TEMPORAL_STEP_MS:-100,200,400,800}" \
  --samples_per_condition "${EVAL_SAMPLES_PER_CONDITION:-64}" \
  --batch "${EVAL_BATCH}" \
  --dino_frame_batch "${EVAL_DINO_FRAME_BATCH}" \
  --support_sigma "${SUPPORT_SIGMA:-0.10}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES:-2000}" \
  --amp "${AMP:-bf16}" \
  --seed "${EVAL_SEED:-117}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_name "${WANDB_NAME:-v60_step10000_observation_reeval_${EVALUATOR_REVISION:0:7}}" \
  --wandb_group "${WANDB_GROUP:-v60-observation-reconstruction-eval}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
