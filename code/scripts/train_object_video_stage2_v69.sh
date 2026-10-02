#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
STATE_RUN="${RUNTIME_ROOT}/outputs/object_video_v69_state_seed17_40e4586_20260929_003931"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
if [[ -n "${RESUME:-}" ]]; then
  SAVED_METADATA="$("${PY}" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",mmap=True,weights_only=False); print(c["world_size"]); print(c["args"]["source_revision"]); print(c["args"]["out"]); print(c["args"]["workers"])' "${RESUME}")"
  mapfile -t SAVED <<< "${SAVED_METADATA}"
  NPROC_PER_NODE="${SAVED[0]}"
  SOURCE_REVISION="${SAVED[1]}"
  ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${SOURCE_REVISION}"
  OUT="${SAVED[2]}"
  WORKERS_PER_RANK="${SAVED[3]}"
else
  SOURCE_REVISION="$(<"${ROOT}/SOURCE_REVISION")"
  OUT="${OUT:-${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_seed17_state6000}"
fi
STATE_CHECKPOINT="${STATE_CHECKPOINT:-${RUNTIME_ROOT}/outputs/object_video_v69_state_change_held_aed14bb_20261001_235841/checkpoint_snapshot.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
ENCODER="${ENCODER:-dinov3_vitl16}"
ENCODER_REPOSITORY="${ENCODER_REPOSITORY:-${STATE_RUN}/encoder_source}"
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
NUMERICAL_ARGS=()
if [[ "${DETERMINISTIC:-0}" == 1 ]]; then
  NUMERICAL_ARGS+=(--deterministic)
fi
mkdir -p "${OUT}"
echo "[object-video-stage2-v69] mode=foreground stage=dynamics preset=large processes=${NPROC_PER_NODE} out=${OUT} code=${SOURCE_REVISION}"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT}/code/scripts/train_object_video_sequence_v69.py" \
  --manifest "${MANIFEST:-${STATE_RUN}/dataset.json}" --out "${OUT}" \
  --encoder "${ENCODER}" --encoder_repository "${ENCODER_REPOSITORY}" --encoder_weights "${ENCODER_WEIGHTS}" \
  --encoder_frame_batch "${ENCODER_FRAME_BATCH:-2}" --stage dynamics --stage2_preset large --dynamics_checkpoint_blocks \
  --state_checkpoint "${STATE_CHECKPOINT}" --config "${MODEL_CONFIG}" \
  --posterior_geometry "${POSTERIOR_GEOMETRY:-inherit}" \
  --resume "${RESUME:-}" --source_revision "${SOURCE_REVISION}" \
  "${NUMERICAL_ARGS[@]}" \
  --batch "${BATCH_PER_GPU:-2}" --global_batch "${TARGET_GLOBAL_BATCH:-256}" --workers "${WORKERS_PER_RANK}" \
  --steps "${STEPS:-30000}" --stop_after "${STOP_AFTER:-0}" --lr "${LR:-0.0002}" --seed "${SEED:-17}" \
  --log_every "${LOG_EVERY:-20}" --save_every "${SAVE_EVERY:-2500}" --recovery_every "${RECOVERY_EVERY:-250}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${RUN_NAME:-$(basename "${OUT}")}" --wandb_mode "${WANDB_MODE:-online}" \
  2>&1 | tee -a "${OUT}/train.log"
