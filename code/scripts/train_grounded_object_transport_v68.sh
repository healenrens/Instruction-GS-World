#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
STAGE="${STAGE:-state}"
SOURCE_REVISION="${SOURCE_REVISION:-local-unversioned}"
RUN_NAME="${RUN_NAME:-grounded_object_transport_v68_${STAGE}_${SOURCE_REVISION:0:7}}"
OUT="${OUT:-${RUNTIME_ROOT}/outputs/${RUN_NAME}}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${OUT}"
"${VENV_ROOT}/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-8}" \
  "${ROOT}/code/scripts/train_grounded_object_transport_v68.py" \
  --manifest "${TEACHER_MANIFEST}" --out "${OUT}" --stage "${STAGE}" --source_revision "${SOURCE_REVISION}" \
  --state_checkpoint "${STATE_CHECKPOINT:-}" --resume "${RESUME:-}" \
  --batch "${BATCH_PER_GPU:-4}" --global_batch "${TARGET_GLOBAL_BATCH:-256}" --workers "${WORKERS_PER_RANK:-2}" \
  --points "${POINTS_PER_SAMPLE:-256}" --steps "${STEPS:-30000}" --lr "${LR:-0.0002}" \
  --seed "${SEED:-17}" \
  --dino_checkpoint "${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}" \
  --siglip_checkpoint "${SIGLIP_CHECKPOINT:-${RUNTIME_ROOT}/models/siglip2-base-patch16-224}" \
  --dino_frame_batch "${DINO_FRAME_BATCH:-96}" --siglip_frame_batch "${SIGLIP_FRAME_BATCH:-96}" \
  --save_every "${SAVE_EVERY:-2500}" --recovery_every "${RECOVERY_EVERY:-500}" --log_every "${LOG_EVERY:-20}" \
  --wandb_mode "${WANDB_MODE:-online}" --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" --wandb_name "${RUN_NAME}" \
  2>&1 | tee -a "${OUT}/train.log"
