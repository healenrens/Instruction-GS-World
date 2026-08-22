#!/usr/bin/env bash

set -u

COMMAND="${1:-teacher}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
PY="${VENV_ROOT}/.venv/bin/python"
DATA_INDEX="${DATA_INDEX:-${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json}"
TRAINING_RUN="${TRAINING_RUN:-verified_relation_object_state_v56_seed17_20accfe}"
CHECKPOINT="${CHECKPOINT:-${RUNTIME_ROOT}/outputs/${TRAINING_RUN}/v56_object_state_0020000.pt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-${RUNTIME_ROOT}/models/dinov2_vitl14/model.safetensors}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth}"
EVALUATOR_REVISION="${EVALUATOR_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
EVAL_NAME="${EVAL_NAME:-verified_relation_object_state_v56_step20000_${EVALUATOR_REVISION:0:7}}"
EVAL_ROOT="${EVAL_ROOT:-${RUNTIME_ROOT}/outputs/v56_evaluations/${EVAL_NAME}}"
TEACHER_REPORT="${TEACHER_REPORT:-${EVAL_ROOT}/source_balanced_teacher_diagnostics.json}"
INDEPENDENT_REPORT="${INDEPENDENT_REPORT:-${EVAL_ROOT}/independent_object_truth.json}"
INDEPENDENT_DATA="${INDEPENDENT_DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
TRUTH_MANIFEST="${TRUTH_MANIFEST:-}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RUNTIME_ROOT}/third_party/co-tracker:${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${EVAL_ROOT}" "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

run_on_selected_gpu() {
  if [ -n "${EVAL_CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "$@"
  else
    "$@"
  fi
}

teacher_evaluation() {
  run_on_selected_gpu "${PY}" \
    "${ROOT}/code/scripts/evaluate_verified_relation_object_state_v56.py" \
    --data_index "${DATA_INDEX}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${TEACHER_REPORT}" \
    --evaluator_revision "${EVALUATOR_REVISION}" \
    --dino_checkpoint "${DINO_CHECKPOINT}" \
    --tracker_checkpoint "${TRACKER_CHECKPOINT}" \
    --expected_step "${EXPECTED_STEP:-20000}" \
    --chunk_lengths "${EVAL_CHUNK_LENGTHS:-4,8}" \
    --temporal_step_ms "${EVAL_TEMPORAL_STEP_MS:-100,200,400}" \
    --items_per_source "${EVAL_ITEMS_PER_SOURCE:-32}" \
    --batch "${EVAL_BATCH:-8}" \
    --workers "${EVAL_WORKERS:-4}" \
    --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-64}" \
    --causal_items "${EVAL_CAUSAL_ITEMS:-8}" \
    --amp "${AMP:-bf16}" \
    --seed "${SEED:-17}" \
    --wandb_mode "${WANDB_MODE:-online}" \
    --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
    --wandb_entity "${WANDB_ENTITY:-}" \
    --wandb_name "${WANDB_NAME:-${EVAL_NAME}_teacher}" \
    --wandb_group "${WANDB_GROUP:-verified-relation-object-state-v56-eval}" \
    --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}" \
    ${REQUIRE_TRAINING_EVIDENCE_GATE:+--require_training_evidence_gate}
}

independent_evaluation() {
  if [ -z "${TRUTH_MANIFEST}" ]; then
    echo "[v56-eval] TRUTH_MANIFEST is required for independent promotion evaluation"
    return 2
  fi
  run_on_selected_gpu "${PY}" \
    "${ROOT}/code/scripts/evaluate_independent_verified_relation_object_state_v56.py" \
    --data "${INDEPENDENT_DATA}" \
    --truth_manifest "${TRUTH_MANIFEST}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${INDEPENDENT_REPORT}" \
    --evaluator_revision "${EVALUATOR_REVISION}" \
    --dino_checkpoint "${DINO_CHECKPOINT}" \
    --expected_step "${EXPECTED_STEP:-20000}" \
    --splits "${EVAL_SPLITS:-heldseed,heldtask}" \
    --dino_frame_batch "${EVAL_DINO_FRAME_BATCH:-64}" \
    --amp "${AMP:-bf16}" \
    --wandb_mode "${WANDB_MODE:-online}" \
    --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
    --wandb_entity "${WANDB_ENTITY:-}" \
    --wandb_name "${INDEPENDENT_WANDB_NAME:-${EVAL_NAME}_independent}" \
    --wandb_group "${INDEPENDENT_WANDB_GROUP:-verified-relation-object-state-v56-independent-eval}" \
    --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
}

case "${COMMAND}" in
  teacher)
    teacher_evaluation
    ;;
  independent)
    independent_evaluation
    ;;
  all)
    teacher_evaluation || exit $?
    independent_evaluation
    ;;
  *)
    echo "usage: $0 {teacher|independent|all}"
    exit 2
    ;;
esac
