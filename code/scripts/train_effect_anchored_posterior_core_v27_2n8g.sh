#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
required=(WORLD_SIZE RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "[effect-core-2n8g] missing environment variable: ${name}" >&2
        exit 2
    fi
done
if [[ "${WORLD_SIZE}" -ne 2 || "${NPROC_PER_NODE}" -ne 8 ]]; then
    echo "[effect-core-2n8g] expected exactly two nodes x eight GPUs" >&2
    exit 2
fi
if [[ "${RANK}" -lt 0 || "${RANK}" -ge "${WORLD_SIZE}" ]]; then
    echo "[effect-core-2n8g] invalid node rank: ${RANK}" >&2
    exit 2
fi

DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
SOURCE_RUN="${SOURCE_RUN:-${ROOT}/outputs/visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
SOURCE="${SOURCE:-${SOURCE_RUN}/joint_0012000.pt}"
GATE_REPORT="${GATE_REPORT:-${SOURCE_RUN}/design_gate_step12000.json}"
CODE_MANIFEST="${CODE_MANIFEST:-${SOURCE_RUN}/training_code_manifest.sha256}"
RUNTIME_CODE_MANIFEST="${RUNTIME_CODE_MANIFEST:-${SOURCE_RUN}/training_runtime_code_manifest.sha256}"
EVALUATION_MANIFEST="${EVALUATION_MANIFEST:-${SOURCE_RUN}/evaluation_code_manifest.sha256}"
CODE_PROVENANCE_EXCEPTION="${CODE_PROVENANCE_EXCEPTION:-${SOURCE_RUN}/code_provenance_exception.json}"
EVIDENCE_MANIFEST="${EVIDENCE_MANIFEST:-${SOURCE_RUN}/design_evidence_manifest.sha256}"
SCALE_RUN="${SCALE_RUN:-${ROOT}/outputs/visual_sequence_matched_flat_h4q4_dino_rgb_4gpu_v2_seed17_20260724}"
SCALE_GATE="${SCALE_GATE:-${SCALE_RUN}/object_flat_evaluation_step12000/object_flat_gate.json}"
SCALE_MANIFEST="${SCALE_MANIFEST:-${SCALE_RUN}/scale_readiness_evidence_manifest.sha256}"
FLAT_CHECKPOINT="${FLAT_CHECKPOINT:-${SCALE_RUN}/flat_suite_0012000.pt}"
RESUME="${RESUME:-}"
SEED="${SEED:-17}"
JOINT_STEPS="${JOINT_STEPS:-100000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
LR="${LR:-5e-5}"
LR_FLOOR="${LR_FLOOR:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
RGB_LOSS_WEIGHT="${RGB_LOSS_WEIGHT:-0.5}"
RGB_CHANGE_LOSS_WEIGHT="${RGB_CHANGE_LOSS_WEIGHT:-1.0}"
RGB_CHANGE_THRESHOLD="${RGB_CHANGE_THRESHOLD:-0.04}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
RUN_NAME="${RUN_NAME:-effect_anchored_posterior_core_h4q4_2n8g_v27_seed${SEED}}"
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-effect-anchored-posterior-core-2n8g}"
WANDB_TAGS="${WANDB_TAGS:-rt2-visual,no-language,posterior-oracle,change-balanced,v27,ddp-2n8g}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"
CONTRACT_WAIT_SECONDS="${CONTRACT_WAIT_SECONDS:-300}"

for path in "${DATA}" "${SOURCE_RUN}" "${SCALE_RUN}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[effect-core-2n8g] required absolute directory is missing: ${path}" >&2
        exit 2
    fi
done
for path in "${SOURCE}" "${GATE_REPORT}" "${CODE_MANIFEST}" \
    "${RUNTIME_CODE_MANIFEST}" "${EVALUATION_MANIFEST}" \
    "${CODE_PROVENANCE_EXCEPTION}" "${EVIDENCE_MANIFEST}" "${SCALE_GATE}" \
    "${SCALE_MANIFEST}" "${FLAT_CHECKPOINT}"; do
    if [[ "${path}" != /* || ! -f "${path}" ]]; then
        echo "[effect-core-2n8g] required absolute file is missing: ${path}" >&2
        exit 2
    fi
done
if [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! sha256sum -c --status "${DATA}/episode_manifest.verified.sha256"; then
    echo "[effect-core-2n8g] verified sequence data is missing" >&2
    exit 2
fi
if [[ "${OUT}" != /* || "${LOG_ROOT}" != /* || "${WANDB_DIR}" != /* ]]; then
    echo "[effect-core-2n8g] output and log paths must be absolute" >&2
    exit 2
fi
if ! "${ROOT}/.venv/bin/python" \
    "${ROOT}/code/scripts/verify_visual_sequence_code_provenance.py" \
    --root "${ROOT}" \
    --training_manifest "${CODE_MANIFEST}" \
    --runtime_manifest "${RUNTIME_CODE_MANIFEST}" \
    --evaluation_manifest "${EVALUATION_MANIFEST}" \
    --exception "${CODE_PROVENANCE_EXCEPTION}" >/dev/null; then
    echo "[effect-core-2n8g] code provenance differs from validated v27" >&2
    exit 2
fi
if ! (sha256sum -c --status "${EVIDENCE_MANIFEST}" \
    && sha256sum -c --status "${SCALE_MANIFEST}"); then
    echo "[effect-core-2n8g] code differs from the validated v27 evidence" >&2
    exit 2
fi
preflight="$(${ROOT}/.venv/bin/python \
    "${ROOT}/code/scripts/verify_devup_preflight.py" \
    --gate "${GATE_REPORT}" \
    --source "${SOURCE}" \
    --data "${DATA}" \
    --scale_gate "${SCALE_GATE}" \
    --scale_manifest "${SCALE_MANIFEST}" \
    --flat_checkpoint "${FLAT_CHECKPOINT}")"
echo "[effect-core-2n8g] preflight=${preflight}"

if [[ "${WANDB_MODE}" != "online" \
      && "${WANDB_MODE}" != "offline" ]]; then
    echo "[effect-core-2n8g] W&B must be online or offline" >&2
    exit 2
fi
if ! "${ROOT}/.venv/bin/python" -c "import wandb" >/dev/null 2>&1; then
    echo "[effect-core-2n8g] wandb is not installed" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" == "online" ]] \
    && [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[effect-core-2n8g] online W&B requires credentials" >&2
    exit 2
fi
if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[effect-core-2n8g] resume checkpoint is missing: ${RESUME}" >&2
        exit 2
    fi
    resume_dir="$(dirname "$(readlink -f "${RESUME}")")"
    output_dir="$(readlink -m "${OUT}")"
    if [[ "${resume_dir}" != "${output_dir}" ]]; then
        echo "[effect-core-2n8g] resume checkpoint must belong to ${OUT}" >&2
        exit 2
    fi
    if [[ -z "${WANDB_RUN_ID}" && ! -s "${OUT}/wandb_run_id.txt" ]]; then
        echo "[effect-core-2n8g] resume requires the original W&B run id" >&2
        exit 2
    fi
    RESUME="$(readlink -f "${RESUME}")"
    init_args=(--resume "${RESUME}")
else
    if [[ -e "${OUT}/latest.pt" ]]; then
        echo "[effect-core-2n8g] refusing to overwrite run: ${OUT}" >&2
        exit 2
    fi
    init_args=(--init_from "${SOURCE}")
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-30}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

cd "${ROOT}"
mkdir -p "${OUT}" "${LOG_ROOT}" "${WANDB_DIR}"
train_args=(
    code/scripts/train_adaptive_gaussian_wm.py
    --data "${DATA}"
    --data_format sequence
    --history_frames 4
    --future_frames 4
    --sequence_anchors 3,5,8
    --out "${OUT}"
    --profile full
    --representation_steps 0
    --joint_steps "${JOINT_STEPS}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${GRAD_ACCUM}"
    --workers "${WORKERS_PER_RANK}"
    --lr "${LR}"
    --lr_floor "${LR_FLOOR}"
    --warmup_steps "${WARMUP_STEPS}"
    --warmup_fraction 0.0
    --weight_decay 1e-4
    --save_every "${SAVE_EVERY}"
    --log_every "${LOG_EVERY}"
    --seed "${SEED}"
    --amp bf16
    --language_condition off
    --rgb_supervision on
    --rgb_short_side 256
    --rgb_pad_multiple 16
    --rgb_render_chunk 8192
    --rgb_loss_weight "${RGB_LOSS_WEIGHT}"
    --rgb_ssim_weight 0.2
    --rgb_change_loss_weight "${RGB_CHANGE_LOSS_WEIGHT}"
    --rgb_change_threshold "${RGB_CHANGE_THRESHOLD}"
    --language_effect_weight 0.0
    --zero_action_margin_weight 5.0
    --posterior_dynamics_gate
    --posterior_update_scope full
    --action_anchor object_slot
    --canonical_center_gate 1.0
    --canonical_activity_gate
    --canonical_activity_power 1.0
    --action_residual_dim 8
    --action_residual_gate 1.0
    --action_residual_dropout 0.0
    --semantic_action_basis rgb
    --wandb_mode "${WANDB_MODE}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_name "${WANDB_NAME}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_tags "${WANDB_TAGS}"
    --wandb_run_id "${WANDB_RUN_ID}"
    --wandb_dir "${WANDB_DIR}"
)
train_args+=("${init_args[@]}")
cmd=(
    .venv/bin/torchrun
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${WORLD_SIZE}"
    --node_rank "${RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    "${train_args[@]}"
)
global_batch=$((BATCH_PER_GPU * WORLD_SIZE * NPROC_PER_NODE * GRAD_ACCUM))
if [[ "${global_batch}" -ne 256 ]]; then
    echo "[effect-core-2n8g] global batch must remain 256, got ${global_batch}" >&2
    exit 2
fi
echo "[effect-core-2n8g] node=${RANK}/${WORLD_SIZE} global_batch=${global_batch}"
echo "[effect-core-2n8g] scope=future_conditioned_posterior_core_only"
printf '[effect-core-2n8g] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
dry_run="${DRY_RUN:-0}"
if [[ "${dry_run}" != "0" && "${dry_run}" != "1" ]]; then
    echo "[effect-core-2n8g] DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ "${dry_run}" == "0" ]]; then
    visible_gpus="$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "${visible_gpus}" -ne "${NPROC_PER_NODE}" ]]; then
        echo "[effect-core-2n8g] visible GPUs=${visible_gpus}, expected ${NPROC_PER_NODE}" >&2
        exit 2
    fi
else
    echo "[effect-core-2n8g] dry-run skips the local GPU topology check"
fi
if [[ "${CONTRACT_WAIT_SECONDS}" -le 0 ]]; then
    echo "[effect-core-2n8g] contract wait must be positive" >&2
    exit 2
fi
mode="warm_start"
attempt_checkpoint="${SOURCE}"
if [[ -n "${RESUME}" ]]; then
    mode="resume"
    attempt_checkpoint="${RESUME}"
fi
attempt_key="${mode}_$(basename "${attempt_checkpoint%.pt}")"
contract_dir="${OUT}/launch_attempts"
node_contract="${contract_dir}/${attempt_key}.node_${RANK}.json"
mkdir -p "${contract_dir}"
.venv/bin/python code/scripts/prepare_effect_core_launch_contract.py \
    --root "${ROOT}" \
    --out "${OUT}" \
    --data "${DATA}" \
    --attempt_key "${attempt_key}" \
    --mode "${mode}" \
    --checkpoint "${attempt_checkpoint}" \
    --nnodes "${WORLD_SIZE}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --batch_per_gpu "${BATCH_PER_GPU}" \
    --grad_accum "${GRAD_ACCUM}" \
    --evidence "source_checkpoint=${SOURCE}" \
    --evidence "design_gate=${GATE_REPORT}" \
    --evidence "training_code_manifest=${CODE_MANIFEST}" \
    --evidence "runtime_code_manifest=${RUNTIME_CODE_MANIFEST}" \
    --evidence "evaluation_code_manifest=${EVALUATION_MANIFEST}" \
    --evidence "code_provenance_exception=${CODE_PROVENANCE_EXCEPTION}" \
    --evidence "design_evidence_manifest=${EVIDENCE_MANIFEST}" \
    --evidence "scale_gate=${SCALE_GATE}" \
    --evidence "scale_evidence_manifest=${SCALE_MANIFEST}" \
    --evidence "flat_checkpoint=${FLAT_CHECKPOINT}" \
    --evidence "data_manifest=${DATA}/episode_manifest.json" \
    --evidence "verified_data_manifest=${DATA}/episode_manifest.verified.sha256" \
    --evidence "episode_source_index=${DATA}/episode_source_index.json" \
    --output "${node_contract}" \
    --training_args "${train_args[@]}"
contract_started="$(date +%s)"
while true; do
    ready=1
    for node in 0 1; do
        if [[ ! -f "${contract_dir}/${attempt_key}.node_${node}.json" ]]; then
            ready=0
        fi
    done
    if [[ "${ready}" -eq 1 ]]; then
        break
    fi
    if (( $(date +%s) - contract_started >= CONTRACT_WAIT_SECONDS )); then
        echo "[effect-core-2n8g] timed out waiting for peer launch contract" >&2
        exit 2
    fi
    sleep 2
done
reference_contract="${contract_dir}/${attempt_key}.node_0.json"
for node in 1; do
    if ! cmp -s "${reference_contract}" \
        "${contract_dir}/${attempt_key}.node_${node}.json"; then
        echo "[effect-core-2n8g] node launch contracts differ" >&2
        exit 2
    fi
done
attempt_contract="${contract_dir}/${attempt_key}.json"
if [[ "${RANK}" -eq 0 ]]; then
    if [[ -f "${attempt_contract}" ]]; then
        cmp -s "${reference_contract}" "${attempt_contract}" || exit 2
    else
        cp "${reference_contract}" "${attempt_contract}.tmp.$$"
        mv "${attempt_contract}.tmp.$$" "${attempt_contract}"
    fi
    if [[ "${mode}" == "warm_start" ]]; then
        if [[ -f "${OUT}/launch_contract.json" ]]; then
            cmp -s "${reference_contract}" "${OUT}/launch_contract.json" || exit 2
        else
            cp "${reference_contract}" "${OUT}/launch_contract.json.tmp.$$"
            mv "${OUT}/launch_contract.json.tmp.$$" "${OUT}/launch_contract.json"
        fi
    fi
fi
while [[ ! -f "${attempt_contract}" ]]; do
    if (( $(date +%s) - contract_started >= CONTRACT_WAIT_SECONDS )); then
        echo "[effect-core-2n8g] timed out waiting for canonical contract" >&2
        exit 2
    fi
    sleep 2
done
if ! cmp -s "${reference_contract}" "${attempt_contract}"; then
    echo "[effect-core-2n8g] canonical launch contract differs" >&2
    exit 2
fi
echo "[effect-core-2n8g] launch_contract=${attempt_contract}"
if [[ "${dry_run}" == "1" ]]; then
    echo "[effect-core-2n8g] dry-run validated both node contracts; torchrun skipped"
    exit 0
fi
exec > >(tee -a "${LOG_ROOT}/node_${RANK}.log") 2>&1
exec "${cmd[@]}"
