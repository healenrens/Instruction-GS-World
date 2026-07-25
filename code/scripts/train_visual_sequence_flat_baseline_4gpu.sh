#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
OBJECT_RUN="${OBJECT_RUN:-${ROOT}/outputs/visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
REFERENCE="${REFERENCE:-${OBJECT_RUN}/joint_0012000.pt}"
DESIGN_GATE="${DESIGN_GATE:-${OBJECT_RUN}/design_gate_step12000.json}"
DESIGN_MANIFEST="${DESIGN_MANIFEST:-${OBJECT_RUN}/design_evidence_manifest.sha256}"
REQUIRE_DESIGN_PASS="${REQUIRE_DESIGN_PASS:-1}"
RESUME="${RESUME:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-17}"
STEPS="${STEPS:-12000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-2}"
LR="${LR:-2e-4}"
LR_FLOOR="${LR_FLOOR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-600}"
SAVE_EVERY="${SAVE_EVERY:-4000}"
LOG_EVERY="${LOG_EVERY:-20}"
RUN_DATE="${RUN_DATE:-20260724}"
RUN_NAME="${RUN_NAME:-visual_sequence_matched_flat_h4q4_dino_rgb_4gpu_v2_seed${SEED}_${RUN_DATE}}"
OUT="${OUT:-${ROOT}/outputs/${RUN_NAME}}"
TRAINING_MANIFEST="${TRAINING_MANIFEST:-${OUT}/training_code_manifest.sha256}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-instruct-gs-world}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-visual-sequence-matched-flat-baseline}"
WANDB_TAGS="${WANDB_TAGS:-rt2,no-language,dino-rgb,continuous-action,capacity-matched,unstructured-latent,v2,ddp-4gpu}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_DIR="${WANDB_DIR:-${OUT}/wandb}"

for path in "${ROOT}" "${DATA}" "${OBJECT_RUN}" "${REFERENCE}" \
    "${DESIGN_GATE}" "${DESIGN_MANIFEST}" "${OUT}" "${TRAINING_MANIFEST}"; do
    if [[ "${path}" != /* ]]; then
        echo "[flat-baseline] all paths must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ ! -d "${DATA}" ]] \
    || [[ ! -f "${DATA}/episode_manifest.verified.sha256" ]] \
    || ! sha256sum -c --status "${DATA}/episode_manifest.verified.sha256"; then
    echo "[flat-baseline] verified sequence data is missing: ${DATA}" >&2
    exit 2
fi
for path in "${REFERENCE}" "${DESIGN_GATE}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[flat-baseline] required object evidence is missing: ${path}" >&2
        exit 2
    fi
done
if [[ "${REQUIRE_DESIGN_PASS}" != "0" && "${REQUIRE_DESIGN_PASS}" != "1" ]]; then
    echo "[flat-baseline] REQUIRE_DESIGN_PASS must be 0 or 1" >&2
    exit 2
fi
if [[ "${REQUIRE_DESIGN_PASS}" == "1" ]]; then
    if [[ ! -f "${DESIGN_MANIFEST}" ]] \
        || ! sha256sum -c --status "${DESIGN_MANIFEST}"; then
        echo "[flat-baseline] passed object design evidence is invalid" >&2
        exit 2
    fi
elif [[ -f "${DESIGN_MANIFEST}" ]] \
    && ! sha256sum -c --status "${DESIGN_MANIFEST}"; then
    echo "[flat-baseline] existing object design evidence manifest failed" >&2
    exit 2
fi
"${ROOT}/.venv/bin/python" - "${DESIGN_GATE}" "${REQUIRE_DESIGN_PASS}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
status = report.get("status")
required = "pass" if sys.argv[2] == "1" else "fail"
if status != required:
    raise SystemExit(f"object design gate status={status!r}, expected {required!r}")
PY
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[flat-baseline] expected exactly four GPU ids" >&2
    exit 2
fi
global_batch=$((BATCH_PER_GPU * 4 * GRAD_ACCUM))
if [[ "${global_batch}" -ne 256 ]]; then
    echo "[flat-baseline] global batch must remain 256, got ${global_batch}" >&2
    exit 2
fi
if [[ "${STEPS}" -ne 12000 ]]; then
    echo "[flat-baseline] evidence contract requires exactly 12000 steps" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" != "online" \
      && "${WANDB_MODE}" != "offline" \
      && "${WANDB_MODE}" != "disabled" ]]; then
    echo "[flat-baseline] invalid W&B mode: ${WANDB_MODE}" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" != "disabled" ]] \
    && ! "${ROOT}/.venv/bin/python" -c "import wandb" >/dev/null 2>&1; then
    echo "[flat-baseline] wandb is not installed in ${ROOT}/.venv" >&2
    exit 2
fi
if [[ "${WANDB_MODE}" == "online" ]] \
    && [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -q "machine api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[flat-baseline] online W&B requires WANDB_API_KEY or ~/.netrc" >&2
    exit 2
fi
if [[ -n "${RESUME}" ]]; then
    if [[ "${RESUME}" != /* || ! -f "${RESUME}" ]]; then
        echo "[flat-baseline] resume checkpoint is invalid: ${RESUME}" >&2
        exit 2
    fi
    resume_args=(--resume "${RESUME}")
else
    if [[ -d "${OUT}" ]] && [[ -n "$(find "${OUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "[flat-baseline] refusing to overwrite run: ${OUT}" >&2
        exit 2
    fi
    resume_args=()
fi

export HF_HOME="${HF_HOME:-/mnt/pfs/public/xuhaoming/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pfs/public/xuhaoming/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

cd "${ROOT}"
cmd=(
    .venv/bin/torchrun
    --standalone
    --nproc_per_node=4
    code/scripts/train_visual_sequence_flat_baseline.py
    --reference_checkpoint "${REFERENCE}"
    --required_reference_step 12000
    --data "${DATA}"
    --out "${OUT}"
    --history_frames 4
    --future_frames 4
    --sequence_anchors 3,5,8
    --steps "${STEPS}"
    --batch "${BATCH_PER_GPU}"
    --grad_accum "${GRAD_ACCUM}"
    --workers "${WORKERS_PER_RANK}"
    --gradient_checkpointing on
    --lr "${LR}"
    --lr_floor "${LR_FLOOR}"
    --warmup_steps "${WARMUP_STEPS}"
    --weight_decay 1e-4
    --change_loss_weight 1.0
    --history_loss_weight 1.0
    --rgb_short_side 256
    --rgb_pad_multiple 16
    --rgb_loss_weight 0.5
    --rgb_ssim_weight 0.2
    --rgb_change_loss_weight 1.0
    --rgb_change_threshold 0.04
    --gap_reference 1.0
    --save_every "${SAVE_EVERY}"
    --log_every "${LOG_EVERY}"
    --seed "${SEED}"
    --amp bf16
    --wandb_mode "${WANDB_MODE}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_name "${WANDB_NAME}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_tags "${WANDB_TAGS}"
    --wandb_run_id "${WANDB_RUN_ID}"
    --wandb_dir "${WANDB_DIR}"
)
cmd+=("${resume_args[@]}")
printf '[flat-baseline] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
echo "[flat-baseline] global_batch=${global_batch} scope=dino_rgb_capacity_matched_unstructured_posterior_oracle_v2 require_design_pass=${REQUIRE_DESIGN_PASS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
visible_gpus="$(CUDA_VISIBLE_DEVICES="${GPU_IDS}" .venv/bin/python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_gpus}" -ne 4 ]]; then
    echo "[flat-baseline] visible GPUs=${visible_gpus}, expected 4" >&2
    exit 2
fi
mkdir -p "${OUT}" "${WANDB_DIR}"
code_files=(
    code/igsw/adaptive_gaussian_wm/action_embedding.py
    code/igsw/adaptive_gaussian_wm/change_objectives.py
    code/igsw/adaptive_gaussian_wm/checkpointing.py
    code/igsw/adaptive_gaussian_wm/experiment_tracking.py
    code/igsw/adaptive_gaussian_wm/flat_baseline_checkpointing.py
    code/igsw/adaptive_gaussian_wm/goal_eval_statistics.py
    code/igsw/adaptive_gaussian_wm/gradient_health.py
    code/igsw/adaptive_gaussian_wm/group_balanced_sampler.py
    code/igsw/adaptive_gaussian_wm/matched_flat_objective.py
    code/igsw/adaptive_gaussian_wm/matched_flat_rgb.py
    code/igsw/adaptive_gaussian_wm/matched_flat_world_model.py
    code/igsw/adaptive_gaussian_wm/metrics.py
    code/igsw/adaptive_gaussian_wm/rgb_supervision.py
    code/igsw/adaptive_gaussian_wm/scale.py
    code/igsw/adaptive_gaussian_wm/sequence_dataset.py
    code/igsw/adaptive_gaussian_wm/episode_sequence_dataset.py
    code/igsw/adaptive_gaussian_wm/train_runtime.py
    code/scripts/train_visual_sequence_flat_baseline.py
    code/scripts/train_visual_sequence_flat_baseline_4gpu.sh
)
if [[ -n "${RESUME}" ]]; then
    if [[ ! -f "${TRAINING_MANIFEST}" ]] \
        || ! sha256sum -c --status "${TRAINING_MANIFEST}"; then
        echo "[flat-baseline] resume training code identity differs" >&2
        exit 2
    fi
else
    temporary_manifest="${TRAINING_MANIFEST}.tmp.$$"
    sha256sum "${code_files[@]}" >"${temporary_manifest}"
    mv "${temporary_manifest}" "${TRAINING_MANIFEST}"
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}" \
    2>&1 | tee -a "${OUT}/train.console.log"
