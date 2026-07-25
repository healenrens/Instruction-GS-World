#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
DATA="${DATA:-${ROOT}/data/rt2_visual_sequences_no_language_v1}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a version-25 sequence checkpoint}"
OUT="${OUT:?set OUT to a new absolute evaluation directory}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_ITEMS="${MAX_ITEMS:-48}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-4}"

if [[ "${CHECKPOINT}" != /* || "${DATA}" != /* || "${OUT}" != /* ]]; then
    echo "[visual-sequence-eval] paths must be absolute" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" || ! -d "${DATA}" ]]; then
    echo "[visual-sequence-eval] checkpoint or data is missing" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[visual-sequence-eval] expected exactly four GPU ids" >&2
    exit 2
fi
reports=(
    "${OUT}/posterior_heldseed.json"
    "${OUT}/posterior_heldtask.json"
    "${OUT}/prior_heldseed.json"
    "${OUT}/prior_heldtask.json"
)
for report in "${reports[@]}"; do
    if [[ -e "${report}" ]]; then
        echo "[visual-sequence-eval] refusing to overwrite ${report}" >&2
        exit 2
    fi
done

cd "${ROOT}"
common=(
    --checkpoint "${CHECKPOINT}"
    --data "${DATA}"
    --max_items "${MAX_ITEMS}"
    --batch "${BATCH}"
    --workers "${WORKERS}"
    --device cuda
)
scripts=(
    "code/scripts/evaluate_visual_sequence_posterior.py"
    "code/scripts/evaluate_visual_sequence_posterior.py"
    "code/scripts/evaluate_visual_sequence_prior.py"
    "code/scripts/evaluate_visual_sequence_prior.py"
)
splits=(heldseed heldtask heldseed heldtask)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    for index in 0 1 2 3; do
        printf "[visual-sequence-eval] gpu=%q command=.venv/bin/python %q" \
            "${gpus[$index]}" "${scripts[$index]}"
        printf " %q" "${common[@]}" --split "${splits[$index]}" \
            --output "${reports[$index]}"
        if [[ "${index}" -ge 2 ]]; then
            printf " %q %q" --prior_samples "${PRIOR_SAMPLES}"
        fi
        printf "\n"
    done
    exit 0
fi

mkdir -p "${OUT}"
pids=()
for index in 0 1 2 3; do
    extra=()
    if [[ "${index}" -ge 2 ]]; then
        extra=(--prior_samples "${PRIOR_SAMPLES}")
    fi
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" .venv/bin/python "${scripts[$index]}" \
        "${common[@]}" \
        --split "${splits[$index]}" \
        --output "${reports[$index]}" \
        "${extra[@]}" \
        >"${OUT}/eval_${index}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if [[ "${status}" -ne 0 ]]; then
    echo "[visual-sequence-eval] at least one evaluation failed" >&2
    exit "${status}"
fi
echo "[visual-sequence-eval] complete: ${OUT}"
