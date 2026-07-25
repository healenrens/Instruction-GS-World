#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
OBJECT_RUN="${OBJECT_RUN:-${ROOT}/outputs/visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
FLAT_RUN="${FLAT_RUN:-${ROOT}/outputs/visual_sequence_matched_flat_h4q4_dino_rgb_4gpu_v2_seed17_20260724}"
DATA="${DATA:-${ROOT}/data/rt2_visual_episodes_no_language_v1}"
EVAL_DIR="${EVAL_DIR:-${FLAT_RUN}/object_flat_evaluation_step12000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-604800}"
DESIGN_GATE="${OBJECT_RUN}/design_gate_step12000.json"
DESIGN_MANIFEST="${OBJECT_RUN}/design_evidence_manifest.sha256"
OBJECT_CHECKPOINT="${OBJECT_RUN}/joint_0012000.pt"
FLAT_CHECKPOINT="${FLAT_RUN}/flat_suite_0012000.pt"

json_status() {
    "${ROOT}/.venv/bin/python" - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("status", "missing"))
PY
}

for path in "${ROOT}" "${OBJECT_RUN}" "${FLAT_RUN}" "${DATA}" "${EVAL_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "[flat-queue] all paths must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ "${POLL_SECONDS}" -le 0 || "${MAX_WAIT_SECONDS}" -le 0 ]]; then
    echo "[flat-queue] wait intervals must be positive" >&2
    exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
    echo "[flat-queue] expected exactly four GPU ids" >&2
    exit 2
fi
started="$(date +%s)"
while true; do
    if [[ -f "${DESIGN_GATE}" ]]; then
        gate_status="$(json_status "${DESIGN_GATE}")"
        if [[ "${gate_status}" != "pass" && "${gate_status}" != "fail" ]]; then
            echo "[flat-queue] object design gate has invalid status" >&2
            exit 2
        fi
    fi
    if [[ -f "${OBJECT_CHECKPOINT}" && -f "${DESIGN_GATE}" ]]; then
        if [[ "${gate_status}" == "pass" ]]; then
            if [[ ! -f "${DESIGN_MANIFEST}" ]]; then
                echo "[flat-queue] waiting for passed design evidence manifest"
            elif ! sha256sum -c --status "${DESIGN_MANIFEST}"; then
                echo "[flat-queue] object design evidence manifest is invalid" >&2
                exit 2
            else
                break
            fi
        else
            echo "[flat-queue] object design gate failed; running diagnostic baseline"
            break
        fi
    fi
    elapsed=$(( $(date +%s) - started ))
    if [[ "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
        echo "[flat-queue] timed out waiting for the object design evidence" >&2
        exit 2
    fi
    echo "[flat-queue] waiting for 12k object design evidence elapsed=${elapsed}s"
    sleep "${POLL_SECONDS}"
done

sync_marker="${OBJECT_RUN}/wandb_sync_complete.txt"
sync_log="${OBJECT_RUN}/wandb_sync.log"
if [[ ! -f "${sync_marker}" ]]; then
    mapfile -t offline_runs < <(
        find "${OBJECT_RUN}/wandb" -maxdepth 3 -type d \
            -name 'offline-run-*' -print | sort
    )
    if [[ "${#offline_runs[@]}" -ne 1 ]]; then
        echo "[flat-queue] expected one completed offline W&B run" >&2
        exit 2
    fi
    for attempt in 1 2 3 4 5; do
        if "${ROOT}/.venv/bin/wandb" sync "${offline_runs[0]}" \
            >>"${sync_log}" 2>&1; then
            date --iso-8601=seconds >"${sync_marker}"
            break
        fi
        echo "[flat-queue] W&B sync retry=${attempt}" >>"${sync_log}"
        sleep 60
    done
fi
if [[ ! -s "${sync_marker}" || ! -s "${sync_log}" ]]; then
    echo "[flat-queue] completed object W&B sync is required" >&2
    exit 2
fi
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; do
    echo "[flat-queue] waiting for all four evaluation GPUs to become idle"
    sleep "${POLL_SECONDS}"
done
if [[ ! -f "${FLAT_CHECKPOINT}" ]]; then
    resume=""
    if [[ -f "${FLAT_RUN}/latest.pt" ]]; then
        resume="${FLAT_RUN}/latest.pt"
    fi
    require_design_pass=1
    if [[ "${gate_status}" == "fail" ]]; then
        require_design_pass=0
    fi
    ROOT="${ROOT}" DATA="${DATA}" OBJECT_RUN="${OBJECT_RUN}" \
        OUT="${FLAT_RUN}" GPU_IDS="${GPU_IDS}" RESUME="${resume}" \
        REQUIRE_DESIGN_PASS="${require_design_pass}" \
        bash "${ROOT}/code/scripts/train_visual_sequence_flat_baseline_4gpu.sh"
fi
if [[ ! -f "${FLAT_CHECKPOINT}" ]]; then
    echo "[flat-queue] flat baseline did not produce its 12k checkpoint" >&2
    exit 2
fi

mkdir -p "${EVAL_DIR}"
pids=()
temporary_outputs=()
final_outputs=()
if [[ ! -f "${EVAL_DIR}/heldseed_1024.json" ]]; then
    output="${EVAL_DIR}/heldseed_1024.json"
    temporary="${output}.tmp.$$"
    rm -f "${temporary}"
    CUDA_VISIBLE_DEVICES="${gpus[0]}" "${ROOT}/.venv/bin/python" \
        "${ROOT}/code/scripts/evaluate_visual_sequence_object_flat.py" \
        --object_checkpoint "${OBJECT_CHECKPOINT}" \
        --flat_checkpoint "${FLAT_CHECKPOINT}" \
        --required_step 12000 \
        --data "${DATA}" \
        --split heldseed \
        --max_items 1024 \
        --batch 2 \
        --workers 2 \
        --device cuda \
        --amp bf16 \
        --output "${temporary}" \
        >"${EVAL_DIR}/heldseed_1024.log" 2>&1 &
    pids+=("$!")
    temporary_outputs+=("${temporary}")
    final_outputs+=("${output}")
fi
if [[ ! -f "${EVAL_DIR}/heldtask_450.json" ]]; then
    output="${EVAL_DIR}/heldtask_450.json"
    temporary="${output}.tmp.$$"
    rm -f "${temporary}"
    CUDA_VISIBLE_DEVICES="${gpus[1]}" "${ROOT}/.venv/bin/python" \
        "${ROOT}/code/scripts/evaluate_visual_sequence_object_flat.py" \
        --object_checkpoint "${OBJECT_CHECKPOINT}" \
        --flat_checkpoint "${FLAT_CHECKPOINT}" \
        --required_step 12000 \
        --data "${DATA}" \
        --split heldtask \
        --max_items 450 \
        --batch 2 \
        --workers 2 \
        --device cuda \
        --amp bf16 \
        --output "${temporary}" \
        >"${EVAL_DIR}/heldtask_450.log" 2>&1 &
    pids+=("$!")
    temporary_outputs+=("${temporary}")
    final_outputs+=("${output}")
fi
status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}" \
        && [[ -s "${temporary_outputs[$index]}" ]] \
        && [[ ! -e "${final_outputs[$index]}" ]]; then
        mv "${temporary_outputs[$index]}" "${final_outputs[$index]}"
    else
        rm -f "${temporary_outputs[$index]}"
        status=1
    fi
done
if [[ "${status}" -ne 0 ]]; then
    echo "[flat-queue] object-flat evaluation failed" >&2
    exit "${status}"
fi
flat_gate="${EVAL_DIR}/object_flat_gate.json"
if [[ ! -f "${flat_gate}" ]]; then
    if "${ROOT}/.venv/bin/python" \
        "${ROOT}/code/scripts/verify_visual_sequence_object_flat_gate.py" \
        --heldseed "${EVAL_DIR}/heldseed_1024.json" \
        --heldtask "${EVAL_DIR}/heldtask_450.json" \
        --required_step 12000 \
        --heldseed_samples 1024 \
        --heldtask_samples 450 \
        --minimum_clusters 100 \
        --superiority 0.03 \
        --output "${flat_gate}" \
        >"${EVAL_DIR}/object_flat_gate.log" 2>&1; then
        echo "[flat-queue] object-flat gate=pass"
    else
        echo "[flat-queue] object-flat gate=fail; preserving diagnostic evidence" >&2
    fi
fi
flat_status="$(json_status "${flat_gate}")"
if [[ "${gate_status}" != "pass" || "${flat_status}" != "pass" ]]; then
    echo "[flat-queue] DevUp No-Go object=${gate_status} flat=${flat_status}" >&2
    exit 2
fi
if [[ ! -f "${FLAT_RUN}/scale_readiness_evidence_manifest.sha256" ]]; then
    ROOT="${ROOT}" OBJECT_RUN="${OBJECT_RUN}" FLAT_RUN="${FLAT_RUN}" \
        EVAL_DIR="${EVAL_DIR}" \
        bash "${ROOT}/code/scripts/freeze_visual_sequence_flat_evidence.sh"
fi
sha256sum -c --status "${FLAT_RUN}/scale_readiness_evidence_manifest.sha256"
echo "[flat-queue] scale readiness evidence is complete: ${FLAT_RUN}"
