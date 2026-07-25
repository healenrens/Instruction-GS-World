#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${RUN_NAME}}"
STEP="${STEP:-12000}"
OUTPUT="${OUTPUT:-${RUN_DIR}/design_evidence_manifest.sha256}"
TRAINING_CODE_MANIFEST="${TRAINING_CODE_MANIFEST:-${RUN_DIR}/training_code_manifest.sha256}"
RUNTIME_CODE_MANIFEST="${RUNTIME_CODE_MANIFEST:-${RUN_DIR}/training_runtime_code_manifest.sha256}"
EVALUATION_CODE_MANIFEST="${EVALUATION_CODE_MANIFEST:-${RUN_DIR}/evaluation_code_manifest.sha256}"
CODE_PROVENANCE_EXCEPTION="${CODE_PROVENANCE_EXCEPTION:-${RUN_DIR}/code_provenance_exception.json}"
CODE_PROVENANCE_REPORT="${CODE_PROVENANCE_REPORT:-${RUN_DIR}/code_provenance_report.json}"

if [[ "${ROOT}" != /* || "${RUN_DIR}" != /* || "${OUTPUT}" != /* ]]; then
    echo "[freeze-design-evidence] all paths must be absolute" >&2
    exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "[freeze-design-evidence] refusing to overwrite ${OUTPUT}" >&2
    exit 2
fi
"${ROOT}/.venv/bin/python" \
    "${ROOT}/code/scripts/verify_visual_sequence_code_provenance.py" \
    --root "${ROOT}" \
    --training_manifest "${TRAINING_CODE_MANIFEST}" \
    --runtime_manifest "${RUNTIME_CODE_MANIFEST}" \
    --evaluation_manifest "${EVALUATION_CODE_MANIFEST}" \
    --exception "${CODE_PROVENANCE_EXCEPTION}" \
    --output "${CODE_PROVENANCE_REPORT}" >/dev/null

files=(
    "${RUN_DIR}/joint_0004000.pt"
    "${RUN_DIR}/joint_0008000.pt"
    "${RUN_DIR}/joint_$(printf '%07d' "${STEP}").pt"
    "${RUN_DIR}/train.jsonl"
    "${RUN_DIR}/warm_start_report.json"
    "${TRAINING_CODE_MANIFEST}"
    "${RUNTIME_CODE_MANIFEST}"
    "${EVALUATION_CODE_MANIFEST}"
    "${CODE_PROVENANCE_EXCEPTION}"
    "${CODE_PROVENANCE_REPORT}"
    "${ROOT}/data/rt2_visual_episodes_no_language_v1/episode_manifest.json"
    "${ROOT}/data/rt2_visual_episodes_no_language_v1/episode_manifest.verified.sha256"
    "${ROOT}/data/rt2_visual_episodes_no_language_v1/episode_source_index.json"
    "${RUN_DIR}/temporal_regions_step4000/heldseed_1024.json"
    "${RUN_DIR}/temporal_regions_step4000/heldtask_1024.json"
    "${RUN_DIR}/temporal_regions_step4000/collapse_gate.json"
    "${RUN_DIR}/temporal_regions_step8000/heldseed_1024.json"
    "${RUN_DIR}/temporal_regions_step8000/heldtask_1024.json"
    "${RUN_DIR}/temporal_regions_step8000/collapse_gate.json"
    "${RUN_DIR}/temporal_regions_step12000/heldseed_1024.json"
    "${RUN_DIR}/temporal_regions_step12000/heldtask_1024.json"
    "${RUN_DIR}/temporal_regions_step12000/collapse_gate.json"
    "${RUN_DIR}/longitudinal_gate_step12000.json"
    "${RUN_DIR}/component_ablation_step12000/heldseed_1024.json"
    "${RUN_DIR}/component_ablation_step12000/heldtask_1024.json"
    "${RUN_DIR}/component_ablation_step12000/transfer_gate.json"
    "${RUN_DIR}/representation_step12000/heldseed_144.json"
    "${RUN_DIR}/representation_step12000/heldtask_144.json"
    "${RUN_DIR}/causal_contract_step12000.json"
    "${RUN_DIR}/design_gate_step12000.json"
)
for path in "${files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[freeze-design-evidence] missing evidence: ${path}" >&2
        exit 2
    fi
done
temporary="${OUTPUT}.tmp.$$"
sha256sum "${files[@]}" >"${temporary}"
mv "${temporary}" "${OUTPUT}"
sha256sum -c --status "${OUTPUT}"
echo "[freeze-design-evidence] files=${#files[@]} manifest=${OUTPUT}"
