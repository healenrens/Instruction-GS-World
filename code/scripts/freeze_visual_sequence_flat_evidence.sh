#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
OBJECT_RUN="${OBJECT_RUN:-${ROOT}/outputs/visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
FLAT_RUN="${FLAT_RUN:-${ROOT}/outputs/visual_sequence_matched_flat_h4q4_dino_rgb_4gpu_v2_seed17_20260724}"
EVAL_DIR="${EVAL_DIR:-${FLAT_RUN}/object_flat_evaluation_step12000}"
OUTPUT="${OUTPUT:-${FLAT_RUN}/scale_readiness_evidence_manifest.sha256}"

for path in "${ROOT}" "${OBJECT_RUN}" "${FLAT_RUN}" "${EVAL_DIR}" "${OUTPUT}"; do
    if [[ "${path}" != /* ]]; then
        echo "[freeze-flat-evidence] all paths must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${OUTPUT}" ]]; then
    echo "[freeze-flat-evidence] refusing to overwrite ${OUTPUT}" >&2
    exit 2
fi
if ! sha256sum -c --status "${OBJECT_RUN}/design_evidence_manifest.sha256"; then
    echo "[freeze-flat-evidence] object design evidence is invalid" >&2
    exit 2
fi
"${ROOT}/.venv/bin/python" - "${EVAL_DIR}/object_flat_gate.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("status") != "pass":
    raise SystemExit("object-flat gate is not pass")
PY
files=(
    "${OBJECT_RUN}/joint_0012000.pt"
    "${OBJECT_RUN}/design_evidence_manifest.sha256"
    "${OBJECT_RUN}/evaluation_code_manifest.sha256"
    "${OBJECT_RUN}/wandb_sync_complete.txt"
    "${OBJECT_RUN}/wandb_sync.log"
    "${FLAT_RUN}/flat_suite_0012000.pt"
    "${FLAT_RUN}/train.jsonl"
    "${FLAT_RUN}/initialization_report.json"
    "${FLAT_RUN}/training_code_manifest.sha256"
    "${EVAL_DIR}/heldseed_1024.json"
    "${EVAL_DIR}/heldtask_450.json"
    "${EVAL_DIR}/object_flat_gate.json"
    "${ROOT}/data/rt2_visual_episodes_no_language_v1/episode_manifest.json"
    "${ROOT}/data/rt2_visual_episodes_no_language_v1/episode_manifest.verified.sha256"
    "${ROOT}/data/rt2_visual_episodes_no_language_v1/episode_source_index.json"
)
for path in "${files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[freeze-flat-evidence] missing evidence: ${path}" >&2
        exit 2
    fi
done
temporary="${OUTPUT}.tmp.$$"
sha256sum "${files[@]}" >"${temporary}"
mv "${temporary}" "${OUTPUT}"
sha256sum -c --status "${OUTPUT}"
echo "[freeze-flat-evidence] files=${#files[@]} manifest=${OUTPUT}"
