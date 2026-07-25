#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
RUN_NAME="${RUN_NAME:-visual_sequence_collapse_repair_h4q4_nolang_posterior_4gpu_v27_seed17_20260723}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${RUN_NAME}}"
OUTPUT="${OUTPUT:-${RUN_DIR}/evaluation_code_manifest.sha256}"

if [[ "${ROOT}" != /* || "${RUN_DIR}" != /* || "${OUTPUT}" != /* ]]; then
    echo "[freeze-evaluation] all paths must be absolute" >&2
    exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "[freeze-evaluation] refusing to overwrite ${OUTPUT}" >&2
    exit 2
fi

files=(
    code/igsw/adaptive_gaussian_wm/action_embedding.py
    code/igsw/adaptive_gaussian_wm/action_evidence_evaluation.py
    code/igsw/adaptive_gaussian_wm/action_interventions.py
    code/igsw/adaptive_gaussian_wm/action_probe.py
    code/igsw/adaptive_gaussian_wm/change_objectives.py
    code/igsw/adaptive_gaussian_wm/flat_baseline_checkpointing.py
    code/igsw/adaptive_gaussian_wm/goal_eval_statistics.py
    code/igsw/adaptive_gaussian_wm/matched_flat_contract.py
    code/igsw/adaptive_gaussian_wm/matched_flat_evaluation.py
    code/igsw/adaptive_gaussian_wm/matched_flat_objective.py
    code/igsw/adaptive_gaussian_wm/matched_flat_rgb.py
    code/igsw/adaptive_gaussian_wm/matched_flat_training_contract.py
    code/igsw/adaptive_gaussian_wm/matched_flat_world_model.py
    code/igsw/adaptive_gaussian_wm/rgb_supervision.py
    code/igsw/adaptive_gaussian_wm/sequence_evidence.py
    code/igsw/adaptive_gaussian_wm/task_group_evidence.py
    code/igsw/adaptive_gaussian_wm/temporal_slot_evaluation.py
    code/scripts/action_transfer_evaluation.py
    code/scripts/evaluate_action_component_ablation.py
    code/scripts/evaluate_adaptive_gaussian_representation.py
    code/scripts/evaluate_posterior_dynamics_gate.py
    code/scripts/evaluate_visual_sequence_temporal_regions.py
    code/scripts/evaluate_visual_sequence_representation.py
    code/scripts/evaluate_visual_sequence_object_flat.py
    code/scripts/evaluate_visual_sequence_action_evidence.py
    code/scripts/evaluate_visual_sequence_action_probe.py
    code/scripts/freeze_visual_sequence_design_evidence.sh
    code/scripts/freeze_visual_sequence_evaluation.sh
    code/scripts/freeze_visual_sequence_flat_evidence.sh
    code/scripts/monitor_visual_sequence_collapse_repair.sh
    code/scripts/monitor_visual_sequence_component_ablation.sh
    code/scripts/monitor_visual_sequence_design_gate.sh
    code/scripts/monitor_visual_sequence_representation.sh
    code/scripts/monitor_visual_sequence_training_health.py
    code/scripts/prepare_effect_core_launch_contract.py
    code/scripts/queue_visual_sequence_flat_baseline_4gpu.sh
    code/scripts/supervise_visual_sequence_collapse_repair.sh
    code/scripts/test_visual_sequence_code_provenance.py
    code/scripts/task_group_test_fixtures.py
    code/scripts/test_effect_core_launch_contract.py
    code/scripts/test_matched_flat_training_contract.py
    code/scripts/test_model_evidence_metrics.py
    code/scripts/test_task_group_evidence.py
    code/scripts/test_visual_sequence_design_gate.py
    code/scripts/test_visual_sequence_longitudinal_gate.py
    code/scripts/test_visual_sequence_object_flat_contract.py
    code/scripts/train_effect_anchored_posterior_core_v27_2n8g.sh
    code/scripts/train_visual_sequence_flat_baseline.py
    code/scripts/train_visual_sequence_flat_baseline_4gpu.sh
    code/scripts/verify_rt2_causal_pairs.py
    code/scripts/verify_rt2_pair_dino.py
    code/scripts/verify_action_transfer_gate.py
    code/scripts/verify_devup_preflight.py
    code/scripts/verify_visual_sequence_collapse_gate.py
    code/scripts/verify_visual_sequence_core.py
    code/scripts/verify_visual_sequence_design_gate.py
    code/scripts/verify_visual_sequence_longitudinal_gate.py
    code/scripts/verify_visual_sequence_object_flat_gate.py
    code/scripts/verify_visual_sequence_code_provenance.py
)

cd "${ROOT}"
for path in "${files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[freeze-evaluation] missing evaluation source: ${path}" >&2
        exit 2
    fi
done
mkdir -p "$(dirname "${OUTPUT}")"
temporary="${OUTPUT}.tmp.$$"
sha256sum "${files[@]}" >"${temporary}"
mv "${temporary}" "${OUTPUT}"
echo "[freeze-evaluation] files=${#files[@]} manifest=${OUTPUT}"
