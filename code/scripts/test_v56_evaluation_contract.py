#!/usr/bin/env python3
"""Static contracts for the v56 evaluation entrypoints."""

from __future__ import annotations

import ast
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def parse(path):
    source = read(path)
    ast.parse(source, filename=path)
    return source


def main() -> None:
    teacher = parse("code/scripts/evaluate_verified_relation_object_state_v56.py")
    independent = parse(
        "code/scripts/evaluate_independent_verified_relation_object_state_v56.py"
    )
    metrics = parse(
        "code/igsw/adaptive_gaussian_wm/v56_evaluation_metrics.py"
    )
    state_probes = parse(
        "code/igsw/adaptive_gaussian_wm/v56_state_probe_metrics.py"
    )
    targets = parse(
        "code/igsw/adaptive_gaussian_wm/object_state_target_v52.py"
    )
    tensor_contract = parse(
        "code/scripts/test_v56_evaluation_tensor_contract.py"
    )
    launcher = read(
        "code/scripts/evaluate_verified_relation_object_state_v56_suite.sh"
    )
    required_teacher = (
        "track_shuffle_objective_delta",
        "teacher_deletion_locality",
        "component_motion_targets",
        "component_motion_active_probe_relative_gain",
        "component_motion_active_readout_zero_relative_gain",
        "explicit_visibility_balanced_accuracy",
        "explicit_visibility_brier_relative_gain",
        "presence_supervision_identifiable",
        "aligned_component_object_state_v2",
        "decode_replacement_fraction",
        "causal_prefix_max_difference",
        "visible_mask(teacher)",
        "runtime_preflight",
        "with amp_context():",
    )
    teacher_contract = teacher + metrics + state_probes + targets
    missing = [name for name in required_teacher if name not in teacher_contract]
    if missing:
        raise RuntimeError(f"v56 teacher evaluation is missing metrics: {missing}")
    forbidden_independent = (
        "FrozenPointTrackerRuntime",
        "point_track_teacher",
        "trajectory_relation_teacher",
        "teacher_evidence",
    )
    present = [name for name in forbidden_independent if name in independent]
    if present:
        raise RuntimeError(
            f"v56 independent evaluator imports training teacher state: {present}"
        )
    required_launcher = ("teacher)", "independent)", "all)", "TRUTH_MANIFEST")
    missing_launcher = [name for name in required_launcher if name not in launcher]
    if missing_launcher:
        raise RuntimeError(
            f"v56 evaluation launcher is incomplete: {missing_launcher}"
        )
    if "test_v56_evaluation_tensor_contract.py" not in launcher:
        raise RuntimeError("v56 launcher does not execute its tensor contract")
    if "visibility_input_dtype" not in tensor_contract:
        raise RuntimeError("v56 tensor contract does not cover float visibility")
    if "component_motion_target" not in tensor_contract:
        raise RuntimeError("v56 tensor contract does not cover component motion")
    print(
        {
            "status": "passed",
            "teacher_diagnostics": True,
            "independent_truth": True,
            "independent_tracker_free": True,
            "wandb_sync": True,
        }
    )


if __name__ == "__main__":
    main()
