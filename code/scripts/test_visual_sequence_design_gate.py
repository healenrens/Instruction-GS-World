"""Remote CPU end-to-end schema contract for the posterior-core design gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.sequence_evidence import (  # noqa: E402
    manifest_split_summary,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    TASK_AGGREGATION, TASK_CONTRACT, task_source_index_sha256,
)
from verify_devup_preflight import REQUIRED_CHECKS, REQUIRED_SCALE_CHECKS
GATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "verify_visual_sequence_design_gate.py",
)
PREFLIGHT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "verify_devup_preflight.py",
)

def _paired() -> dict:
    return {
        "positive_ci95_lower": True,
        "relative_improvement": 0.10,
    }

def _write(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.write("\n")

def _representation(
    checkpoint: str,
    data: str,
    data_sha256: str,
    split: str,
) -> dict:
    return {
        "status": "ok",
        "checkpoint": checkpoint,
        "checkpoint_global_step": 12000,
        "checkpoint_phase": "joint",
        "data": data,
        "data_sha256": data_sha256,
        "split": split,
        "history_frames": 4,
        "future_frames": 4,
        "anchors": [3, 5, 8],
        "requested_max_items": 2,
        "available_samples": 2,
        "samples": 2,
        "source_clusters": 2,
        "metrics": {
            "feature_improvement_vs_global": 0.20,
            "feature_vs_global_paired": _paired(),
            "feature_slot_conditioning_paired": _paired(),
            "rgb_improvement_vs_global_color": 0.20,
            "rgb_vs_global_color_paired": _paired(),
            "rgb_slot_conditioning_paired": _paired(),
        },
    }

def _comparisons(metrics: tuple[str, ...]) -> dict:
    references = (
        "posterior_over_zero",
        "posterior_over_shuffled",
        "posterior_over_copy",
        "posterior_over_canonical",
        "posterior_over_residual",
        "canonical_over_zero",
        "residual_over_zero",
    )
    return {
        metric: {reference: _paired() for reference in references}
        for metric in metrics
    }

def _task_evidence(split: str, data: str) -> dict:
    task_count = TASK_CONTRACT[split]["task_count"]
    result = {
        "aggregation": TASK_AGGREGATION,
        "task_count": task_count,
        "positive_task_fraction": 1.0,
        "median_relative_improvement": 0.10,
    }
    return {
        "aggregation": TASK_AGGREGATION,
        "samples": task_count,
        "episodes": task_count,
        "task_count": task_count,
        "task_hashes": [f"hash-{index}" for index in range(task_count)],
        "task_names": [f"task-{index}" for index in range(task_count)],
        "source_index_sha256": task_source_index_sha256(data),
        "comparison": {
            metric: {name: dict(result) for name in values}
            for metric, values in _comparisons((
                "change_weighted_feature_mse",
                "change_weighted_rgb_charbonnier",
            )).items()
        },
    }

def _component(
    checkpoint: str,
    data: str,
    data_sha256: str,
    split: str,
) -> dict:
    task_count = TASK_CONTRACT[split]["task_count"]
    primary = _comparisons(("feature_mse", "latent_mse", "rgb_distance"))
    changed = _comparisons(
        (
            "change_weighted_feature_mse",
            "change_weighted_rgb_charbonnier",
        )
    )
    return {
        "status": "ok",
        "checkpoint": checkpoint,
        "checkpoint_global_step": 12000,
        "checkpoint_phase": "joint",
        "data": data,
        "data_sha256": data_sha256,
        "split": split,
        "requested_max_items": task_count,
        "available_samples": task_count,
        "anchors": [3, 5, 8],
        "history_frames": 4,
        "future_frames": 4,
        "action_dim": 14,
        "canonical_action_dim": 6,
        "action_residual_dim": 8,
        "evaluation": {
            "samples": task_count,
            "clusters": task_count,
            "comparison": primary,
            "change_weighted_comparison": changed,
            "task_group_evidence": _task_evidence(split, data),
            "by_future_query": {
                str(index): {"comparison": changed}
                for index in range(4)
            },
            "posterior_component_slot_rms": {
                "canonical_only": 0.01,
                "residual_only": 0.01,
            },
        },
    }


def _command(paths: dict[str, str], data: str, output: str) -> list[str]:
    command = [sys.executable, GATE]
    for name in (
        "representation_heldseed",
        "representation_heldtask",
        "component_heldseed",
        "component_heldtask",
        "collapse_gate",
        "longitudinal_gate",
        "transfer_gate",
        "causal_contract",
    ):
        command.extend((f"--{name}", paths[name]))
    command.extend(
        (
            "--sequence_data_root",
            data,
            "--minimum_representation_samples",
            "1",
            "--minimum_heldseed_dynamics_samples",
            "1",
            "--minimum_heldtask_dynamics_samples",
            "1",
            "--minimum_representation_clusters",
            "1",
            "--minimum_dynamics_clusters",
            "1",
            "--output",
            output,
        )
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_data_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = os.path.abspath(args.sequence_data_root)
    manifest = os.path.join(data, "episode_manifest.json")
    source_index = os.path.join(data, "episode_source_index.json")
    data_sha256 = hashlib.sha256(open(manifest, "rb").read()).hexdigest()
    manifest_payload = json.load(open(manifest, encoding="utf-8"))
    if not manifest_split_summary(manifest_payload)["valid"]:
        raise AssertionError("valid sequence split fixture failed")
    overlap = json.loads(json.dumps(manifest_payload))
    train_group = next(
        episode["sampling_group"]
        for episode in overlap["episodes"]
        if episode["split"] == "train"
    )
    next(
        episode for episode in overlap["episodes"]
        if episode["split"] == "heldtask"
    )["sampling_group"] = train_group
    if manifest_split_summary(overlap)["valid"]:
        raise AssertionError("held-task overlap passed split isolation")
    duplicate_cache = json.loads(json.dumps(manifest_payload))
    duplicate_cache["episodes"][1]["cache_sha256"] = (
        duplicate_cache["episodes"][0]["cache_sha256"]
    )
    if manifest_split_summary(duplicate_cache)["valid"]:
        raise AssertionError("duplicate cache payload hash passed")
    invalid_cache_size = json.loads(json.dumps(manifest_payload))
    invalid_cache_size["episodes"][0]["cache_bytes"] = 0
    if manifest_split_summary(invalid_cache_size)["valid"]:
        raise AssertionError("non-positive cache payload size passed")
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = os.path.join(temporary, "joint_0012000.pt")
        open(checkpoint, "wb").close()
        flat_checkpoint = os.path.join(temporary, "flat_suite_0012000.pt")
        open(flat_checkpoint, "wb").close()
        checkpoint_sha256 = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
        scale_gate = os.path.join(temporary, "object_flat_gate.json")
        _write(
            scale_gate,
            {
                "status": "pass",
                "gate": "visual_sequence_object_vs_matched_flat_v2",
                "required_step": 12000,
                "checks": [
                    {"name": name, "passed": True}
                    for name in sorted(REQUIRED_SCALE_CHECKS)
                ],
                "failed_checks": [],
                "evidence_identity": {
                    "object_checkpoint": checkpoint,
                    "object_checkpoint_sha256": checkpoint_sha256,
                    "flat_checkpoint": flat_checkpoint,
                    "flat_checkpoint_sha256": checkpoint_sha256,
                    "data": data,
                    "data_sha256": data_sha256,
                    "task_source_index_sha256": task_source_index_sha256(data),
                    "action_contract": {
                        "type": "continuous",
                        "tokens": 16,
                        "dimensions": 14,
                        "state_tokens": 16,
                        "layout": "dino_effect_3_plus_rgb_logit_effect_3_plus_residual_8",
                        "object_source": "future_conditioned_object_posterior_oracle",
                        "flat_source": "future_conditioned_unstructured_latent_posterior_oracle",
                        "dynamics_future_access": "latent_action_only",
                        "flat_posterior_future_modalities": ["dino", "rgb"],
                    },
                    "parameter_count": {
                        "object_model": 1000,
                        "total_parameters": 900,
                        "flat_to_object_ratio": 0.9,
                    },
                    "training_contract": {
                        "optimizer_contract": "posterior_core_matched_flat_optimization_v1",
                        "object_checkpoint_phase_steps": 12000,
                        "flat_additional_steps": 12000,
                        "object_effective_global_batch": 256,
                        "flat_effective_global_batch": 256,
                        "shared_dynamics_initialization": "object_dynamics_blocks_only",
                        "optimization_budget_bias": "flat_receives_additional_updates_after_object_source",
                        "modality_matching": "dino_rgb",
                        "flat_rgb_supervision": True,
                        "flat_posterior_observes_future_rgb": True,
                        "history_encoder_input": "dino_only",
                    },
                },
            },
        )
        scale_manifest = os.path.join(temporary, "scale_evidence.sha256")
        with open(scale_manifest, "w", encoding="utf-8") as handle:
            for path in (checkpoint, flat_checkpoint, scale_gate, source_index):
                digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
                handle.write(f"{digest}  {path}\n")
        paths = {
            name: os.path.join(temporary, f"{name}.json")
            for name in (
                "representation_heldseed",
                "representation_heldtask",
                "component_heldseed",
                "component_heldtask",
                "collapse_gate",
                "longitudinal_gate",
                "transfer_gate",
                "causal_contract",
            )
        }
        for split in ("heldseed", "heldtask"):
            _write(
                paths[f"representation_{split}"],
                _representation(checkpoint, data, data_sha256, split),
            )
            _write(
                paths[f"component_{split}"],
                _component(checkpoint, data, data_sha256, split),
            )
        split_summary = {
            split: {
                "checkpoint": checkpoint,
                "checkpoint_global_step": 12000,
                "checkpoint_phase": "joint",
                "data": data,
                "data_sha256": data_sha256,
            }
            for split in ("heldseed", "heldtask")
        }
        _write(
            paths["collapse_gate"],
            {
                "status": "pass",
                "checks": {"synthetic_collapse": True},
                "evaluation": split_summary,
            },
        )
        _write(
            paths["longitudinal_gate"],
            {
                "status": "pass",
                "scope": "posterior_core_stability_steps_4000_8000_12000",
                "final_checkpoint": checkpoint,
                "checks": {"synthetic_longitudinal": True},
                "steps": {str(step): {} for step in (4000, 8000, 12000)},
                "source_reports": {
                    str(step): paths["collapse_gate"]
                    for step in (4000, 8000, 12000)
                },
            },
        )
        _write(
            paths["transfer_gate"],
            {
                "status": "pass",
                "scope": "oracle_matched_effect_cross_episode_transfer",
                "deployable_prediction_proven": False,
                "checks": {"synthetic_transfer": True},
                "summary": split_summary,
            },
        )
        _write(
            paths["causal_contract"],
            {
                "status": "passed",
                "checkpoint": checkpoint,
                "data_root": data,
                "model": {
                    "passed": True,
                    "warm_start": {
                        "source_checkpoint_version": 27,
                        "loaded": 1,
                        "missing": [],
                        "unexpected": [],
                        "shape_mismatch": [],
                        "transformed": [],
                        "dropped": [],
                    },
                },
            },
        )
        passing = os.path.join(temporary, "passing.json")
        failing = os.path.join(temporary, "failing.json")
        subprocess.run(
            _command(paths, data, passing),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        passing_report = json.load(open(passing, encoding="utf-8"))
        if passing_report["status"] != "pass":
            raise AssertionError("valid design fixture did not pass")
        if set(passing_report["checks"]) != REQUIRED_CHECKS:
            raise AssertionError("design and DevUp check sets diverged")
        preflight = subprocess.run(
            [
                sys.executable,
                PREFLIGHT,
                "--gate",
                passing,
                "--source",
                checkpoint,
                "--data",
                data,
                "--scale_gate",
                scale_gate,
                "--scale_manifest",
                scale_manifest,
                "--flat_checkpoint",
                flat_checkpoint,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if json.loads(preflight.stdout)["status"] != "pass":
            raise AssertionError("valid DevUp fixture did not pass")

        invalid_representation = _representation(
            checkpoint, data, data_sha256, "heldseed"
        )
        invalid_representation["checkpoint_global_step"] = 11999
        invalid_representation["anchors"] = [3, 5]
        invalid_component = _component(checkpoint, data, data_sha256, "heldtask")
        invalid_component["future_frames"] = 3
        invalid_component["evaluation"]["by_future_query"].pop("0")
        invalid_component["evaluation"]["task_group_evidence"][
            "task_names"
        ] = []
        _write(paths["representation_heldtask"], invalid_representation)
        _write(paths["component_heldtask"], invalid_component)
        invalid_contract_path = os.path.join(temporary, "invalid_contract.json")
        invalid_contract = subprocess.run(
            _command(paths, data, invalid_contract_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        invalid_report = json.load(open(invalid_contract_path, encoding="utf-8"))
        expected_failures = {
            "heldtask_representation_report", "heldtask_component_report",
            "heldtask_longest_horizon_uses_action",
            "heldtask_task_group_consistency",
        }
        if invalid_contract.returncode == 0 or any(
            invalid_report["checks"][name] for name in expected_failures
        ):
            raise AssertionError("split, step, or temporal corruption passed")
        _write(
            paths["representation_heldtask"],
            _representation(checkpoint, data, data_sha256, "heldtask"),
        )
        _write(
            paths["component_heldtask"],
            _component(checkpoint, data, data_sha256, "heldtask"),
        )

        incomplete_scale = json.load(open(scale_gate, encoding="utf-8"))
        incomplete_scale["checks"] = [
            check
            for check in incomplete_scale["checks"]
            if check["name"] != "heldtask.object_static_rgb_noninferior"
        ]
        _write(scale_gate, incomplete_scale)
        with open(scale_manifest, "w", encoding="utf-8") as handle:
            for path in (checkpoint, flat_checkpoint, scale_gate, source_index):
                digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
                handle.write(f"{digest}  {path}\n")
        missing_scale_check = subprocess.run(
            preflight.args,
            capture_output=True,
            text=True,
        )
        if missing_scale_check.returncode == 0:
            raise AssertionError("missing scale-region check passed DevUp preflight")

        mismatched = _representation(checkpoint, data, "c" * 64, "heldtask")
        _write(paths["representation_heldtask"], mismatched)
        failed = subprocess.run(
            _command(paths, data, failing),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        failing_report = json.load(open(failing, encoding="utf-8"))
        if failed.returncode == 0 or failing_report["status"] != "fail":
            raise AssertionError("manifest mismatch did not fail closed")
        if failing_report["checks"]["sequence_data_manifest_identity"]:
            raise AssertionError("manifest mismatch passed identity check")

    report = {
        "status": "ok",
        "valid_design_contract_passes": True,
        "valid_devup_preflight_passes": True,
        "missing_scale_region_check_fails_closed": True,
        "manifest_mismatch_fails_closed": True,
        "split_overlap_fails_closed": True,
        "duplicate_cache_hash_fails_closed": True,
        "invalid_cache_size_fails_closed": True,
        "evaluation_identity_contract_fails_closed": True,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    _write(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
