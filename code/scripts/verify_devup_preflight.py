"""Validate the complete posterior-core evidence bundle before DevUp."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

REQUIRED_CHECKS = frozenset(
    {
        "action_transfer_gate",
        "causal_checkpoint_identity",
        "causal_contract",
        "collapse_repair_gate",
        "longitudinal_checkpoint_provenance",
        "longitudinal_stability_gate",
        "representation_data_provenance",
        "sequence_data_manifest_identity",
        "sequence_data_provenance",
        "sequence_manifest_verified",
        "sequence_split_isolation",
        "single_checkpoint_provenance",
        "transfer_checkpoint_provenance",
        "transfer_data_provenance",
        *(
            f"{split}_{name}"
            for split in ("heldseed", "heldtask")
            for name in (
                "canonical_anchor_contributes",
                "canonical_anchor_standalone",
                "component_report",
                "feature_representation",
                "full_action_beats_baselines",
                "full_action_changes_observed_regions",
                "learned_residual_contributes",
                "longest_horizon_uses_action",
                "representation_report",
                "rgb_representation",
                "task_group_consistency",
            )
        ),
    }
)
REQUIRED_SCALE_CHECKS = frozenset(
    {
        "capacity_matched_unstructured_baseline",
        "checkpoint_hash_contract",
        "checkpoint_step_identity",
        "continuous_matched_action_contract",
        "cross_split_identity",
        "shared_dynamics_and_conservative_flat_budget",
        *(
            f"{split}.{name}"
            for split in ("heldseed", "heldtask")
            for name in (
                "episode_clusters",
                "flat_action_noncollapsed",
                "flat_causal_bottleneck",
                "flat_posterior_uses_future_feature",
                "flat_posterior_uses_future_rgb",
                "flat_posterior_uses_future_on_observed_change_rgb",
                "future_query_contract",
                "history_flat_beats_copy_feature",
                "history_flat_beats_copy_rgb",
                "object_beats_flat_longest_horizon_feature",
                "object_beats_flat_longest_horizon_rgb",
                "object_beats_flat_on_change_feature",
                "object_beats_flat_on_change_rgb",
                "object_beats_flat_on_observed_change_rgb",
                "object_static_rgb_noninferior",
                "object_whole_feature_noninferior",
                "object_whole_rgb_noninferior",
                "report_status",
                "rgb_region_contract",
                "sample_coverage",
                "task_group_contract",
                "object_beats_flat_by_task_on_change_feature",
                "object_beats_flat_by_task_on_change_rgb",
                "object_beats_flat_by_task_on_observed_change_rgb",
                "object_static_rgb_task_noninferior",
            )
        ),
    }
)


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_checks_pass(checks: dict) -> bool:
    return bool(checks) and all(value is True for value in checks.values())


def verified_sha_manifest(path: str) -> dict[str, str]:
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            digest, target = line.rstrip("\n").split(maxsplit=1)
            target = target.lstrip("*")
            if not os.path.isabs(target):
                raise ValueError("scale evidence manifest paths must be absolute")
            if target in records:
                raise ValueError(f"duplicate scale evidence path: {target}")
            if file_sha256(target) != digest:
                raise ValueError(f"scale evidence digest differs: {target}")
            records[target] = digest
    if not records:
        raise ValueError("scale evidence manifest is empty")
    return records


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scale_gate", required=True)
    parser.add_argument("--scale_manifest", required=True)
    parser.add_argument("--flat_checkpoint", required=True)
    args = parser.parse_args()
    gate_path = os.path.abspath(args.gate)
    source = os.path.abspath(args.source)
    data = os.path.abspath(args.data)
    scale_gate_path = os.path.abspath(args.scale_gate)
    scale_manifest_path = os.path.abspath(args.scale_manifest)
    flat_checkpoint = os.path.abspath(args.flat_checkpoint)
    if not os.path.isfile(gate_path):
        raise FileNotFoundError(gate_path)
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    if not os.path.isdir(data):
        raise FileNotFoundError(data)
    for path in (scale_gate_path, scale_manifest_path, flat_checkpoint):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    manifest_path = os.path.join(data, "episode_manifest.json")
    verified_path = os.path.join(data, "episode_manifest.verified.sha256")
    source_index_path = os.path.join(data, "episode_source_index.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    if not os.path.isfile(verified_path):
        raise FileNotFoundError(verified_path)
    if not os.path.isfile(source_index_path):
        raise FileNotFoundError(source_index_path)
    manifest_sha256 = file_sha256(manifest_path)
    with open(verified_path, encoding="utf-8") as handle:
        verified_sha256 = handle.read().split()[0]

    report = load_report(gate_path)
    scale_gate = load_report(scale_gate_path)
    scale_records = verified_sha_manifest(scale_manifest_path)
    failures: list[str] = []
    require(report.get("status") == "pass", "design gate status", failures)
    require(
        report.get("scope") == "future_conditioned_posterior_core_only",
        "design gate scope",
        failures,
    )
    require(
        report.get("deployable_world_model_proven") is False,
        "deployability boundary",
        failures,
    )
    checks = report.get("checks", {})
    require(set(checks) == REQUIRED_CHECKS, "exact design check set", failures)
    require(all_checks_pass(checks), "all design checks", failures)
    require(report.get("checkpoint") == [source], "single source checkpoint", failures)
    scale_identity = scale_gate.get("evidence_identity", {})
    scale_checks = scale_gate.get("checks", [])
    scale_check_names = {
        check.get("name")
        for check in scale_checks
        if isinstance(check, dict)
    }
    require(scale_gate.get("status") == "pass", "scale gate status", failures)
    require(
        scale_gate.get("gate") == "visual_sequence_object_vs_matched_flat_v2",
        "scale gate contract",
        failures,
    )
    require(scale_gate.get("required_step") == 12000, "scale gate step", failures)
    require(
        bool(scale_checks)
        and all(check.get("passed") is True for check in scale_checks)
        and not scale_gate.get("failed_checks"),
        "all scale checks",
        failures,
    )
    require(
        len(scale_check_names) == len(scale_checks)
        and scale_check_names == REQUIRED_SCALE_CHECKS,
        "required scale checks",
        failures,
    )
    require(
        os.path.abspath(scale_identity.get("object_checkpoint", "")) == source,
        "scale object checkpoint",
        failures,
    )
    require(
        scale_identity.get("object_checkpoint_sha256") == file_sha256(source),
        "scale object checkpoint identity",
        failures,
    )
    require(
        os.path.abspath(scale_identity.get("flat_checkpoint", ""))
        == flat_checkpoint,
        "scale flat checkpoint",
        failures,
    )
    require(
        scale_identity.get("flat_checkpoint_sha256")
        == scale_records.get(flat_checkpoint),
        "scale flat checkpoint identity",
        failures,
    )
    require(
        os.path.abspath(scale_identity.get("data", "")) == data
        and scale_identity.get("data_sha256") == manifest_sha256,
        "scale data identity",
        failures,
    )
    require(
        scale_identity.get("task_source_index_sha256")
        == scale_records.get(source_index_path)
        == file_sha256(source_index_path),
        "scale task source index identity",
        failures,
    )
    action_contract = scale_identity.get("action_contract", {})
    require(
        action_contract.get("type") == "continuous"
        and action_contract.get("tokens") == 16
        and action_contract.get("dimensions") == 14
        and action_contract.get("state_tokens") == 16
        and action_contract.get("layout")
        == "dino_effect_3_plus_rgb_logit_effect_3_plus_residual_8"
        and action_contract.get("object_source")
        == "future_conditioned_object_posterior_oracle"
        and action_contract.get("flat_source")
        == "future_conditioned_unstructured_latent_posterior_oracle"
        and action_contract.get("dynamics_future_access") == "latent_action_only"
        and action_contract.get("flat_posterior_future_modalities")
        == ["dino", "rgb"],
        "scale continuous action contract",
        failures,
    )
    training_contract = scale_identity.get("training_contract", {})
    require(
        training_contract.get("optimizer_contract")
        == "posterior_core_matched_flat_optimization_v1"
        and training_contract.get("object_checkpoint_phase_steps") == 12000
        and training_contract.get("flat_additional_steps") == 12000
        and training_contract.get("object_effective_global_batch") == 256
        and training_contract.get("flat_effective_global_batch") == 256
        and training_contract.get("shared_dynamics_initialization")
        == "object_dynamics_blocks_only"
        and training_contract.get("optimization_budget_bias")
        == "flat_receives_additional_updates_after_object_source"
        and training_contract.get("modality_matching") == "dino_rgb"
        and training_contract.get("flat_rgb_supervision") is True
        and training_contract.get("flat_posterior_observes_future_rgb") is True
        and training_contract.get("history_encoder_input") == "dino_only",
        "scale training contract",
        failures,
    )
    capacity_ratio = float(
        scale_identity.get("parameter_count", {}).get(
            "flat_to_object_ratio",
            0.0,
        )
    )
    require(
        0.75 <= capacity_ratio <= 1.25,
        "scale capacity match",
        failures,
    )
    require(
        {source, flat_checkpoint, scale_gate_path}.issubset(scale_records),
        "scale evidence manifest coverage",
        failures,
    )
    data_contracts = report.get("data_contracts", {})
    sequence_contract = data_contracts.get("sequence", {})
    require(set(data_contracts) == {"sequence"}, "sequence-only data contract", failures)
    require(
        os.path.abspath(sequence_contract.get("root", "")) == data,
        "sequence contract root",
        failures,
    )
    require(
        sequence_contract.get("manifest_sha256")
        == manifest_sha256
        == verified_sha256,
        "sequence manifest identity",
        failures,
    )
    require(
        sequence_contract.get("complete") is True
        and sequence_contract.get("episodes") == 7007,
        "sequence manifest completeness",
        failures,
    )
    file_inventory = sequence_contract.get("file_inventory", {})
    require(
        file_inventory.get("valid") is True
        and all(
            value is True
            for value in file_inventory.get("checks", {}).values()
        ),
        "sequence file inventory",
        failures,
    )
    split_isolation = sequence_contract.get("split_isolation", {})
    require(
        split_isolation.get("valid") is True
        and all(
            value is True
            for value in split_isolation.get("checks", {}).values()
        ),
        "sequence split isolation",
        failures,
    )
    expected_step = 12000
    expected_phase = "joint"

    collapse = report.get("collapse_gate", {})
    require(collapse.get("status") == "pass", "collapse gate status", failures)
    require(all_checks_pass(collapse.get("checks", {})), "collapse checks", failures)
    for split in ("heldseed", "heldtask"):
        evaluation = collapse.get("evaluation", {}).get(split, {})
        require(
            os.path.abspath(evaluation.get("checkpoint", "")) == source,
            f"collapse {split} checkpoint",
            failures,
        )
        require(
            os.path.abspath(evaluation.get("data", "")) == data,
            f"collapse {split} data",
            failures,
        )
        require(
            evaluation.get("data_sha256") == manifest_sha256,
            f"collapse {split} data identity",
            failures,
        )
        require(
            evaluation.get("checkpoint_global_step") == expected_step
            and evaluation.get("checkpoint_phase") == expected_phase,
            f"collapse {split} checkpoint metadata",
            failures,
        )

    longitudinal = report.get("longitudinal_gate", {})
    require(longitudinal.get("status") == "pass", "longitudinal status", failures)
    require(
        longitudinal.get("scope")
        == "posterior_core_stability_steps_4000_8000_12000",
        "longitudinal scope",
        failures,
    )
    require(
        os.path.abspath(longitudinal.get("final_checkpoint", "")) == source,
        "longitudinal final checkpoint",
        failures,
    )
    require(
        all_checks_pass(longitudinal.get("checks", {})),
        "longitudinal checks",
        failures,
    )
    require(
        set(longitudinal.get("steps", {})) == {"4000", "8000", "12000"},
        "longitudinal steps",
        failures,
    )
    require(
        set(longitudinal.get("source_reports", {}))
        == {"4000", "8000", "12000"}
        and all(
            os.path.isfile(os.path.abspath(path))
            for path in longitudinal.get("source_reports", {}).values()
        ),
        "longitudinal source reports",
        failures,
    )

    transfer = report.get("action_transfer_gate", {})
    require(transfer.get("status") == "pass", "transfer status", failures)
    require(
        transfer.get("scope") == "oracle_matched_effect_cross_episode_transfer",
        "transfer scope",
        failures,
    )
    require(
        transfer.get("deployable_prediction_proven") is False,
        "transfer deployability boundary",
        failures,
    )
    require(all_checks_pass(transfer.get("checks", {})), "transfer checks", failures)
    for split in ("heldseed", "heldtask"):
        summary = transfer.get("summary", {}).get(split, {})
        require(
            os.path.abspath(summary.get("checkpoint", "")) == source,
            f"transfer {split} checkpoint",
            failures,
        )
        require(
            os.path.abspath(summary.get("data", "")) == data,
            f"transfer {split} data",
            failures,
        )
        require(
            summary.get("data_sha256") == manifest_sha256,
            f"transfer {split} data identity",
            failures,
        )
        require(
            summary.get("checkpoint_global_step") == expected_step
            and summary.get("checkpoint_phase") == expected_phase,
            f"transfer {split} checkpoint metadata",
            failures,
        )

    causal = report.get("causal_contract", {})
    require(causal.get("status") == "passed", "causal status", failures)
    require(bool(causal.get("model", {}).get("passed")), "causal model", failures)
    require(
        os.path.abspath(causal.get("checkpoint", "")) == source,
        "causal checkpoint",
        failures,
    )
    require(
        os.path.abspath(causal.get("data_root", "")) == data,
        "causal data",
        failures,
    )
    for family in ("representation", "components"):
        for split in ("heldseed", "heldtask"):
            summary = report.get(family, {}).get(split, {})
            require(
                os.path.abspath(summary.get("checkpoint", "")) == source,
                f"{family} {split} checkpoint",
                failures,
            )
            require(
                os.path.abspath(summary.get("data", "")) == data,
                f"{family} {split} data",
                failures,
            )
            require(
                summary.get("data_sha256") == manifest_sha256,
                f"{family} {split} data identity",
                failures,
            )
            require(
                summary.get("checkpoint_global_step") == expected_step
                and summary.get("checkpoint_phase") == expected_phase,
                f"{family} {split} checkpoint metadata",
                failures,
            )

    if failures:
        raise AssertionError("DevUp preflight failed: " + ", ".join(failures))
    result = {
        "status": "pass",
        "scope": report["scope"],
        "deployable_world_model_proven": False,
        "gate": gate_path,
        "source": source,
        "data": data,
        "data_sha256": manifest_sha256,
        "design_checks": len(checks),
        "longitudinal_steps": [4000, 8000, 12000],
        "transfer_scope": transfer["scope"],
        "scale_gate": scale_gate_path,
        "scale_manifest": scale_manifest_path,
        "flat_checkpoint": flat_checkpoint,
        "scale_checks": len(scale_checks),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
