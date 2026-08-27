#!/usr/bin/env python3
"""Sampler-unseen evaluation and factorization diagnosis for v60."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.gated_residual_object_transition_v60 import (  # noqa: E402
    GatedResidualObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.gated_transition_evaluation_v60 import (  # noqa: E402
    BASELINES,
    GatedTransitionEvaluationAccumulator,
    bootstrap_gains_v60,
    bootstrap_standard_gains_v60,
    factorization_diagnosis_v60,
    macro_metrics_v60,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.query_object_teacher_v57 import (  # noqa: E402
    build_query_object_teacher_v57,
)
from igsw.adaptive_gaussian_wm.query_transition_target_v60 import (  # noqa: E402
    build_query_transition_target_v60,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v59_evaluation_sampling import (  # noqa: E402
    training_base_indices_v59,
    training_exclusion_contract_v59,
    unseen_source_evaluation_indices_v59,
)
from igsw.adaptive_gaussian_wm.v60_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    GatedResidualTransitionConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluator_revision", required=True)
    parser.add_argument("--expected_checkpoint_step", type=int, default=10000)
    parser.add_argument("--history_lengths", default="1,2,3,4")
    parser.add_argument("--teacher_future_frames", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400,800")
    parser.add_argument("--samples_per_condition", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dino_frame_batch", type=int, default=192)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--minimum_gain", type=float, default=0.10)
    parser.add_argument("--minimum_low_change_gain", type=float, default=0.0)
    parser.add_argument("--minimum_gate_correlation", type=float, default=0.80)
    parser.add_argument("--maximum_gate_mae", type=float, default=0.10)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument(
        "--wandb_mode", choices=("disabled", "online", "offline"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="gated-residual-transition-v60-eval")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def nearest_temporal_step(seconds, temporal_steps_ms):
    candidates = torch.tensor(
        temporal_steps_ms, device=seconds.device, dtype=torch.float32
    )
    distance = (seconds.float()[:, None] * 1000.0 - candidates[None]).abs()
    return candidates[distance.argmin(dim=1)].long()


@torch.no_grad()
def evaluate_condition(
    dataset,
    indices,
    source,
    history,
    args,
    model,
    dino,
    tracker,
    device,
    accumulators,
):
    condition = GatedTransitionEvaluationAccumulator(model.config.dynamic_horizons)
    total_frames = history + args.teacher_future_frames
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    temporal_steps_ms = parse_ints(args.temporal_step_ms)
    for start in range(0, len(indices), args.batch):
        samples = [
            dataset[(index, total_frames)]
            for index in indices[start : start + args.batch]
        ]
        batch = move_batch(default_collate(samples), device)
        features = dino(batch)
        evidence = tracker(batch, features.patches, features.grid_hw)
        relation = build_trajectory_relation_teacher_v56(
            evidence, model.config, batch["frame_times"]
        )
        binding = build_query_object_teacher_v57(
            evidence, relation, model.config, observed_frames=history
        )
        target = build_query_transition_target_v60(
            evidence,
            relation,
            binding,
            batch["frame_times"],
            history,
            model.config,
        )
        with amp_context():
            output = model(
                features.patches[:, :history],
                features.coordinates[:, :history],
                features.valid[:, :history],
                batch["frame_times"][:, :history],
                binding.query_coordinate,
                target,
            )
        for accumulator in (
            condition,
            accumulators["micro"],
            accumulators["source"][source],
            accumulators["history"][history],
        ):
            accumulator.update(output, target, model.config)
        buckets = nearest_temporal_step(
            batch["temporal_step_seconds"], temporal_steps_ms
        )
        for milliseconds in temporal_steps_ms:
            mask = buckets == milliseconds
            if bool(mask.any()):
                accumulators["temporal"][milliseconds].update(
                    output, target, model.config, mask
                )
    return condition.finalize()


def standard_checks(metrics, minimum):
    checks = {
        baseline: metrics[f"gain_over_{baseline}"] >= minimum for baseline in BASELINES
    }
    checks["all"] = all(checks.values())
    return checks


def horizon_checks(metrics, horizons, minimum):
    return {
        f"h{horizon}": {
            **{
                baseline: metrics[f"h{horizon}_gain_over_{baseline}"] >= minimum
                for baseline in BASELINES
            },
            "all": all(
                metrics[f"h{horizon}_gain_over_{baseline}"] >= minimum
                for baseline in BASELINES
            ),
        }
        for horizon in horizons
    }


def init_wandb(args, config, checkpoint):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        job_type="v60-sampler-unseen-factorization-evaluation",
        config={
            **vars(args),
            **config.to_dict(),
            "checkpoint_step": checkpoint["global_step"],
            "checkpoint_git_commit": checkpoint["git_commit"],
            "evaluation_scope": "train_split_sampler_unseen_windows",
        },
    )


def metric_payload(prefix, metrics):
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def log_condition(run, name, metrics, completed, total):
    if run is None:
        return
    payload = metric_payload(f"eval/condition/{name}", metrics)
    payload["eval/progress/completed_conditions"] = float(completed)
    payload["eval/progress/total_conditions"] = float(total)
    run.log(payload, step=completed)


def log_report(run, report):
    if run is None:
        return
    payload = {}
    for prefix, metrics in (
        ("eval/macro", report["macro"]),
        ("eval/micro", report["micro"]),
    ):
        payload.update(metric_payload(prefix, metrics))
    for section, prefix in (
        ("sources", "source"),
        ("histories", "history"),
        ("temporal_steps", "temporal"),
    ):
        for name, metrics in report[section].items():
            payload.update(metric_payload(f"eval/{prefix}/{name}", metrics))
    for baseline, interval in report["standard_gain_bootstrap"].items():
        payload.update(metric_payload(f"eval/bootstrap/standard/{baseline}", interval))
    for group, routes in report["factorization_bootstrap"].items():
        for route, interval in routes.items():
            payload.update(
                metric_payload(
                    f"eval/bootstrap/factorization/{group}/{route}", interval
                )
            )
    payload.update(metric_payload("eval/diagnosis", report["diagnosis"]))
    payload.update(
        {f"eval/gate/{name}": float(value) for name, value in report["gate"].items()}
    )
    run.log(payload)
    run.summary.update(report)
    run.finish()


def main():
    args = parse_args()
    for name in (
        "data_index",
        "checkpoint",
        "dino_checkpoint",
        "tracker_checkpoint",
        "output",
        "wandb_dir",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v60 evaluation requires a version-60 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v60 evaluation checkpoint architecture differs")
    if int(checkpoint["global_step"]) != args.expected_checkpoint_step:
        raise ValueError("v60 evaluation checkpoint step differs")
    config = GatedResidualTransitionConfig(**checkpoint["config"])
    config.validate()
    histories = parse_ints(args.history_lengths)
    chunks = ",".join(str(value + args.teacher_future_frames) for value in histories)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        chunks,
        args.temporal_step_ms,
        int(checkpoint["args"].get("max_train_items", 0)),
        args.seed,
    )
    excluded = training_base_indices_v59(dataset, checkpoint)
    indices_by_source = {
        source: unseen_source_evaluation_indices_v59(
            dataset, source_index, args.samples_per_condition, excluded
        )
        for source_index, source in enumerate(dataset.source_names)
    }
    selected = {index for indices in indices_by_source.values() for index in indices}
    exclusion = training_exclusion_contract_v59(checkpoint, excluded, selected)
    print(json.dumps({"training_exclusion": exclusion}, sort_keys=True), flush=True)

    device = torch.device("cuda:0")
    model = GatedResidualObjectTransitionModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    horizons = tuple(config.dynamic_horizons)
    accumulators = {
        "micro": GatedTransitionEvaluationAccumulator(horizons),
        "source": {
            source: GatedTransitionEvaluationAccumulator(horizons)
            for source in dataset.source_names
        },
        "history": {
            history: GatedTransitionEvaluationAccumulator(horizons)
            for history in histories
        },
        "temporal": {
            milliseconds: GatedTransitionEvaluationAccumulator(horizons)
            for milliseconds in parse_ints(args.temporal_step_ms)
        },
    }
    run = init_wandb(args, config, checkpoint)
    conditions = {}
    total_conditions = len(dataset.source_names) * len(histories)
    for source in dataset.source_names:
        for history in histories:
            name = f"{source}/h{history}"
            conditions[name] = evaluate_condition(
                dataset,
                indices_by_source[source],
                source,
                history,
                args,
                model,
                dino,
                tracker,
                device,
                accumulators,
            )
            print(
                json.dumps({"condition": name, **conditions[name]}, sort_keys=True),
                flush=True,
            )
            log_condition(
                run, name, conditions[name], len(conditions), total_conditions
            )

    records = list(conditions.values())
    macro = macro_metrics_v60(records)
    micro = accumulators["micro"].finalize()
    sources = {name: value.finalize() for name, value in accumulators["source"].items()}
    histories_out = {
        f"h{name}": value.finalize() for name, value in accumulators["history"].items()
    }
    temporal = {
        f"ms{name}": value.finalize()
        for name, value in accumulators["temporal"].items()
    }
    standard_bootstrap = bootstrap_standard_gains_v60(
        records, args.bootstrap_samples, args.seed
    )
    factorization_bootstrap = {
        group: bootstrap_gains_v60(records, args.bootstrap_samples, args.seed, group)
        for group in ("active", "low_change", "high_change")
    }
    macro_checks = standard_checks(macro, args.minimum_gain)
    micro_checks = standard_checks(micro, args.minimum_gain)
    temporal_checks = horizon_checks(macro, horizons, args.minimum_gain)
    gate = {
        "training_evaluation_overlap_is_zero": exclusion[
            "training_evaluation_overlap_count"
        ]
        == 0,
        "every_condition_has_motion": all(
            item["motion_active_count"] > 0.0 for item in records
        ),
        "every_condition_has_low_change": all(
            item["low_change_count"] > 0.0 for item in records
        ),
        "every_condition_has_high_change": all(
            item["high_change_count"] > 0.0 for item in records
        ),
        "macro_beats_all_baselines": macro_checks["all"],
        "micro_beats_all_baselines": micro_checks["all"],
        "standard_gain_ci_is_positive": all(
            interval["ci95_low"] > 0.0 for interval in standard_bootstrap.values()
        ),
        "every_horizon_beats_all_baselines": all(
            item["all"] for item in temporal_checks.values()
        ),
        "low_change_does_not_lose_to_persistence": macro[
            "low_change_gain_over_persistence"
        ]
        >= args.minimum_low_change_gain,
        "high_change_beats_persistence": macro["high_change_gain_over_persistence"]
        >= args.minimum_gain,
        "agibot_beats_persistence": sources["agibot"]["active_gain_over_persistence"]
        > 0.0,
        "droid_beats_persistence": sources["droid"]["active_gain_over_persistence"]
        > 0.0,
        "gate_correlation_is_calibrated": macro["change_gate_correlation"]
        >= args.minimum_gate_correlation,
        "gate_mae_is_calibrated": macro["change_gate_mae"] <= args.maximum_gate_mae,
    }
    aggregate_names = (
        "training_evaluation_overlap_is_zero",
        "every_condition_has_motion",
        "every_condition_has_low_change",
        "every_condition_has_high_change",
        "macro_beats_all_baselines",
        "micro_beats_all_baselines",
        "standard_gain_ci_is_positive",
    )
    temporal_names = ("every_horizon_beats_all_baselines",)
    calibration_names = (
        "low_change_does_not_lose_to_persistence",
        "high_change_beats_persistence",
        "agibot_beats_persistence",
        "droid_beats_persistence",
        "gate_correlation_is_calibrated",
        "gate_mae_is_calibrated",
    )
    report = {
        "status": "passed" if all(gate.values()) else "failed",
        "aggregate_status": "passed"
        if all(gate[name] for name in aggregate_names)
        else "failed",
        "temporal_status": "passed"
        if all(gate[name] for name in temporal_names)
        else "failed",
        "calibration_status": "passed"
        if all(gate[name] for name in calibration_names)
        else "failed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint["global_step"],
        "checkpoint_git_commit": checkpoint["git_commit"],
        "evaluator_revision": args.evaluator_revision,
        "teacher_dependent_objective_evaluation": True,
        "independent_object_validity_claim": False,
        "evaluation_scope": "train_split_sampler_unseen_windows",
        "factorization_scope": "additive_base_and_gate_substitution_with_student_conditioned_residual",
        "training_exclusion": exclusion,
        "minimum_gain": args.minimum_gain,
        "gate": gate,
        "diagnosis": factorization_diagnosis_v60(macro),
        "macro_checks": macro_checks,
        "micro_checks": micro_checks,
        "horizon_checks": temporal_checks,
        "standard_gain_bootstrap": standard_bootstrap,
        "factorization_bootstrap": factorization_bootstrap,
        "macro": macro,
        "micro": micro,
        "sources": sources,
        "histories": histories_out,
        "temporal_steps": temporal,
        "conditions": conditions,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    log_report(run, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
