#!/usr/bin/env python3
"""Unseen-window evaluation of v59 effect necessity and sufficiency."""

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
from igsw.adaptive_gaussian_wm.latent_object_transition_v59 import (  # noqa: E402
    QueryObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_evaluation_v59 import (  # noqa: E402
    BASELINES,
    TransitionEvaluationAccumulator,
    bootstrap_macro_gains_v59,
    macro_metrics_v59,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.query_object_teacher_v57 import (  # noqa: E402
    build_query_object_teacher_v57,
)
from igsw.adaptive_gaussian_wm.query_transition_target_v59 import (  # noqa: E402
    build_query_transition_target_v59,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v59_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    ObjectTransitionConfig,
)
from igsw.adaptive_gaussian_wm.v59_evaluation_sampling import (  # noqa: E402
    training_base_indices_v59,
    training_exclusion_contract_v59,
    unseen_source_evaluation_indices_v59,
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
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument(
        "--wandb_mode", choices=("disabled", "online", "offline"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="object-transition-objective-v59-eval")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def nearest_temporal_step(
    seconds: torch.Tensor, temporal_steps_ms: tuple[int, ...]
) -> torch.Tensor:
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
    condition = TransitionEvaluationAccumulator(model.config.dynamic_horizons)
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
        target = build_query_transition_target_v59(
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


def gain_checks(metrics: dict[str, float], minimum: float) -> dict[str, bool]:
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
        job_type="v59-object-transition-unseen-evaluation",
        config={
            **vars(args),
            **config.to_dict(),
            "checkpoint_step": checkpoint["global_step"],
            "checkpoint_git_commit": checkpoint["git_commit"],
            "evaluation_scope": "train_split_sampler_unseen_windows",
        },
    )


def metric_payload(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def log_wandb(run, report):
    if run is None:
        return
    payload = {}
    payload.update(metric_payload("eval/macro", report["macro"]))
    payload.update(metric_payload("eval/micro", report["micro"]))
    for name, metrics in report["sources"].items():
        payload.update(metric_payload(f"eval/source/{name}", metrics))
    for name, metrics in report["histories"].items():
        payload.update(metric_payload(f"eval/history/{name}", metrics))
    for name, metrics in report["temporal_steps"].items():
        payload.update(metric_payload(f"eval/temporal/{name}", metrics))
    for name, metrics in report["conditions"].items():
        payload.update(metric_payload(f"eval/condition/{name}", metrics))
    for name, value in report["gate"].items():
        payload[f"eval/gate/{name}"] = float(value)
    for baseline, interval in report["macro_gain_bootstrap"].items():
        payload.update(metric_payload(f"eval/bootstrap/{baseline}", interval))
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
        raise ValueError("v59 evaluation requires a version-59 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v59 evaluation checkpoint architecture differs")
    if int(checkpoint["global_step"]) != args.expected_checkpoint_step:
        raise ValueError("v59 evaluation checkpoint step differs")
    config = ObjectTransitionConfig(**checkpoint["config"])
    config.validate()
    history_values = parse_ints(args.history_lengths)
    chunks = ",".join(
        str(value + args.teacher_future_frames) for value in history_values
    )
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

    device = torch.device("cuda:0")
    model = QueryObjectTransitionModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    horizons = tuple(config.dynamic_horizons)
    accumulators = {
        "micro": TransitionEvaluationAccumulator(horizons),
        "source": {
            source: TransitionEvaluationAccumulator(horizons)
            for source in dataset.source_names
        },
        "history": {
            history: TransitionEvaluationAccumulator(horizons)
            for history in history_values
        },
        "temporal": {
            milliseconds: TransitionEvaluationAccumulator(horizons)
            for milliseconds in parse_ints(args.temporal_step_ms)
        },
    }
    conditions = {}
    for source in dataset.source_names:
        for history in history_values:
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

    macro = macro_metrics_v59(list(conditions.values()))
    micro = accumulators["micro"].finalize()
    sources = {name: value.finalize() for name, value in accumulators["source"].items()}
    histories = {
        f"h{name}": value.finalize() for name, value in accumulators["history"].items()
    }
    temporal = {
        f"ms{name}": value.finalize()
        for name, value in accumulators["temporal"].items()
    }
    bootstrap = bootstrap_macro_gains_v59(
        list(conditions.values()), args.bootstrap_samples, args.seed
    )
    macro_checks = gain_checks(macro, args.minimum_gain)
    micro_checks = gain_checks(micro, args.minimum_gain)
    temporal_checks = horizon_checks(macro, horizons, args.minimum_gain)
    condition_checks = {
        name: gain_checks(metrics, args.minimum_gain)
        for name, metrics in conditions.items()
    }
    gate = {
        "training_evaluation_overlap_is_zero": exclusion[
            "training_evaluation_overlap_count"
        ]
        == 0,
        "every_condition_has_motion": all(
            metrics["motion_active_count"] > 0.0 for metrics in conditions.values()
        ),
        "macro_beats_all_baselines": macro_checks["all"],
        "micro_beats_all_baselines": micro_checks["all"],
        "macro_gain_ci_is_positive": all(
            interval["ci95_low"] > 0.0 for interval in bootstrap.values()
        ),
        "every_horizon_beats_all_baselines": all(
            checks["all"] for checks in temporal_checks.values()
        ),
    }
    aggregate_names = (
        "training_evaluation_overlap_is_zero",
        "every_condition_has_motion",
        "macro_beats_all_baselines",
        "micro_beats_all_baselines",
        "macro_gain_ci_is_positive",
    )
    report = {
        "status": "passed" if all(gate.values()) else "failed",
        "aggregate_status": "passed"
        if all(gate[name] for name in aggregate_names)
        else "failed",
        "temporal_status": "passed"
        if gate["every_horizon_beats_all_baselines"]
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
        "training_exclusion": exclusion,
        "minimum_gain": args.minimum_gain,
        "gate": gate,
        "macro_checks": macro_checks,
        "micro_checks": micro_checks,
        "horizon_checks": temporal_checks,
        "condition_checks": condition_checks,
        "macro_gain_bootstrap": bootstrap,
        "macro": macro,
        "micro": micro,
        "sources": sources,
        "histories": histories,
        "temporal_steps": temporal,
        "conditions": conditions,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run = init_wandb(args, config, checkpoint)
    log_wandb(run, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
