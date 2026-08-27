#!/usr/bin/env python3
"""Evaluate V60 compact states against the real future DINO observation field."""

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

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.gated_residual_object_transition_v60 import (  # noqa: E402
    GatedResidualObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_observation_evaluation_v60 import (  # noqa: E402
    ObjectObservationAccumulator,
    build_object_observation_target_v60,
    macro_observation_metrics,
    observation_reconstruction_statistics_v60,
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
    parser.add_argument("--support_sigma", type=float, default=0.10)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument(
        "--wandb_mode", choices=("disabled", "online", "offline"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="v60-observation-reconstruction-eval")
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


def update_accumulators(accumulators, statistics):
    for accumulator in accumulators:
        accumulator.update(statistics)


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
    condition = ObjectObservationAccumulator()
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
        observation = build_object_observation_target_v60(
            features,
            evidence,
            relation,
            binding,
            target,
            history,
            tuple(model.config.dynamic_horizons),
            args.support_sigma,
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
        statistics = observation_reconstruction_statistics_v60(
            output, target, observation, args.support_sigma
        )
        update_accumulators(
            (
                condition,
                accumulators["micro"],
                accumulators["source"][source],
                accumulators["history"][history],
            ),
            statistics,
        )
        buckets = nearest_temporal_step(
            batch["temporal_step_seconds"], temporal_steps_ms
        )
        for milliseconds in temporal_steps_ms:
            mask = buckets == milliseconds
            if bool(mask.any()):
                temporal_statistics = observation_reconstruction_statistics_v60(
                    output, target, observation, args.support_sigma, mask
                )
                accumulators["temporal"][milliseconds].update(temporal_statistics)
    return condition.finalize()


def bootstrap_macro(records, metrics, samples, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(records), (samples, len(records)), generator=generator)
    result = {}
    for name in metrics:
        values = torch.tensor([record[name] for record in records])
        draws = values[indices].mean(dim=1)
        result[name] = {
            "estimate": float(values.mean()),
            "ci95_low": float(torch.quantile(draws, 0.025)),
            "ci95_high": float(torch.quantile(draws, 0.975)),
        }
    return result


def metric_payload(prefix, metrics):
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


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
        job_type="v60-observation-reconstruction-evaluation",
        config={
            **vars(args),
            **config.to_dict(),
            "checkpoint_step": checkpoint["global_step"],
            "checkpoint_git_commit": checkpoint["git_commit"],
            "evaluation_scope": "train_split_sampler_unseen_real_future_observation",
            "observation_gt": "frozen_dinov2_patch_field_from_real_future_rgb",
        },
    )


def log_condition(run, name, metrics, completed, total):
    if run is None:
        return
    payload = metric_payload(f"reeval/condition/{name}", metrics)
    payload["reeval/progress/completed_conditions"] = float(completed)
    payload["reeval/progress/total_conditions"] = float(total)
    run.log(payload, step=completed)


def log_report(run, report):
    if run is None:
        return
    payload = {}
    for prefix, metrics in (("macro", report["macro"]), ("micro", report["micro"])):
        payload.update(metric_payload(f"reeval/{prefix}", metrics))
    for section, prefix in (
        ("sources", "source"),
        ("histories", "history"),
        ("temporal_steps", "temporal"),
    ):
        for name, metrics in report[section].items():
            payload.update(metric_payload(f"reeval/{prefix}/{name}", metrics))
    for name, interval in report["bootstrap"].items():
        payload.update(metric_payload(f"reeval/bootstrap/{name}", interval))
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
        raise ValueError("observation re-evaluation requires a version-60 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("observation re-evaluation checkpoint architecture differs")
    if int(checkpoint["global_step"]) != args.expected_checkpoint_step:
        raise ValueError("observation re-evaluation checkpoint step differs")
    config = GatedResidualTransitionConfig(**checkpoint["config"])
    config.validate()
    histories = parse_ints(args.history_lengths)
    temporal_steps = parse_ints(args.temporal_step_ms)
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
    accumulators = {
        "micro": ObjectObservationAccumulator(),
        "source": {
            name: ObjectObservationAccumulator() for name in dataset.source_names
        },
        "history": {value: ObjectObservationAccumulator() for value in histories},
        "temporal": {value: ObjectObservationAccumulator() for value in temporal_steps},
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
    report = {
        "status": "completed",
        "contract": "v60_real_future_object_observation_reconstruction_v1",
        "evaluator_revision": args.evaluator_revision,
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint["global_step"]),
        "observation_gt": "frozen_dinov2_patch_field_from_real_future_rgb",
        "rgb_pixel_reconstruction": "not_available_in_v60_state_contract",
        "interpretation": {
            "actual_support_oracle": "semantic compression floor",
            "teacher_state_oracle": "semantic plus five-number geometry state ceiling",
            "correct": "complete V60 state and Dynamics prediction",
            "persistence": "copy source teacher object state",
        },
        "training_exclusion": exclusion,
        "conditions": conditions,
        "macro": macro_observation_metrics(records),
        "micro": accumulators["micro"].finalize(),
        "sources": {
            name: value.finalize() for name, value in accumulators["source"].items()
        },
        "histories": {
            f"h{name}": value.finalize()
            for name, value in accumulators["history"].items()
        },
        "temporal_steps": {
            f"ms{name}": value.finalize()
            for name, value in accumulators["temporal"].items()
        },
    }
    report["bootstrap"] = bootstrap_macro(
        records,
        (
            "semantic_compression_floor",
            "teacher_moment_support_iou",
            "direct_observation_floor",
            "teacher_state_observation_error",
            "geometry_state_penalty",
            "dynamics_observation_gap",
            "prediction_gain_over_persistence",
            "teacher_state_gain_over_persistence",
        ),
        args.bootstrap_samples,
        args.seed,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    log_report(run, report)


if __name__ == "__main__":
    main()
