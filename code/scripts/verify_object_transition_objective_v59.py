#!/usr/bin/env python3
"""Real-data GPU verifier for the v59 dynamic-objective contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from contextlib import nullcontext

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.gradient_health import (  # noqa: E402
    clip_finite_grad_norm_,
)
from igsw.adaptive_gaussian_wm.latent_object_transition_v59 import (  # noqa: E402
    QueryObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_objective_v59 import (  # noqa: E402
    object_transition_objective_v59,
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
from igsw.adaptive_gaussian_wm.v59_checkpointing import (  # noqa: E402
    load_v58_encoder,
)
from igsw.adaptive_gaussian_wm.v59_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    ObjectTransitionConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--history_lengths", default="1,2,3,4")
    parser.add_argument("--teacher_future_frames", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400,800")
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def find_probe(dataset, dino, tracker, config, args, device):
    history = max(int(value) for value in args.history_lengths.split(","))
    total = history + args.teacher_future_frames
    for source_index in range(len(dataset.source_names)):
        indices = dataset.balanced_source_evaluation_indices(source_index, 12)
        for start in range(0, len(indices), 4):
            samples = [dataset[(index, total)] for index in indices[start : start + 4]]
            batch = move_batch(default_collate(samples), device)
            features = dino(batch)
            evidence = tracker(batch, features.patches, features.grid_hw)
            relation = build_trajectory_relation_teacher_v56(
                evidence, config, batch["frame_times"]
            )
            binding = build_query_object_teacher_v57(
                evidence, relation, config, observed_frames=history
            )
            target = build_query_transition_target_v59(
                evidence,
                relation,
                binding,
                batch["frame_times"],
                history,
                config,
            )
            if bool(target.pair_valid.any()) and bool(target.motion_active.any()):
                return batch, features, binding, target, history
    raise RuntimeError("v59 verifier found no valid motion-active object transition")


def prediction_difference(first, second):
    return max(
        float((first.semantic - second.semantic).abs().max().detach()),
        float((first.geometry - second.geometry).abs().max().detach()),
        float(
            (first.visibility_logits - second.visibility_logits).abs().max().detach()
        ),
    )


def main():
    args = parse_args()
    for name in (
        "data_index",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
        "init_from",
        "output",
    ):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    for name in (
        "data_index",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
        "init_from",
    ):
        require(os.path.isfile(getattr(args, name)), f"v59 {name} is missing")
    require(torch.cuda.is_available(), "v59 verifier requires CUDA")
    config = ObjectTransitionConfig(teacher_future_frames=args.teacher_future_frames)
    config.validate()
    histories = tuple(int(value) for value in args.history_lengths.split(","))
    chunks = ",".join(str(value + args.teacher_future_frames) for value in histories)
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        chunks,
        args.temporal_step_ms,
        0,
        args.seed,
    )
    device = torch.device("cuda:0")
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    require(
        not any(parameter.requires_grad for parameter in dino.backbone.parameters()),
        "v59 DINO is not frozen",
    )
    require(
        not any(parameter.requires_grad for parameter in tracker.model.parameters()),
        "v59 point tracker is not frozen",
    )
    batch, features, binding, target, history = find_probe(
        dataset, dino, tracker, config, args, device
    )
    model = QueryObjectTransitionModel(config).to(device).train()
    init_report = load_v58_encoder(model, args.init_from)
    require(
        not any(parameter.requires_grad for parameter in model.encoder.parameters()),
        "v59 state encoder is trainable",
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    require(
        all(
            name.startswith("posterior.") or name.startswith("dynamics.")
            for name, _ in trainable
        ),
        "v59 has trainable parameters outside posterior and dynamics",
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
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
        loss, parts = object_transition_objective_v59(output, target, config)
    loss.backward()
    missing = [name for name, parameter in trainable if parameter.grad is None]
    require(not missing, f"v59 trainable parameters have no gradients: {missing}")
    grad_norm = clip_finite_grad_norm_(trainable, 5.0)
    require(float(grad_norm) > 0.0, "v59 objective produced zero gradient norm")
    require(
        all(parameter.grad is None for parameter in model.encoder.parameters()),
        "v59 frozen state encoder received gradients",
    )

    model.eval()
    swapped_target = replace(
        target,
        future_semantic=target.future_semantic.roll(1, dims=0),
        future_geometry=target.future_geometry.roll(1, dims=0),
        future_visibility=target.future_visibility.roll(1, dims=0),
    )
    with torch.no_grad(), amp_context():
        source = model.encode_source(
            features.patches[:, :history],
            features.coordinates[:, :history],
            features.valid[:, :history],
            batch["frame_times"][:, :history],
            binding.query_coordinate,
        )
        effect = model.posterior(target)
        swapped_effect = model.posterior(swapped_target)
        zero = model.dynamics(source, torch.zeros_like(effect), target.delta_seconds)
        zero_swapped = model.dynamics(
            source, torch.zeros_like(effect), swapped_target.delta_seconds
        )
        correct = model.dynamics(source, effect, target.delta_seconds)
        swapped = model.dynamics(source, swapped_effect, target.delta_seconds)
    effect_target_difference = float((effect - swapped_effect).abs().max().detach())
    zero_target_difference = prediction_difference(zero, zero_swapped)
    correct_target_difference = prediction_difference(correct, swapped)
    require(effect_target_difference > 1e-6, "v59 posterior ignores the future target")
    require(zero_target_difference < 1e-6, "v59 zero effect reads future target")
    require(
        correct_target_difference > 1e-6, "v59 Dynamics ignores the posterior effect"
    )

    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.source_revision,
        "data_index": args.data_index,
        "decode_report": args.decode_report,
        "dino_checkpoint": args.dino_checkpoint,
        "tracker_checkpoint": args.tracker_checkpoint,
        "init_from": args.init_from,
        "history_lengths": args.history_lengths,
        "teacher_future_frames": args.teacher_future_frames,
        "temporal_step_ms": args.temporal_step_ms,
        "encoder_frozen": True,
        "point_velocity_is_core_objective": False,
        "latent_effect_present": True,
        "effect_conditioned_dynamics_present": True,
        "teacher_target": "related_track_object_semantic_geometry_lifecycle",
        "student_reads_future_target": False,
        "posterior_reads_future_target": True,
        "zero_effect_reads_future_target": False,
        "effect_target_swap_max_difference": effect_target_difference,
        "zero_target_swap_max_difference": zero_target_difference,
        "correct_target_swap_max_difference": correct_target_difference,
        "motion_active_fraction": float(target.motion_active.float().mean()),
        "transition_valid_fraction": float(target.pair_valid.float().mean()),
        "objective_loss": float(loss.detach()),
        "objective_gradient_norm": float(grad_norm),
        "trainable_parameter_tensors": len(trainable),
        "v58_initialization": init_report,
        "initial_objective_metrics": {
            name: float(value.detach()) for name, value in parts.items()
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
