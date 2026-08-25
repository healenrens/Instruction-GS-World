#!/usr/bin/env python3
"""Real-data GPU verifier for v60 calibrated residual Dynamics."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from igsw.adaptive_gaussian_wm.gradient_health import (  # noqa: E402
    clip_finite_grad_norm_,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_objective_v60 import (  # noqa: E402
    object_transition_objective_v60,
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
from igsw.adaptive_gaussian_wm.v56_data_contract import (  # noqa: E402
    audit_decode_frontier,
)
from igsw.adaptive_gaussian_wm.v60_checkpointing import (  # noqa: E402
    load_v59_state_and_effect,
)
from igsw.adaptive_gaussian_wm.v60_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    GatedResidualTransitionConfig,
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
            target = build_query_transition_target_v60(
                evidence,
                relation,
                binding,
                batch["frame_times"],
                history,
                config,
            )
            strengths = target.change_strength[target.pair_valid]
            if (
                bool(target.pair_valid.any())
                and bool(target.motion_active.any())
                and strengths.numel() > 1
                and float(strengths.max() - strengths.min()) > 1e-4
            ):
                return batch, features, binding, target, history
    raise RuntimeError("v60 verifier found no varied valid object transitions")


def prediction_difference(first, second):
    return max(
        float((first.semantic - second.semantic).abs().max().detach()),
        float((first.geometry - second.geometry).abs().max().detach()),
        float(
            (first.visibility_logits - second.visibility_logits).abs().max().detach()
        ),
    )


def transition_distance(prediction, base):
    return (
        (
            1.0
            - torch.nn.functional.cosine_similarity(
                prediction.semantic.float(), base.semantic.float(), dim=-1
            )
        ).mean()
        + (prediction.geometry.float() - base.geometry.float()).abs().mean()
        + (prediction.visibility_logits.float() - base.visibility_logits.float())
        .abs()
        .mean()
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
        require(os.path.isfile(getattr(args, name)), f"v60 {name} is missing")
    require(torch.cuda.is_available(), "v60 verifier requires CUDA")
    config = GatedResidualTransitionConfig(
        teacher_future_frames=args.teacher_future_frames
    )
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
    decode = audit_decode_frontier(
        args.decode_report, args.data_index, args.seed, dataset.source_names
    )
    require(decode["status"] == "passed", "v60 decode frontier differs")
    device = torch.device("cuda:0")
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    require(
        not any(parameter.requires_grad for parameter in dino.backbone.parameters()),
        "v60 DINO is not frozen",
    )
    require(
        not any(parameter.requires_grad for parameter in tracker.model.parameters()),
        "v60 point tracker is not frozen",
    )
    batch, features, binding, target, history = find_probe(
        dataset, dino, tracker, config, args, device
    )
    model = GatedResidualObjectTransitionModel(config).to(device).train()
    init_report = load_v59_state_and_effect(model, args.init_from)
    require(
        not any(parameter.requires_grad for parameter in model.encoder.parameters()),
        "v60 state encoder is trainable",
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    allowed = ("posterior.", "change_gate.", "dynamics.")
    require(
        all(name.startswith(allowed) for name, _ in trainable),
        "v60 has trainable parameters outside posterior, gate, and Dynamics",
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
        loss, parts = object_transition_objective_v60(output, target, config)
    loss.backward()
    missing = [name for name, parameter in trainable if parameter.grad is None]
    require(not missing, f"v60 trainable parameters have no gradients: {missing}")
    grad_norm = clip_finite_grad_norm_(trainable, 5.0)
    require(float(grad_norm) > 0.0, "v60 objective produced zero gradient norm")
    require(
        all(parameter.grad is None for parameter in model.encoder.parameters()),
        "v60 frozen state encoder received gradients",
    )

    model.eval()
    swapped_target = replace(
        target,
        future_semantic=target.future_semantic.roll(1, dims=0),
        future_geometry=target.future_geometry.roll(1, dims=0),
        future_visibility=target.future_visibility.roll(1, dims=0),
        change_distance=target.change_distance.roll(1, dims=0),
        change_strength=target.change_strength.roll(1, dims=0),
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
        gated = model.change_gate(effect)
        swapped_gated = model.change_gate(swapped_effect)
        base, correct = model.dynamics(source, effect, target.delta_seconds, gated.gate)
        base_swapped, swapped = model.dynamics(
            source, swapped_effect, target.delta_seconds, swapped_gated.gate
        )
        _, low = model.dynamics(
            source, effect, target.delta_seconds, torch.full_like(gated.gate, 0.1)
        )
        _, high = model.dynamics(
            source, effect, target.delta_seconds, torch.full_like(gated.gate, 0.9)
        )
    effect_swap_difference = float((effect - swapped_effect).abs().max().detach())
    base_swap_difference = prediction_difference(base, base_swapped)
    correct_swap_difference = prediction_difference(
        correct.prediction, swapped.prediction
    )
    zero_base_difference = prediction_difference(output["zero"], output["base"])
    low_distance = float(transition_distance(low.prediction, base).detach())
    high_distance = float(transition_distance(high.prediction, base).detach())
    require(effect_swap_difference > 1e-6, "v60 posterior ignores future targets")
    require(base_swap_difference < 1e-6, "v60 source base reads future targets")
    require(correct_swap_difference > 1e-6, "v60 Dynamics ignores posterior effects")
    require(zero_base_difference == 0.0, "v60 zero branch is not exactly source base")
    require(high_distance > low_distance, "v60 change gate does not scale residuals")
    require(
        bool((target.change_strength >= 0.0).all())
        and bool((target.change_strength <= 1.0).all()),
        "v60 teacher change strength is outside [0,1]",
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
        "v59_posterior_loaded": init_report["v59_posterior_loaded"],
        "v59_dynamics_loaded": init_report["v59_dynamics_loaded"],
        "continuous_change_gate": True,
        "zero_branch_is_source_base": True,
        "student_reads_future_target": False,
        "posterior_reads_future_target": True,
        "source_base_reads_future_target": False,
        "effect_target_swap_max_difference": effect_swap_difference,
        "base_target_swap_max_difference": base_swap_difference,
        "correct_target_swap_max_difference": correct_swap_difference,
        "zero_base_max_difference": zero_base_difference,
        "low_gate_transition_distance": low_distance,
        "high_gate_transition_distance": high_distance,
        "change_strength_min": float(target.change_strength.min()),
        "change_strength_mean": float(target.change_strength[target.pair_valid].mean()),
        "change_strength_max": float(target.change_strength.max()),
        "motion_active_fraction": float(target.motion_active.float().mean()),
        "transition_valid_fraction": float(target.pair_valid.float().mean()),
        "objective_loss": float(loss.detach()),
        "objective_gradient_norm": float(grad_norm),
        "trainable_parameter_tensors": len(trainable),
        "v59_initialization": init_report,
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
