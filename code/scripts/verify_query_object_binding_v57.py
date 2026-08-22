#!/usr/bin/env python3
"""Real-data GPU verifier for v57 causal query binding."""

from __future__ import annotations

import argparse
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
    FrozenVideoFeatures,
)
from igsw.adaptive_gaussian_wm.gradient_health import (  # noqa: E402
    clip_finite_grad_norm_,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
    PointTrackEvidence,
)
from igsw.adaptive_gaussian_wm.query_object_binding_v57 import (  # noqa: E402
    QueryObjectBindingModel,
)
from igsw.adaptive_gaussian_wm.query_object_coverage_v57 import (  # noqa: E402
    future_track_shuffle,
    select_query_teacher,
    teacher_target_difference,
)
from igsw.adaptive_gaussian_wm.query_object_teacher_v57 import (  # noqa: E402
    build_query_object_teacher_v57,
    observed_evidence_prefix,
    query_teacher_contract_metrics,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v57_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    QueryConditionedObjectStateConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--coverage_report", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--history_lengths", default="1,2,3,4")
    parser.add_argument("--teacher_future_frames", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400")
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


def select_batch(batch, index):
    size = len(batch["video_rgb"])
    return {
        name: value[index : index + 1]
        if torch.is_tensor(value) and value.ndim and len(value) == size
        else value
        for name, value in batch.items()
    }


def select_features(features, index):
    return FrozenVideoFeatures(
        patches=features.patches[index : index + 1],
        coordinates=features.coordinates[index : index + 1],
        valid=features.valid[index : index + 1],
        grid_hw=features.grid_hw,
    )


def select_evidence(evidence, index):
    return PointTrackEvidence(
        coordinates=evidence.coordinates[index : index + 1],
        visibility=evidence.visibility[index : index + 1],
        residual_flow=evidence.residual_flow[index : index + 1],
        motion_salience=evidence.motion_salience[index : index + 1],
        query_times=evidence.query_times,
        sampled_features=evidence.sampled_features[index : index + 1],
    )


def read_coverage(args):
    with open(args.coverage_report, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.source_revision,
        "data_index": args.data_index,
        "decode_report": args.decode_report,
        "dino_checkpoint": args.dino_checkpoint,
        "tracker_checkpoint": args.tracker_checkpoint,
        "history_lengths": args.history_lengths,
        "teacher_future_frames": args.teacher_future_frames,
        "temporal_step_ms": args.temporal_step_ms,
    }
    differences = {
        name: (report.get(name), value)
        for name, value in expected.items()
        if report.get(name) != value
    }
    require(not differences, f"v57 verifier coverage contract differs: {differences}")
    require(report["query_coverage"]["status"] == "passed", "v57 coverage failed")
    return report


def find_probe(dataset, dino, tracker, config, args, device):
    history = max(int(value) for value in args.history_lengths.split(","))
    total = history + args.teacher_future_frames
    for source_index in range(len(dataset.source_names)):
        indices = dataset.balanced_source_evaluation_indices(source_index, 8)
        for start in range(0, len(indices), 4):
            samples = [dataset[(index, total)] for index in indices[start : start + 4]]
            batch = move_batch(default_collate(samples), device)
            features = dino(batch)
            evidence = tracker(batch, features.patches, features.grid_hw)
            relation = build_trajectory_relation_teacher_v56(
                evidence, config, batch["frame_times"]
            )
            teacher = build_query_object_teacher_v57(
                evidence, relation, config, observed_frames=history
            )
            shuffled = future_track_shuffle(evidence, history)
            shuffled_relation = build_trajectory_relation_teacher_v56(
                shuffled, config, batch["frame_times"]
            )
            shuffled_teacher = build_query_object_teacher_v57(
                shuffled, shuffled_relation, config, observed_frames=history
            )
            usable = (
                teacher.query_valid
                & teacher.alternate_valid
                & teacher.negative_valid
                & (teacher.heldout_track_mask & (teacher.same_target > 0.0)).any(-1)
                & (teacher.heldout_track_mask & (teacher.different_target > 0.0)).any(-1)
            )
            for item in usable.nonzero(as_tuple=False).flatten().tolist():
                selected_teacher = select_query_teacher(teacher, item)
                selected_shuffled = select_query_teacher(shuffled_teacher, item)
                if teacher_target_difference(selected_teacher, selected_shuffled) <= 1e-6:
                    continue
                metrics = query_teacher_contract_metrics(selected_teacher)
                return (
                    select_batch(batch, item),
                    select_features(features, item),
                    select_evidence(evidence, item),
                    selected_teacher,
                    history,
                    {name: float(value.detach()) for name, value in metrics.items()},
                )
    raise RuntimeError("v57 coverage passed but verifier found no trainable probe")


def state_max_difference(first, second) -> float:
    tensors = (
        (first.support.float() - second.support.float()).abs().max(),
        (first.identity.float() - second.identity.float()).abs().max(),
        (first.center.float() - second.center.float()).abs().max(),
        (first.visibility.float() - second.visibility.float()).abs().max(),
    )
    return float(torch.stack(tensors).max().detach())


def verify_history_gradients(
    model,
    features,
    evidence,
    relation,
    batch,
    config,
    histories,
    grid_hw,
    amp_context,
):
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    gradient_norms = {}
    for history in histories:
        teacher = build_query_object_teacher_v57(
            evidence, relation, config, observed_frames=history
        )
        with amp_context():
            output = model(
                features.patches[:, :history],
                features.coordinates[:, :history],
                features.valid[:, :history],
                batch["frame_times"][:, :history],
                teacher,
                observed_evidence_prefix(evidence, history),
                grid_hw,
            )
        output["loss"].backward()
        missing = [name for name, parameter in trainable if parameter.grad is None]
        require(not missing, f"v57 H={history} has unused parameters: {missing}")
        grad_norm = clip_finite_grad_norm_(trainable, 5.0)
        gradient_norms[str(history)] = float(grad_norm)
        model.zero_grad(set_to_none=True)
    return trainable, gradient_norms


def main():
    args = parse_args()
    for name in (
        "data_index",
        "coverage_report",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
        "output",
    ):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    for name in (
        "data_index",
        "coverage_report",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
    ):
        require(os.path.isfile(getattr(args, name)), f"v57 {name} is missing")
    require(torch.cuda.is_available(), "v57 verifier requires CUDA")
    coverage = read_coverage(args)
    config = QueryConditionedObjectStateConfig(
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
    device = torch.device("cuda:0")
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    require(
        not any(parameter.requires_grad for parameter in dino.backbone.parameters()),
        "v57 DINO is not frozen",
    )
    require(
        not any(parameter.requires_grad for parameter in tracker.model.parameters()),
        "v57 tracker is not frozen",
    )
    batch, features, evidence, teacher, history, teacher_metrics = find_probe(
        dataset, dino, tracker, config, args, device
    )
    observed = observed_evidence_prefix(evidence, history)
    model = QueryObjectBindingModel(config).to(device).train()
    require(
        not any(parameter.requires_grad for parameter in model.encoder.dynamic.parameters()),
        "v57 unsupervised dynamic head is trainable",
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
            teacher,
            observed,
            features.grid_hw,
        )
    trainable, history_gradient_norms = verify_history_gradients(
        model,
        features,
        evidence,
        build_trajectory_relation_teacher_v56(
            evidence, config, batch["frame_times"]
        ),
        batch,
        config,
        histories,
        features.grid_hw,
        amp_context,
    )
    require(
        history_gradient_norms[str(history)] > 0.0,
        "v57 verifier produced zero gradient norm for its trainable probe",
    )
    model.eval()
    future_swapped_batch = dict(batch)
    future_swapped_rgb = batch["video_rgb"].clone()
    future_swapped_rgb[:, history:] = future_swapped_rgb[:, history:].flip(-1)
    require(
        bool((future_swapped_rgb[:, history:] != batch["video_rgb"][:, history:]).any()),
        "v57 future RGB perturbation changed no pixels",
    )
    future_swapped_batch["video_rgb"] = future_swapped_rgb
    future_swapped_features = dino(future_swapped_batch)
    with torch.no_grad(), amp_context():
        first = model.encode(
            features.patches[:, :history],
            features.coordinates[:, :history],
            features.valid[:, :history],
            batch["frame_times"][:, :history],
            teacher.query_coordinate,
        )
        swapped_future = model.encode(
            future_swapped_features.patches[:, :history],
            future_swapped_features.coordinates[:, :history],
            future_swapped_features.valid[:, :history],
            batch["frame_times"][:, :history],
            teacher.query_coordinate,
        )
    student_difference = state_max_difference(first, swapped_future)
    shuffled = future_track_shuffle(evidence, history)
    shuffled_relation = build_trajectory_relation_teacher_v56(
        shuffled, config, batch["frame_times"]
    )
    shuffled_teacher = build_query_object_teacher_v57(
        shuffled, shuffled_relation, config, observed_frames=history
    )
    teacher_difference = teacher_target_difference(teacher, shuffled_teacher)
    require(student_difference < 1e-6, "v57 student path reads future RGB")
    require(teacher_difference > 1e-6, "v57 teacher ignores future track identity")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.source_revision,
        "data_index": args.data_index,
        "coverage_report": args.coverage_report,
        "decode_report": args.decode_report,
        "dino_checkpoint": args.dino_checkpoint,
        "tracker_checkpoint": args.tracker_checkpoint,
        "history_lengths": args.history_lengths,
        "teacher_future_frames": args.teacher_future_frames,
        "temporal_step_ms": args.temporal_step_ms,
        "student_reads_future_rgb": False,
        "student_reads_point_tracker": False,
        "teacher_reads_future_tracks": True,
        "student_future_rgb_swap_max_difference": student_difference,
        "teacher_future_swap_difference": teacher_difference,
        "binding_loss": float(output["loss"].detach()),
        "binding_gradient_norm": history_gradient_norms[str(history)],
        "history_gradient_norms": history_gradient_norms,
        "trainable_parameter_tensors": len(trainable),
        "dynamic_head_trainable": False,
        "historical_checkpoint_used": False,
        "dynamics_present": False,
        "latent_effect_present": False,
        "teacher_probe_metrics": teacher_metrics,
        "coverage_status": coverage["status"],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
