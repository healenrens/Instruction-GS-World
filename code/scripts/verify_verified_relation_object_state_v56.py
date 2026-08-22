#!/usr/bin/env python3
"""GPU and real-data verifier for v56 Object State learning."""

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
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    parse_int_choices,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v56_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    VerifiedRelationObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v56_data_audit import (  # noqa: E402
    factorization_probe,
    summarize_real_target_audit,
    teacher_batch_metrics,
)
from igsw.adaptive_gaussian_wm.v56_data_contract import (  # noqa: E402
    audit_decode_frontier,
)
from igsw.adaptive_gaussian_wm.v56_independent_gates import (  # noqa: E402
    run_v56_independent_gates,
)
from igsw.adaptive_gaussian_wm.verified_relation_object_state_v56 import (  # noqa: E402
    VerifiedRelationObjectStateModel,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_lengths", default="3,4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--audit_chunk_lengths", default="4,8")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def select_evidence(evidence, index: int) -> PointTrackEvidence:
    return PointTrackEvidence(
        coordinates=evidence.coordinates[index : index + 1],
        visibility=evidence.visibility[index : index + 1],
        residual_flow=evidence.residual_flow[index : index + 1],
        motion_salience=evidence.motion_salience[index : index + 1],
        query_times=evidence.query_times,
        sampled_features=evidence.sampled_features[index : index + 1],
    )


def finite_gradient_contract(model, output):
    output["loss"].backward()
    trainable = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    missing, nonfinite, gradients = [], [], []
    for name, parameter in trainable:
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        gradients.append(parameter.grad.detach().float().square().sum())
    require(not missing, f"v56 has unused trainable parameters: {missing}")
    require(not nonfinite, f"v56 has non-finite gradients: {nonfinite}")
    norm = torch.stack(gradients).sum().sqrt()
    require(float(norm) > 0.0, "v56 produced zero gradient norm")
    preclip = clip_finite_grad_norm_(trainable, 5.0)
    model.zero_grad(set_to_none=True)
    return {
        "object_state_loss": float(output["loss"].detach()),
        "object_state_gradient_norm": float(norm),
        "object_state_preclip_gradient_norm": float(preclip),
        "object_state_trainable_parameter_tensors": float(len(trainable)),
    }


@torch.no_grad()
def structural_contract(model, features, evidence, output):
    state = output["state"]
    batch, frames, patches = features.valid.shape
    slots = model.config.object_slots
    require(
        state["identity"].shape == (batch, frames, slots, model.config.identity_dim),
        "v56 identity shape differs",
    )
    require(
        state["dynamic"].shape == (batch, frames, slots, model.config.dynamic_dim),
        "v56 dynamic shape differs",
    )
    require(
        state["assignment"].shape == (batch, frames, patches, model.config.owner_count),
        "v56 assignment shape differs",
    )
    partition = state["assignment"].sum(dim=-1)
    error = maximum_difference(
        partition[features.valid], torch.ones_like(partition[features.valid])
    )
    require(error < 1e-5, "v56 encoder owners do not partition patches")
    required = {
        "target_relation_partition",
        "target_contrastive_cycle",
        "target_object_support",
        "target_relation_collapse_margin",
        "target_verified_effective_roots",
    }
    missing = required.difference(output["parts"])
    require(not missing, f"v56 metrics are missing: {missing}")
    require(not hasattr(model, "dynamics"), "v56 contains forbidden Dynamics")
    require(
        not hasattr(model, "effect_posterior"),
        "v56 contains forbidden latent effect",
    )
    require(
        float(output["teacher"].scene_confidence.amax()) == 0.0,
        "v56 teacher invents scene labels",
    )
    require(
        float(output["teacher"].transient_confidence.amax()) == 0.0,
        "v56 teacher invents transient labels",
    )
    return {
        "object_slots": float(slots),
        "owner_count": float(model.config.owner_count),
        "patch_count": float(patches),
        "point_track_count": float(evidence.coordinates.shape[2]),
        "encoder_owner_partition_max_error": error,
        "teacher_batch_fraction": float(output["parts"]["teacher_batch_fraction"]),
    }


@torch.no_grad()
def causal_contract(model, features, batch, amp_context):
    midpoint = features.patches.shape[1] // 2
    changed = features.patches.clone()
    changed[:, midpoint:] = changed[:, midpoint:].flip(2)
    with amp_context():
        _, reference = model.encode_student(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
        )
        _, altered = model.encode_student(
            changed,
            features.coordinates,
            features.valid,
            batch["frame_times"],
        )
    difference = maximum_difference(
        reference["identity"][:, :midpoint],
        altered["identity"][:, :midpoint],
    )
    require(difference < 1e-6, "future RGB changed the v56 state prefix")
    return {"future_swap_prefix_max_difference": difference}


@torch.no_grad()
def teacher_isolation_contract(model, features, batch, evidence, output, amp_context):
    changed = PointTrackEvidence(
        coordinates=evidence.coordinates.roll(1, dims=2),
        visibility=evidence.visibility,
        residual_flow=evidence.residual_flow.roll(1, dims=2),
        motion_salience=evidence.motion_salience.roll(1, dims=2),
        query_times=evidence.query_times,
        sampled_features=evidence.sampled_features.roll(1, dims=2),
    )
    changed_teacher = build_trajectory_relation_teacher_v56(
        changed, model.config, batch["frame_times"]
    )
    with amp_context():
        changed_output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            changed,
            torch.tensor([0], device=features.patches.device),
            features.grid_hw,
        )
    state_difference = maximum_difference(
        output["state"]["identity"], changed_output["state"]["identity"]
    )
    target_difference = maximum_difference(
        output["teacher"].relation_score,
        changed_teacher.relation_score,
    )
    require(state_difference < 1e-6, "v56 Student reads teacher tracks")
    require(target_difference > 1e-6, "v56 relation target ignores tracks")
    return {
        "student_state_track_swap_max_difference": state_difference,
        "teacher_relation_track_swap_max_difference": target_difference,
    }


def real_data_audit(dataset, dino, tracker, config, device, audit_lengths):
    require(
        len(dataset.source_audit_indices) == len(dataset.source_names),
        "v56 dataset cannot expose audit probes per source",
    )
    rows, probes = [], []
    model_probe = None
    for chunk_length in audit_lengths:
        audit_indices = [
            index
            for source_indices in dataset.source_audit_indices
            for index in source_indices
        ]
        samples = [dataset[(index, chunk_length)] for index in audit_indices]
        batch = move_batch(default_collate(samples), device)
        source_names = [
            dataset.source_names[int(index)] for index in batch["source_index"].tolist()
        ]
        require(
            len(set(source_names)) == len(dataset.source_names),
            "v56 probe replacement lost source coverage",
        )
        features = dino(batch)
        evidence = tracker(batch, features.patches, features.grid_hw)
        teacher = build_trajectory_relation_teacher_v56(
            evidence, config, batch["frame_times"]
        )
        rows.extend(teacher_batch_metrics(teacher, source_names, config))
        probes.append(factorization_probe(teacher, config.object_slots))
        if model_probe is None:
            one_batch = {
                name: value[:1] if torch.is_tensor(value) and value.ndim else value
                for name, value in batch.items()
            }
            one_features = type(features)(
                patches=features.patches[:1],
                coordinates=features.coordinates[:1],
                valid=features.valid[:1],
                grid_hw=features.grid_hw,
            )
            model_probe = (one_batch, one_features, select_evidence(evidence, 0))
    report = summarize_real_target_audit(rows, probes, config)
    require(report["status"] == "passed", f"v56 real target audit failed: {report}")
    return report, model_probe


def main() -> None:
    args = parse_args()
    for name in (
        "data_index",
        "decode_report",
        "dino_checkpoint",
        "tracker_checkpoint",
    ):
        value = os.path.abspath(getattr(args, name))
        require(os.path.isfile(value), f"v56 {name} is missing: {value}")
        setattr(args, name, value)
    if not torch.cuda.is_available():
        raise RuntimeError("v56 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = VerifiedRelationObjectStateConfig()
    config.validate()
    independent = run_v56_independent_gates(config, device)
    require(independent["status"] == "passed", "v56 independent gates failed")
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        chunk_lengths=args.chunk_lengths,
        temporal_step_ms=args.temporal_step_ms,
        max_items=0,
        seed=args.seed,
    )
    decode_audit = audit_decode_frontier(
        args.decode_report,
        args.data_index,
        args.seed,
        dataset.source_names,
    )
    require(
        decode_audit["status"] == "passed", f"v56 decode audit failed: {decode_audit}"
    )
    dino = FrozenDinoVideoRuntime(
        config,
        device,
        args.amp,
        args.dino_frame_batch,
        args.dino_checkpoint,
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    require(
        not any(parameter.requires_grad for parameter in dino.backbone.parameters()),
        "v56 DINO is not frozen",
    )
    require(
        not any(parameter.requires_grad for parameter in tracker.model.parameters()),
        "v56 point tracker is not frozen",
    )
    audit_lengths = parse_int_choices(
        args.audit_chunk_lengths, "v56 audit chunk lengths"
    )
    data_audit, model_probe = real_data_audit(
        dataset, dino, tracker, config, device, audit_lengths
    )
    batch, features, evidence = model_probe
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    model = VerifiedRelationObjectStateModel(config).to(device).train()
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            evidence,
            torch.tensor([0], device=device),
            features.grid_hw,
        )
    structure = structural_contract(model, features, evidence, output)
    gradients = finite_gradient_contract(model, output)
    model.eval()
    causal = causal_contract(model, features, batch, amp_context)
    isolation = teacher_isolation_contract(
        model, features, batch, evidence, output, amp_context
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
        "chunk_lengths": args.chunk_lengths,
        "temporal_step_ms": args.temporal_step_ms,
        "audit_chunk_lengths": args.audit_chunk_lengths,
        "amp": args.amp,
        "historical_checkpoint_used": False,
        "dynamics_present": False,
        "latent_effect_present": False,
        "point_tracker_in_deployable_model": False,
        "point_tracker_role": "training_only_verified_relation_teacher",
        "legacy_owner_evidence_used": False,
        "legacy_track_cycle_used": False,
        "scene_pseudo_labels_used": False,
        "transient_pseudo_labels_used": False,
        "relation_conditioned_cycle_used": True,
        "positive_only_object_support_used": True,
        "decode_frontier_audit": decode_audit,
        "real_multisource_target_audit": data_audit,
        "independent_objective_gates": independent,
        **structure,
        **gradients,
        **causal,
        **isolation,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
