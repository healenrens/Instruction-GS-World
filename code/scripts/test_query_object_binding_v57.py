#!/usr/bin/env python3
"""CPU contracts and objective falsification for v57 single-query binding."""

from __future__ import annotations

import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.point_track_teacher import PointTrackEvidence  # noqa: E402
from igsw.adaptive_gaussian_wm.query_object_binding_v57 import (  # noqa: E402
    QueryObjectBindingModel,
)
from igsw.adaptive_gaussian_wm.query_object_state_v57 import (  # noqa: E402
    QueryObjectState,
)
from igsw.adaptive_gaussian_wm.query_object_teacher_v57 import (  # noqa: E402
    build_query_object_teacher_v57,
    observed_evidence_prefix,
    query_teacher_contract_metrics,
)
from igsw.adaptive_gaussian_wm.query_objective_v57 import (  # noqa: E402
    query_object_binding_terms,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher import (  # noqa: E402
    TrajectoryRelationTeacher,
)
from igsw.adaptive_gaussian_wm.v57_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    QueryConditionedObjectStateConfig,
)


def synthetic_inputs(config):
    torch.manual_seed(17)
    batch, frames, side, points = 2, 4, 4, 6
    axis = torch.linspace(-1.0, 1.0, side)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
    coordinates = grid[None, None].expand(batch, frames, -1, -1).clone()
    valid = torch.ones(batch, frames, side * side, dtype=torch.bool)
    patches = torch.randn(batch, frames, side * side, config.patch_dim)
    patches = torch.nn.functional.normalize(patches, dim=-1)
    frame_times = torch.arange(frames).float()[None].expand(batch, -1) * 0.1

    point_indices = torch.tensor((0, 1, 4, 10, 11, 14))
    track_coordinates = coordinates[:, :, point_indices].clone()
    visibility = torch.ones(batch, frames, points, dtype=torch.bool)
    sampled = patches[:, :, point_indices].clone()
    evidence = PointTrackEvidence(
        coordinates=track_coordinates,
        visibility=visibility,
        residual_flow=torch.zeros(batch, frames - 1, points, 2),
        motion_salience=torch.ones(batch, frames - 1, points),
        query_times=torch.zeros(points, dtype=torch.long),
        sampled_features=sampled,
    )
    same = torch.zeros(batch, points, points)
    different = torch.zeros_like(same)
    same[:, :3, :3] = 1.0
    same[:, 3:, 3:] = 1.0
    eye = torch.eye(points, dtype=torch.bool)[None]
    same = same.masked_fill(eye, 0.0)
    different[:, :3, 3:] = 1.0
    different[:, 3:, :3] = 1.0
    relation = TrajectoryRelationTeacher(
        track_identity=torch.nn.functional.normalize(sampled.mean(dim=1), dim=-1),
        persistence=torch.ones(batch, points),
        object_confidence=torch.ones(batch, points),
        scene_confidence=torch.zeros(batch, points),
        transient_confidence=torch.zeros(batch, points),
        same_confidence=same,
        different_confidence=different,
        visibility=visibility.float(),
        presence=visibility.float(),
        lifecycle_known=visibility,
        lifecycle_state=torch.zeros(batch, frames, points, dtype=torch.long),
        motion=torch.zeros(batch, frames, points, 1, 2),
        motion_valid=torch.zeros(batch, frames, points, 1, dtype=torch.bool),
        relation_score=same,
    )
    features = type(
        "Features", (), {"patches": patches, "coordinates": coordinates, "valid": valid}
    )()
    return features, frame_times, evidence, relation, (side, side), point_indices


def state_from_support(support, features, identity):
    batch, frames, _, patch_dim = features.patches.shape
    normalized = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    center = torch.einsum("btn,btnd->btd", normalized, features.coordinates)
    offset = features.coordinates - center[:, :, None]
    covariance = torch.einsum("btn,btni,btnj->btij", normalized, offset, offset)
    pooled = torch.einsum("btn,btnd->btd", normalized, features.patches)
    return QueryObjectState(
        support_logits=torch.logit(support.clamp(1e-4, 1.0 - 1e-4)),
        support=support,
        identity=identity,
        dynamic=torch.zeros(batch, frames, identity.shape[-1]),
        center=center,
        covariance=covariance + 1e-4 * torch.eye(2),
        visibility=torch.ones(batch, frames),
        pooled_semantic=pooled.reshape(batch, frames, patch_dim),
    )


def objective_falsification(config, features, evidence, teacher, grid_hw, point_indices):
    batch, frames, patches = features.valid.shape
    first = torch.zeros(batch, frames, patches)
    first[:, :, point_indices[:3]] = 0.99
    second = torch.zeros_like(first)
    second[:, :, point_indices[3:]] = 0.99
    whole = torch.full_like(first, 0.99)
    empty = torch.zeros_like(first)
    seed_only = torch.zeros_like(first)
    seed_only[:, :, point_indices[0]] = 0.99
    identity_a = torch.zeros(batch, config.identity_dim)
    identity_b = torch.zeros_like(identity_a)
    identity_a[:, 0] = 1.0
    identity_b[:, 1] = 1.0
    alternate = state_from_support(first, features, identity_a)
    negative = state_from_support(second, features, identity_b)

    def score(support):
        primary = state_from_support(support, features, identity_a)
        return float(
            query_object_binding_terms(
                primary,
                alternate,
                negative,
                features,
                evidence,
                teacher,
                grid_hw,
                config,
            )["total"].detach()
        )

    scores = {
        "reasonable": score(first),
        "all_scene": score(empty),
        "whole_frame": score(whole),
        "other_entity": score(second),
        "seed_only": score(seed_only),
    }
    ranking = all(
        scores["reasonable"] < value
        for name, value in scores.items()
        if name != "reasonable"
    )
    if not ranking:
        raise RuntimeError(f"v57 objective accepted a shortcut: {scores}")
    return {"scores": scores, "ranking_passed": ranking}


def main():
    config = QueryConditionedObjectStateConfig(
        patch_dim=16,
        model_dim=32,
        identity_dim=16,
        dynamic_dim=16,
        heads=4,
    )
    config.validate()
    features, frame_times, evidence, relation, grid_hw, point_indices = synthetic_inputs(config)
    observed_frames = 2
    teacher = build_query_object_teacher_v57(
        evidence, relation, config, observed_frames=observed_frames
    )
    observed_evidence = observed_evidence_prefix(evidence, observed_frames)
    observed_features = type(
        "Features",
        (),
        {
            "patches": features.patches[:, :observed_frames],
            "coordinates": features.coordinates[:, :observed_frames],
            "valid": features.valid[:, :observed_frames],
        },
    )()
    contract = query_teacher_contract_metrics(teacher)
    if not bool(teacher.query_valid.all()):
        raise RuntimeError("v57 synthetic query selection failed")
    if not bool(teacher.alternate_valid.all() and teacher.negative_valid.all()):
        raise RuntimeError("v57 synthetic query lacks seed or negative")
    if float(contract["prompt_heldout_overlap"]) != 0.0:
        raise RuntimeError("v57 prompt tracks leaked into held-out targets")

    model = QueryObjectBindingModel(config)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    for history in range(1, observed_frames + 1):
        history_teacher = build_query_object_teacher_v57(
            evidence, relation, config, observed_frames=history
        )
        output = model(
            features.patches[:, :history],
            features.coordinates[:, :history],
            features.valid[:, :history],
            frame_times[:, :history],
            history_teacher,
            observed_evidence_prefix(evidence, history),
            grid_hw,
        )
        output["loss"].backward()
        missing = [name for name, parameter in trainable if parameter.grad is None]
        if missing:
            raise RuntimeError(
                f"v57 H={history} has unused trainable parameters: {missing}"
            )
        gradients = [parameter.grad for _, parameter in trainable]
        if not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f"v57 H={history} has non-finite gradients")
        model.zero_grad(set_to_none=True)
    output = model(
        observed_features.patches,
        observed_features.coordinates,
        observed_features.valid,
        frame_times[:, :observed_frames],
        teacher,
        observed_evidence,
        grid_hw,
    )
    primary = output["primary"]

    swapped_relation = TrajectoryRelationTeacher(
        **{
            **relation.__dict__,
            "same_confidence": relation.different_confidence,
            "different_confidence": relation.same_confidence,
        }
    )
    swapped_teacher = build_query_object_teacher_v57(
        evidence, swapped_relation, config, observed_frames=observed_frames
    )
    repeated = model.encode(
        observed_features.patches,
        observed_features.coordinates,
        observed_features.valid,
        frame_times[:, :observed_frames],
        teacher.query_coordinate,
    )
    causal_difference = float(
        (primary.identity - repeated.identity).abs().max().detach()
    )
    teacher_difference = float(
        (teacher.same_target - swapped_teacher.same_target).abs().max().detach()
    )
    if causal_difference != 0.0 or teacher_difference <= 0.0:
        raise RuntimeError("v57 student/teacher causal boundary failed")

    falsification = objective_falsification(
        config,
        observed_features,
        observed_evidence,
        teacher,
        grid_hw,
        point_indices,
    )
    report = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "student_reads_point_tracker": False,
        "teacher_uses_future_tracks": True,
        "student_observed_frames": observed_frames,
        "teacher_total_frames": evidence.visibility.shape[1],
        "prompt_heldout_overlap": float(contract["prompt_heldout_overlap"]),
        "heldout_positive_fraction": float(contract["heldout_positive_fraction"]),
        "heldout_negative_fraction": float(contract["heldout_negative_fraction"]),
        "student_teacher_swap_max_difference": causal_difference,
        "teacher_swap_max_difference": teacher_difference,
        "gradient_tensor_count": len(trainable),
        "dynamic_head_trainable": any(
            parameter.requires_grad for parameter in model.encoder.dynamic.parameters()
        ),
        "objective_falsification": falsification,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
