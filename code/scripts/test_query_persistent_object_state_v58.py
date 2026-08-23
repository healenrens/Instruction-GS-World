#!/usr/bin/env python3
"""CPU objective and gradient contracts for v58 persistent query state."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.point_track_teacher import PointTrackEvidence  # noqa: E402
from igsw.adaptive_gaussian_wm.query_object_teacher_v58 import (  # noqa: E402
    build_query_persistent_teacher_v58,
    observed_evidence_prefix,
)
from igsw.adaptive_gaussian_wm.query_persistent_object_state_v58 import (  # noqa: E402
    QueryPersistentObjectStateModel,
)
from igsw.adaptive_gaussian_wm.query_persistent_objective_v58 import (  # noqa: E402
    query_persistent_state_terms,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher import (  # noqa: E402
    TrajectoryRelationTeacher,
)
from igsw.adaptive_gaussian_wm.v58_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    QueryPersistentObjectStateConfig,
)


def synthetic_inputs(config):
    torch.manual_seed(17)
    batch, observed, total, side, points = 2, 4, 6, 4, 6
    axis = torch.linspace(-1.0, 1.0, side)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
    patch_coordinates = grid[None, None].expand(batch, observed, -1, -1).clone()
    valid = torch.ones(batch, observed, side * side, dtype=torch.bool)
    patches = F.normalize(
        torch.randn(batch, observed, side * side, config.patch_dim), dim=-1
    )
    frame_times = torch.arange(observed).float()[None].expand(batch, -1) * 0.1
    point_indices = torch.tensor((0, 1, 4, 10, 11, 14))
    track_coordinates = grid[point_indices][None, None].expand(
        batch, total, -1, -1
    ).clone()
    track_coordinates[:, :, :3, 0] += torch.arange(total)[None, :, None] * 0.02
    visibility = torch.ones(batch, total, points, dtype=torch.bool)
    visibility[:, 1, 0] = False
    sampled = F.normalize(
        torch.randn(batch, total, points, config.patch_dim), dim=-1
    )
    residual = track_coordinates[:, 1:] - track_coordinates[:, :-1]
    evidence = PointTrackEvidence(
        coordinates=track_coordinates,
        visibility=visibility,
        residual_flow=residual,
        motion_salience=residual.norm(dim=-1).clamp(0.0, 1.0),
        query_times=torch.zeros(points, dtype=torch.long),
        sampled_features=sampled,
    )
    same = torch.zeros(batch, points, points)
    different = torch.zeros_like(same)
    same[:, :3, :3] = 1.0
    same[:, 3:, 3:] = 1.0
    same = same.masked_fill(torch.eye(points, dtype=torch.bool)[None], 0.0)
    different[:, :3, 3:] = 1.0
    different[:, 3:, :3] = 1.0
    relation = TrajectoryRelationTeacher(
        track_identity=F.normalize(sampled.mean(dim=1), dim=-1),
        persistence=visibility.float().mean(dim=1),
        object_confidence=torch.ones(batch, points),
        scene_confidence=torch.zeros(batch, points),
        transient_confidence=torch.zeros(batch, points),
        same_confidence=same,
        different_confidence=different,
        visibility=visibility.float(),
        presence=visibility.float(),
        lifecycle_known=visibility,
        lifecycle_state=torch.zeros(batch, total, points, dtype=torch.long),
        motion=torch.zeros(batch, total, points, len(config.dynamic_horizons), 2),
        motion_valid=torch.zeros(
            batch, total, points, len(config.dynamic_horizons), dtype=torch.bool
        ),
        relation_score=same,
    )
    return patches, patch_coordinates, valid, frame_times, evidence, relation, (side, side)


def objective_with_state(model, output, state, features, observed, teacher, grid_hw):
    return query_persistent_state_terms(
        state,
        output["alternate"],
        output["negative"],
        output["motion_prediction"],
        features,
        observed,
        teacher,
        grid_hw,
        model.config,
    )


def main():
    config = QueryPersistentObjectStateConfig(
        patch_dim=16,
        model_dim=32,
        identity_dim=16,
        dynamic_dim=16,
        heads=4,
        teacher_future_frames=2,
    )
    config.validate()
    if not config.tracker_bidirectional:
        raise RuntimeError("v58 requires bidirectional point tracking")
    if not config.tracker_include_observed_current_anchor:
        raise RuntimeError("v58 requires a tracker query at the observed current frame")
    patches, coordinates, valid, times, evidence, relation, grid_hw = synthetic_inputs(config)
    model = QueryPersistentObjectStateModel(config)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    gradient_norms = {}
    output = None
    teacher = None
    for history in (1, 2, 3, 4):
        teacher = build_query_persistent_teacher_v58(
            evidence, relation, config, times, observed_frames=history
        )
        observed = observed_evidence_prefix(evidence, history)
        output = model(
            patches[:, :history],
            coordinates[:, :history],
            valid[:, :history],
            times[:, :history],
            teacher,
            observed,
            grid_hw,
        )
        output["loss"].backward()
        missing = [name for name, value in trainable if value.grad is None]
        if missing:
            raise RuntimeError(f"v58 H={history} unused parameters: {missing}")
        gradients = [value.grad for _, value in trainable]
        if not all(bool(value.isfinite().all()) for value in gradients):
            raise RuntimeError(f"v58 H={history} produced non-finite gradients")
        gradient_norms[str(history)] = float(
            torch.sqrt(sum(value.float().square().sum() for value in gradients))
        )
        model.zero_grad(set_to_none=True)

    if not bool(teacher.occluded_candidate[:, 1].all()):
        raise RuntimeError("v58 synthetic lifecycle did not expose occlusion")
    if not bool((teacher.query_index == 0).all()):
        raise RuntimeError("v58 query selection did not prefer reappearing tracks")
    if bool(teacher.unknown[:, 1].any()):
        raise RuntimeError("v58 enclosed occlusion was incorrectly marked unknown")
    features = type("Features", (), {"patches": patches, "valid": valid})()
    observed = observed_evidence_prefix(evidence, 4)
    primary = output["primary"]
    target = teacher.visibility_target
    perfect = replace(
        primary,
        visibility_logits=torch.where(target > 0.5, 8.0, -8.0),
        visibility=torch.where(target > 0.5, 1.0 - 1e-4, torch.full_like(target, 1e-4)),
    )
    all_zero = replace(
        primary,
        visibility_logits=torch.full_like(target, -8.0),
        visibility=torch.full_like(target, 1e-4),
    )
    all_one = replace(
        primary,
        visibility_logits=torch.full_like(target, 8.0),
        visibility=torch.full_like(target, 1.0 - 1e-4),
    )
    scores = {
        "perfect": float(
            objective_with_state(model, output, perfect, features, observed, teacher, grid_hw)[
                "total"
            ].detach()
        ),
        "all_visibility_zero": float(
            objective_with_state(model, output, all_zero, features, observed, teacher, grid_hw)[
                "total"
            ].detach()
        ),
        "all_visibility_one": float(
            objective_with_state(model, output, all_one, features, observed, teacher, grid_hw)[
                "total"
            ].detach()
        ),
    }
    if not scores["perfect"] < min(
        scores["all_visibility_zero"], scores["all_visibility_one"]
    ):
        raise RuntimeError(f"v58 objective accepts constant visibility: {scores}")
    reference = objective_with_state(
        model, output, primary, features, observed, teacher, grid_hw
    )
    collapsed = objective_with_state(
        model, output, all_zero, features, observed, teacher, grid_hw
    )
    for name in ("semantic_consistency", "compactness", "identity_persistence"):
        if float((reference[name] - collapsed[name]).abs().detach()) != 0.0:
            raise RuntimeError(f"v58 student visibility gates {name}")
    report = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "gradient_tensor_count": len(trainable),
        "history_gradient_norms": gradient_norms,
        "dynamic_head_trainable": all(
            value.requires_grad for value in model.encoder.dynamic.parameters()
        ),
        "visibility_is_independently_supervised": True,
        "student_visibility_gates_other_losses": False,
        "objective_falsification": scores,
        "lifecycle_occluded_candidate_fraction": float(
            teacher.occluded_candidate.float().mean()
        ),
        "dynamics_present": False,
        "latent_effect_present": False,
        "historical_checkpoint_used": False,
        "tracker_bidirectional": config.tracker_bidirectional,
        "tracker_include_observed_current_anchor": (
            config.tracker_include_observed_current_anchor
        ),
        "reappearing_query_selected": True,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
