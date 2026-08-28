"""CPU tensor-contract test for v61 carriers, roots, and teacher objectives."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.carrier_objective_v61 import (  # noqa: E402
    _relation_terms,
    continuous_carrier_objective_v61,
)
from igsw.adaptive_gaussian_wm.bounded_probability_v61 import (  # noqa: E402
    binary_probability_values,
    normalize_probability_mass,
)
from igsw.adaptive_gaussian_wm.carrier_teacher_v61 import (  # noqa: E402
    TeacherObjectComponentsV61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_state_v61 import (  # noqa: E402
    ContinuousCarrierObjectStateEncoderV61,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    PointTrackEvidence,
)
from igsw.adaptive_gaussian_wm.student_visual_encoder_v61 import (  # noqa: E402
    StudentTokenField,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher import (  # noqa: E402
    TrajectoryRelationTeacher,
)
from igsw.adaptive_gaussian_wm.v61_config import config_for_variant  # noqa: E402


class ObjectiveHarness(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_encoder = ContinuousCarrierObjectStateEncoderV61(config)
        self.identity_to_dino = nn.Linear(
            config.identity_dim, config.teacher_projection_dim
        )
        self.root_to_siglip = nn.Linear(
            config.identity_dim, config.teacher_projection_dim
        )
        self.motion_readout = nn.Linear(
            config.dynamic_dim, len(config.dynamic_horizons) * 2
        )


def synthetic_field(config, batch=2, frames=5, grid=4):
    tokens = grid * grid
    axis = torch.linspace(-1.0, 1.0, grid)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).reshape(1, 1, tokens, 2)
    coordinates = coordinates.expand(batch, frames, -1, -1)
    features = F.normalize(
        torch.randn(batch, frames, tokens, config.student_dim), dim=-1
    )
    return StudentTokenField(
        features=features,
        coordinates=coordinates,
        valid=torch.ones(batch, frames, tokens, dtype=torch.bool),
        pooled=F.normalize(features.mean(dim=2), dim=-1),
        grid_hw=(grid, grid),
    )


def synthetic_teachers(config, batch=2, frames=5, points=12):
    coordinates = torch.rand(batch, frames, points, 2) * 1.8 - 0.9
    visibility = torch.ones(batch, frames, points, dtype=torch.bool)
    visibility[:, 2, ::3] = False
    residual = coordinates[:, 1:] - coordinates[:, :-1]
    evidence = PointTrackEvidence(
        coordinates=coordinates,
        visibility=visibility,
        residual_flow=residual,
        motion_salience=residual.norm(dim=-1).clamp(0.0, 1.0),
        query_times=torch.zeros(points, dtype=torch.long),
        sampled_features=F.normalize(
            torch.randn(batch, frames, points, config.dino_dim), dim=-1
        ),
    )
    same = torch.zeros(batch, points, points)
    different = torch.zeros_like(same)
    for start in range(0, points, 3):
        same[:, start : start + 3, start : start + 3] = 1.0
    different[:, :3, 3:6] = 1.0
    different[:, 3:6, :3] = 1.0
    motion = torch.randn(batch, frames, points, len(config.dynamic_horizons), 2) * 0.01
    motion_valid = visibility[..., None].expand(
        -1, -1, -1, len(config.dynamic_horizons)
    )
    relation = TrajectoryRelationTeacher(
        track_identity=F.normalize(torch.randn(batch, points, config.dino_dim), dim=-1),
        persistence=visibility.float().mean(dim=1),
        object_confidence=torch.ones(batch, points),
        scene_confidence=torch.zeros(batch, points),
        transient_confidence=torch.zeros(batch, points),
        same_confidence=same,
        different_confidence=different,
        visibility=visibility.float(),
        presence=torch.ones(batch, frames, points),
        lifecycle_known=torch.ones(batch, frames, points, dtype=torch.bool),
        lifecycle_state=torch.zeros(batch, frames, points, dtype=torch.long),
        motion=motion,
        motion_valid=motion_valid,
        relation_score=same,
    )
    membership = torch.zeros(batch, config.object_roots, points)
    for root in range(config.object_roots):
        membership[:, root, root % points] = 1.0
    components = TeacherObjectComponentsV61(
        membership=membership,
        valid=torch.ones(batch, config.object_roots, dtype=torch.bool),
        semantic=F.normalize(
            torch.randn(batch, 2, config.object_roots, config.siglip_dim), dim=-1
        ),
        semantic_valid=torch.ones(batch, 2, config.object_roots, dtype=torch.bool),
        frame_indices=torch.tensor((0, frames - 1)),
    )
    return evidence, relation, components


def saturated_probability_contract(config, evidence, relation):
    batch, frames, points = evidence.visibility.shape
    roots = config.object_roots
    low_mass_logits = torch.full((batch, frames, points, roots), -20.0)
    low_mass_logits[..., 0] = 20.0
    low_mass_logits.requires_grad_()
    boundary_logits = torch.full((batch, frames, points, roots), -10.0)
    boundary_logits[..., 0] = 10.0
    boundary_logits.requires_grad_()
    binary_boundary = torch.tensor((0.0, 1.0), requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low_mass = torch.softmax(low_mass_logits, dim=-1) * 1e-12
        normalized = normalize_probability_mass(
            low_mass,
            dim=-1,
            prior_mass=config.assignment_prior_mass,
        )
        low_same, low_different, _, low_coassignment = _relation_terms(
            normalized,
            relation,
            evidence,
            config.relation_probability_floor,
        )
        boundary = torch.softmax(boundary_logits, dim=-1)
        boundary_same, boundary_different, _, boundary_coassignment = _relation_terms(
            boundary,
            relation,
            evidence,
            config.relation_probability_floor,
        )
        binary_loss = binary_probability_values(
            binary_boundary,
            torch.tensor((1.0, 0.0)),
            config.relation_probability_floor,
        ).mean()
        loss = (
            low_same
            + low_different
            + boundary_same
            + boundary_different
            + binary_loss
        )
    loss.backward()
    values = (
        loss,
        normalized,
        low_coassignment,
        boundary_coassignment,
        low_mass_logits.grad,
        boundary_logits.grad,
        binary_boundary.grad,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("v61 saturated probability contract is non-finite")
    maximum_gradient = max(
        float(low_mass_logits.grad.abs().max()),
        float(boundary_logits.grad.abs().max()),
        float(binary_boundary.grad.abs().max()),
    )
    if not 1.0 < maximum_gradient < 100.0:
        raise RuntimeError("v61 saturated probability gradient is not bounded")
    return maximum_gradient


def main():
    torch.manual_seed(17)
    config = config_for_variant("siglip_dino_object")
    model = ObjectiveHarness(config)
    field = synthetic_field(config)
    evidence, relation, components = synthetic_teachers(config)
    saturated_gradient = saturated_probability_contract(config, evidence, relation)
    state = model.state_encoder(field)
    loss, parts, carrier_assignment, root_assignment = continuous_carrier_objective_v61(
        model, field, state, evidence, relation, components
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("v61 CPU objective is non-finite")
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"v61 CPU objective leaves parameters unused: {missing}")
    tensors = (*parts.values(), carrier_assignment, root_assignment)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("v61 CPU objective produced non-finite diagnostics")
    if float((root_assignment.sum(dim=-1) - 1.0).abs().max().detach()) >= 1e-5:
        raise RuntimeError("v61 track-to-root assignment is not normalized")
    print(
        {
            "status": "passed",
            "loss": float(loss.detach()),
            "metric_count": len(parts),
            "carrier_assignment_shape": tuple(carrier_assignment.shape),
            "root_assignment_shape": tuple(root_assignment.shape),
            "saturated_probability_max_gradient": saturated_gradient,
        }
    )


if __name__ == "__main__":
    main()
