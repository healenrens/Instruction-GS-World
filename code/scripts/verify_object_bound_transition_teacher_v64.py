"""Deterministic structural verification for the v64 teacher."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.object_bound_transition_teacher_v64 import (  # noqa: E402
    build_object_bound_membership_v64,
    fit_shared_transition_v64,
)
from igsw.adaptive_gaussian_wm.object_bound_transition_audit_v64 import (  # noqa: E402
    object_bound_transition_audit_v64,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    PointTrackEvidence,
)
from igsw.adaptive_gaussian_wm.v64_config import (  # noqa: E402
    CONTRACT,
    ObjectBoundTransitionConfigV64,
)


def fixture():
    batch, frames, points, channels = 2, 10, 8, 8
    labels = torch.tensor([[0, 0, 0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1]])
    within = torch.arange(points).remainder(4).float() - 1.5
    center_x = torch.where(labels == 0, -0.45, 0.45)
    base_x = center_x + 0.04 * within[None]
    base_y = 0.12 * within[None].expand(batch, -1)
    direction = torch.where(labels == 0, 1.0, -1.0)
    time = torch.arange(frames).float()[None, :, None]
    x = base_x[:, None] + time * direction[:, None] * 0.025
    y = base_y[:, None].expand(batch, frames, points)
    coordinates = torch.stack((x, y), dim=-1)
    visibility = torch.ones(batch, frames, points, dtype=torch.bool)
    residual_flow = coordinates[:, 1:] - coordinates[:, :-1]
    motion_salience = torch.ones(batch, frames - 1, points)
    features = torch.zeros(batch, frames, points, channels)
    features[..., 0] = (labels[:, None] == 0).float()
    features[..., 1] = (labels[:, None] == 1).float()
    features = torch.nn.functional.normalize(features, dim=-1)
    evidence = PointTrackEvidence(
        coordinates=coordinates,
        visibility=visibility,
        residual_flow=residual_flow,
        motion_salience=motion_salience,
        query_times=torch.zeros(points, dtype=torch.long),
        sampled_features=features,
    )
    observation = SimpleNamespace(dino=features, siglip=features)
    return observation, evidence, labels


def main():
    config = ObjectBoundTransitionConfigV64()
    config.validate()
    observation, evidence, labels = fixture()
    split = evidence.visibility.shape[1] // 2
    sequence_index = torch.zeros(len(labels), dtype=torch.long)
    membership = build_object_bound_membership_v64(
        observation, evidence, sequence_index, config, 0, split
    )
    batch = torch.arange(len(labels))
    selected_label = labels[batch, membership.selected_seed]
    expected = labels == selected_label[:, None]
    inside = (membership.selected * expected.float()).sum(dim=1)
    inside = inside / expected.float().sum(dim=1)
    outside = (membership.selected * (~expected).float()).sum(dim=1)
    outside = outside / (~expected).float().sum(dim=1)
    _, _, selected_error, selected_fit_valid = fit_shared_transition_v64(
        evidence, membership.selected, split, 2 * split, config
    )
    mixed = torch.ones_like(membership.selected)
    _, _, mixed_error, mixed_fit_valid = fit_shared_transition_v64(
        evidence, mixed, split, 2 * split, config
    )
    same = (labels[:, :, None] == labels[:, None]).float()
    diagonal = torch.eye(same.shape[-1], dtype=torch.bool)[None]
    relation = SimpleNamespace(same_confidence=same.masked_fill(diagonal, 0.0))
    audit = object_bound_transition_audit_v64(
        SimpleNamespace(
            observation=observation,
            evidence=evidence,
            relation=relation,
        ),
        sequence_index,
        config,
    )
    overlap = (membership.components > 0.0).sum(dim=1).amax()
    checks = {
        "selected_components_are_valid": bool(membership.selected_valid.all()),
        "direct_seed_grouping": bool((inside > outside).all()),
        "shared_transition_beats_mixed_tracks": bool(
            (selected_error < mixed_error).all()
        ),
        "selected_transition_fit_is_valid": bool(selected_fit_valid.all()),
        "mixed_transition_fit_is_valid": bool(mixed_fit_valid.all()),
        "held_audit_is_valid": bool(audit["audit_valid"].all()),
        "held_audit_is_finite": all(
            bool(value.isfinite().all()) for value in audit.values()
        ),
        "components_do_not_overlap": int(overlap) <= 1,
        "all_outputs_are_finite": all(
            bool(value.isfinite().all())
            for value in (
                membership.components,
                membership.scene_membership,
                membership.unknown_membership,
                selected_error,
                mixed_error,
            )
        ),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": CONTRACT,
        "checks": checks,
        "inside_membership": inside.tolist(),
        "outside_membership": outside.tolist(),
        "selected_transition_residual": selected_error.tolist(),
        "mixed_transition_residual": mixed_error.tolist(),
    }
    print(json.dumps(report, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError("v64 structural verification failed")


if __name__ == "__main__":
    main()
