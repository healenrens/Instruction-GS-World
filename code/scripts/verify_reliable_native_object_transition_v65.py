#!/usr/bin/env python3
"""Synthetic contract verification for v65 native reliable transition audit."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.held_visual_transition_audit_v65 import (  # noqa: E402
    held_visual_transition_audit_v65,
)
from igsw.adaptive_gaussian_wm.native_local_feature_field_v65 import (  # noqa: E402
    NativeLocalFeatureFieldV65,
    NativeTokenFieldV65,
    pool_native_local_features_v65,
)
from igsw.adaptive_gaussian_wm.native_video_batch_v65 import (  # noqa: E402
    collate_native_video_batch_v65,
)
from igsw.adaptive_gaussian_wm.reliable_point_tracker_v65 import (  # noqa: E402
    ReliablePointTrackEvidenceV65,
    relay_track_reliability_v65,
)
from igsw.adaptive_gaussian_wm.robust_multitrack_binding_v65 import (  # noqa: E402
    build_reliable_core_binding_v65,
)
from igsw.adaptive_gaussian_wm.v65_config import (  # noqa: E402
    CONTRACT,
    ReliableNativeTransitionConfigV65,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def synthetic_bundle(config):
    batch, frames, points, channels = 2, 10, 16, 8
    labels = torch.arange(points).div(8, rounding_mode="floor")
    offsets = torch.linspace(-0.035, 0.035, 8)
    time = torch.arange(frames).float()
    centers_a = -0.8 + 0.12 * time
    centers_b = 0.8 - 0.12 * time
    x_a = centers_a[:, None] + offsets[None]
    x_b = centers_b[:, None] + offsets[None]
    y_a = -0.15 + 0.03 * torch.sin(torch.arange(8).float())[None]
    y_b = 0.15 + 0.03 * torch.cos(torch.arange(8).float())[None]
    y_a = y_a.expand(frames, -1)
    y_b = y_b.expand(frames, -1)
    coordinates = torch.stack(
        (torch.cat((x_a, x_b), dim=1), torch.cat((y_a, y_b), dim=1)), dim=-1
    )
    coordinates = coordinates[None].expand(batch, -1, -1, -1).clone()
    coordinates[1, :, :, 1] *= -1.0
    visibility = torch.ones(batch, frames, points, dtype=torch.bool)
    flow = coordinates[:, 1:] - coordinates[:, :-1]
    residual = flow - flow.mean(dim=2, keepdim=True)
    salience = residual.norm(dim=-1)
    salience = salience / salience.amax(dim=2, keepdim=True).clamp_min(1e-6)
    reliability = torch.ones(batch, points)
    evidence = ReliablePointTrackEvidenceV65(
        coordinates=coordinates,
        visibility=visibility,
        residual_flow=residual,
        motion_salience=salience,
        query_times=torch.zeros(points, dtype=torch.long),
        relay_coordinates=coordinates,
        relay_visibility=visibility,
        relay_error=torch.zeros(batch, points),
        joint_visibility_fraction=torch.ones(batch, points),
        tracker_reliability=reliability,
        appearance_reliability=reliability,
        reliability=reliability,
    )
    side = 32
    axis = torch.linspace(-1.0, 1.0, side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    token_coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    token_features = torch.zeros(batch, frames, side * side, channels)
    token_features[..., 0] = 1.0
    for sample in range(batch):
        for frame in range(frames):
            centers = torch.tensor(
                [
                    [centers_a[frame], -0.15 * (1.0 if sample == 0 else -1.0)],
                    [centers_b[frame], 0.15 * (1.0 if sample == 0 else -1.0)],
                ]
            )
            for label in range(2):
                distance = (token_coordinates - centers[label]).norm(dim=-1)
                mask = distance <= 0.075
                token_features[sample, frame, mask] = 0.0
                token_features[sample, frame, mask, label + 1] = 1.0
    token_features = F.normalize(token_features, dim=-1)
    token_coordinates = token_coordinates[None, None].expand(
        batch, frames, -1, -1
    )
    field = NativeTokenFieldV65(
        features=token_features,
        coordinates=token_coordinates,
        valid=torch.ones(batch, frames, side * side, dtype=torch.bool),
        image_hw=torch.tensor(((224, 224), (448, 448)), dtype=torch.long),
    )
    local = pool_native_local_features_v65(
        field,
        coordinates,
        config.local_radii_pixels,
        config.local_tokens_per_scale,
    )
    observation = SimpleNamespace(
        dino=local.features,
        siglip=local.features,
        dino_valid=local.valid,
        siglip_valid=local.valid,
    )
    return SimpleNamespace(
        field=NativeLocalFeatureFieldV65(dino=field, siglip=field),
        evidence=evidence,
        observation=observation,
    ), labels


def main():
    config = ReliableNativeTransitionConfigV65(
        carrier_count=2,
        tracker_grid_side=4,
        tracker_anchor_fractions=(0.0,),
        local_radii_pixels=(8.0, 16.0, 24.0),
        local_tokens_per_scale=8,
        core_tracks=4,
        core_candidate_tracks=8,
        minimum_component_tracks=4,
        minimum_holdout_tracks=2,
    )
    config.validate()
    bundle, labels = synthetic_bundle(config)
    first_rgb = torch.arange(3 * 2 * 3, dtype=torch.uint8).reshape(1, 3, 2, 3)
    second_rgb = torch.full((1, 3, 4, 5), 29, dtype=torch.uint8)
    native_batch = collate_native_video_batch_v65(
        [
            {
                "video_rgb": first_rgb,
                "video_pixel_valid": torch.ones(1, 2, 3, dtype=torch.bool),
            },
            {
                "video_rgb": second_rgb,
                "video_pixel_valid": torch.ones(1, 4, 5, dtype=torch.bool),
            },
        ]
    )
    sequence_index = torch.tensor([0, 1])
    binding = build_reliable_core_binding_v65(
        bundle.observation,
        bundle.evidence,
        sequence_index,
        config,
        0,
        5,
    )
    overlap = (binding.components > 0.0).sum(dim=1).amax()
    core_count = (binding.selected_core > 0.0).sum(dim=-1)
    holdout_count = (binding.selected_holdout > 0.0).sum(dim=-1)
    selected_label_mass = []
    for sample in range(len(sequence_index)):
        mass = torch.stack(
            [
                binding.selected[sample, labels == label].sum()
                for label in range(2)
            ]
        )
        selected_label_mass.append(mass.max() / mass.sum().clamp_min(1e-6))
    selected_label_mass = torch.stack(selected_label_mass)
    primary = bundle.evidence.coordinates
    relay_good = primary.clone()
    relay_bad = torch.roll(primary, shifts=4, dims=2)
    visibility = bundle.evidence.visibility
    _, _, reliable_good = relay_track_reliability_v65(
        primary, visibility, relay_good, visibility, config.tracker_relay_sigma
    )
    _, _, reliable_bad = relay_track_reliability_v65(
        primary, visibility, relay_bad, visibility, config.tracker_relay_sigma
    )
    audit = held_visual_transition_audit_v65(bundle, sequence_index, config)
    checks = {
        "native_collate_preserves_pixels": bool(
            torch.equal(native_batch["video_rgb"][0, :, :, :2, :3], first_rgb)
            and torch.equal(native_batch["video_rgb"][1], second_rgb)
            and native_batch["native_image_hw"].tolist() == [[2, 3], [4, 5]]
        ),
        "native_local_features_are_valid": bool(bundle.observation.dino_valid.all()),
        "relay_agreement_rejects_bad_tracks": bool(
            reliable_good.mean() > reliable_bad.mean() + 0.5
        ),
        "multi_track_binding_is_valid": bool(binding.selected_valid.all()),
        "core_has_multiple_tracks": bool((core_count >= config.core_tracks).all()),
        "held_tracks_are_not_fit_tracks": bool(
            (holdout_count >= config.minimum_holdout_tracks).all()
        ),
        "components_do_not_overlap": int(overlap) <= 1,
        "component_is_object_specific": bool((selected_label_mass > 0.9).all()),
        "held_visual_audit_is_valid": bool(audit.metrics["audit_valid"].bool().all()),
        "future_image_beats_persistence": bool(
            (audit.metrics["visual_gain_over_persistence"] > 0.0).all()
        ),
        "correct_core_beats_rolled_core": bool(
            (
                audit.metrics["visual_margin_rolled_core"]
                * audit.valid["visual_margin_rolled_core"].float()
                >= 0.0
            ).all()
        ),
        "all_outputs_are_finite": all(
            bool(value.isfinite().all()) for value in audit.metrics.values()
        ),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": CONTRACT,
        "checks": checks,
        "selected_core_count": core_count.tolist(),
        "selected_holdout_count": holdout_count.tolist(),
        "selected_object_purity": selected_label_mass.tolist(),
        "relay_good_reliability": float(reliable_good.mean()),
        "relay_bad_reliability": float(reliable_bad.mean()),
        "visual_gain_over_persistence": audit.metrics[
            "visual_gain_over_persistence"
        ].tolist(),
        "visual_margin_rolled_core": audit.metrics[
            "visual_margin_rolled_core"
        ].tolist(),
    }
    print(json.dumps(report, sort_keys=True))
    require(all(checks.values()), "v65 synthetic contract verification failed")


if __name__ == "__main__":
    main()
