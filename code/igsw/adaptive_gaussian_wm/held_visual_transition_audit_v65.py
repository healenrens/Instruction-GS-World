"""Held-image local observation audit for reliable v65 object transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .native_local_feature_field_v65 import (
    pool_native_local_features_v65,
    select_native_token_frames_v65,
)
from .robust_multitrack_binding_v65 import (
    build_reliable_core_binding_v65,
    fit_robust_transitions_v65,
)


@dataclass(frozen=True)
class HeldVisualAuditBatchV65:
    metrics: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]


def _prefix_identity(features, visibility, reliability, split):
    weight = visibility[:, :split].float() * reliability[:, None]
    identity = (features[:, :split].float() * weight[..., None]).sum(dim=1)
    identity = identity / weight.sum(dim=1).clamp_min(1.0)[..., None]
    return F.normalize(identity, dim=-1, eps=1e-6), weight.sum(dim=1) > 0.0


def _predict_coordinates(source, coefficients):
    design = torch.cat((source.float(), torch.ones_like(source[..., :1])), dim=-1)
    return torch.einsum("bpi,bhid->bhpd", design, coefficients.float())


def _local_predictions(field, coordinates, target_frames, config):
    selected = select_native_token_frames_v65(field, target_frames)
    return pool_native_local_features_v65(
        selected,
        coordinates,
        config.local_radii_pixels,
        config.local_tokens_per_scale,
    )


def _visual_error(
    dino_identity,
    siglip_identity,
    identity_valid,
    dino_prediction,
    siglip_prediction,
    membership,
    reliability,
    minimum_tracks,
):
    dino_error = 1.0 - F.cosine_similarity(
        dino_prediction.features.float(), dino_identity[:, None], dim=-1
    )
    siglip_error = 1.0 - F.cosine_similarity(
        siglip_prediction.features.float(), siglip_identity[:, None], dim=-1
    )
    error = 0.5 * (dino_error + siglip_error)
    point_valid = dino_prediction.valid & siglip_prediction.valid
    point_valid = point_valid & identity_valid[:, None]
    weight = membership[:, None].float() * reliability[:, None]
    weight = weight * point_valid.float()
    value = (error * weight).sum(dim=(1, 2))
    value = value / weight.sum(dim=(1, 2)).clamp_min(1e-6)
    visible_tracks = ((weight > 0.0).any(dim=1)).sum(dim=1)
    valid = visible_tracks >= minimum_tracks
    return value, valid, visible_tracks.float()


def _coordinate_error(prediction, target, visibility, membership, reliability):
    error = (prediction.float() - target.float()).norm(dim=-1)
    weight = visibility.float() * membership[:, None].float()
    weight = weight * reliability[:, None]
    value = (error * weight).sum(dim=(1, 2))
    value = value / weight.sum(dim=(1, 2)).clamp_min(1e-6)
    return value, weight.sum(dim=(1, 2)) > 0.0


def _roll_spatial_tracks(membership, config):
    anchors = len(config.tracker_anchor_fractions)
    points_per_anchor = membership.shape[-1] // anchors
    blocks = membership.reshape(len(membership), anchors, points_per_anchor)
    return torch.roll(blocks, shifts=points_per_anchor // 2, dims=-1).flatten(1)


def held_visual_transition_audit_v65(bundle, sequence_index, config):
    frames = bundle.evidence.visibility.shape[1]
    split = frames // 2
    target_frames = [split - 1 + horizon for horizon in config.transition_horizons]
    if max(target_frames) >= frames:
        raise ValueError("v65 clip cannot hold all transition horizons")
    binding = build_reliable_core_binding_v65(
        bundle.observation,
        bundle.evidence,
        sequence_index,
        config,
        0,
        split,
    )
    source = bundle.evidence.coordinates[:, split - 1]
    correct_coordinates = _predict_coordinates(
        source, binding.selected_coefficients
    )
    persistence_coordinates = source[:, None].expand_as(correct_coordinates)
    target_index = torch.as_tensor(
        target_frames, device=source.device, dtype=torch.long
    )
    target_coordinates = bundle.evidence.coordinates.index_select(1, target_index)
    target_visibility = bundle.evidence.visibility.index_select(1, target_index)
    rolled_core = _roll_spatial_tracks(binding.selected_core, config)
    rolled_fit = fit_robust_transitions_v65(
        bundle.evidence, rolled_core, 0, split, config
    )
    rolled_coordinates = _predict_coordinates(source, rolled_fit.coefficients)
    shuffled_coefficients = torch.roll(
        binding.selected_coefficients, shifts=1, dims=0
    )
    shuffled_coordinates = _predict_coordinates(source, shuffled_coefficients)
    shuffled_valid = binding.selected_valid & torch.roll(
        binding.selected_valid, shifts=1, dims=0
    )
    dino_identity, dino_identity_valid = _prefix_identity(
        bundle.observation.dino,
        bundle.evidence.visibility,
        bundle.evidence.reliability,
        split,
    )
    siglip_identity, siglip_identity_valid = _prefix_identity(
        bundle.observation.siglip,
        bundle.evidence.visibility,
        bundle.evidence.reliability,
        split,
    )
    identity_valid = dino_identity_valid & siglip_identity_valid
    coordinate_sets = {
        "correct": correct_coordinates,
        "persistence": persistence_coordinates,
        "rolled_core": rolled_coordinates,
        "shuffled_sample": shuffled_coordinates,
        "oracle_track": target_coordinates,
    }
    visual, visual_valid, visible_tracks = {}, {}, {}
    for name, coordinates in coordinate_sets.items():
        dino_prediction = _local_predictions(
            bundle.field.dino, coordinates, target_frames, config
        )
        siglip_prediction = _local_predictions(
            bundle.field.siglip, coordinates, target_frames, config
        )
        visual[name], visual_valid[name], visible_tracks[name] = _visual_error(
            dino_identity,
            siglip_identity,
            identity_valid,
            dino_prediction,
            siglip_prediction,
            binding.selected_holdout,
            bundle.evidence.reliability,
            config.minimum_holdout_tracks,
        )
    coordinate, coordinate_valid = {}, {}
    for name, prediction in (
        ("correct", correct_coordinates),
        ("persistence", persistence_coordinates),
        ("rolled_core", rolled_coordinates),
        ("shuffled_sample", shuffled_coordinates),
    ):
        coordinate[name], coordinate_valid[name] = _coordinate_error(
            prediction,
            target_coordinates,
            target_visibility,
            binding.selected_holdout,
            bundle.evidence.reliability,
        )
    audit_valid = binding.selected_valid & visual_valid["correct"]
    rolled_valid = audit_valid & rolled_fit.valid & visual_valid["rolled_core"]
    shuffled_joint_valid = audit_valid & shuffled_valid
    shuffled_joint_valid = shuffled_joint_valid & visual_valid["shuffled_sample"]
    reliability_weight = binding.selected.float()
    reliability_mean = (
        bundle.evidence.reliability * reliability_weight
    ).sum(dim=-1) / reliability_weight.sum(dim=-1).clamp_min(1e-6)
    relay_error_mean = (
        bundle.evidence.relay_error * reliability_weight
    ).sum(dim=-1) / reliability_weight.sum(dim=-1).clamp_min(1e-6)
    metrics = {
        "candidate_valid": binding.selected_valid.float(),
        "audit_valid": audit_valid.float(),
        "rolled_core_valid": rolled_valid.float(),
        "shuffled_sample_valid": shuffled_joint_valid.float(),
        "component_effective_track_count": binding.selected_effective_track_count,
        "holdout_track_count": (binding.selected_holdout > 0.0).sum(dim=-1).float(),
        "visual_holdout_track_count": visible_tracks["correct"],
        "component_membership_fraction": binding.selected.mean(dim=-1),
        "scene_fraction": binding.scene_membership.mean(dim=-1),
        "unknown_fraction": binding.unknown_membership.mean(dim=-1),
        "component_reliability": reliability_mean,
        "component_relay_error": relay_error_mean,
        "prefix_transition_residual": binding.selected_prefix_transition_residual,
        "correct_visual_error": visual["correct"],
        "persistence_visual_error": visual["persistence"],
        "rolled_core_visual_error": visual["rolled_core"],
        "shuffled_sample_visual_error": visual["shuffled_sample"],
        "oracle_track_visual_error": visual["oracle_track"],
        "visual_gain_over_persistence": visual["persistence"] - visual["correct"],
        "visual_margin_rolled_core": visual["rolled_core"] - visual["correct"],
        "visual_margin_shuffled_sample": (
            visual["shuffled_sample"] - visual["correct"]
        ),
        "correct_coordinate_error": coordinate["correct"],
        "persistence_coordinate_error": coordinate["persistence"],
        "rolled_core_coordinate_error": coordinate["rolled_core"],
        "shuffled_sample_coordinate_error": coordinate["shuffled_sample"],
        "coordinate_gain_over_persistence": (
            coordinate["persistence"] - coordinate["correct"]
        ),
    }
    valid = {
        "candidate_valid": torch.ones_like(audit_valid),
        "audit_valid": torch.ones_like(audit_valid),
        "rolled_core_valid": torch.ones_like(audit_valid),
        "shuffled_sample_valid": torch.ones_like(audit_valid),
    }
    for name in (
        "component_effective_track_count",
        "holdout_track_count",
        "visual_holdout_track_count",
        "component_membership_fraction",
        "scene_fraction",
        "unknown_fraction",
        "component_reliability",
        "component_relay_error",
        "prefix_transition_residual",
        "correct_visual_error",
        "persistence_visual_error",
        "oracle_track_visual_error",
        "visual_gain_over_persistence",
        "correct_coordinate_error",
        "persistence_coordinate_error",
        "coordinate_gain_over_persistence",
    ):
        valid[name] = audit_valid
    for name in (
        "rolled_core_visual_error",
        "visual_margin_rolled_core",
        "rolled_core_coordinate_error",
    ):
        valid[name] = rolled_valid
    for name in (
        "shuffled_sample_visual_error",
        "visual_margin_shuffled_sample",
        "shuffled_sample_coordinate_error",
    ):
        valid[name] = shuffled_joint_valid
    valid["correct_coordinate_error"] = audit_valid & coordinate_valid["correct"]
    valid["persistence_coordinate_error"] = audit_valid & coordinate_valid["persistence"]
    valid["coordinate_gain_over_persistence"] = (
        valid["correct_coordinate_error"] & valid["persistence_coordinate_error"]
    )
    valid["rolled_core_coordinate_error"] = rolled_valid & coordinate_valid["rolled_core"]
    valid["shuffled_sample_coordinate_error"] = (
        shuffled_joint_valid & coordinate_valid["shuffled_sample"]
    )
    return HeldVisualAuditBatchV65(metrics=metrics, valid=valid)
