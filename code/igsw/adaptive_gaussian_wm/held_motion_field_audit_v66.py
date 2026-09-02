"""Cross-fitted future-image audit for compact object motion fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .held_visual_transition_audit_v65 import (
    _coordinate_error,
    _local_predictions,
    _prefix_identity,
    _roll_spatial_tracks,
    _visual_error,
)
from .object_motion_field_v66 import (
    fit_object_motion_models_v66,
    predict_affine_v66,
    predict_motion_field_v66,
    predict_translation_v66,
    roll_motion_field_fit_v66,
)
from .robust_multitrack_binding_v65 import build_reliable_core_binding_v65


@dataclass(frozen=True)
class HeldMotionFieldAuditBatchV66:
    metrics: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]


def held_motion_field_audit_v66(bundle, sequence_index, config):
    frames = bundle.evidence.visibility.shape[1]
    split = frames // 2
    target_frames = [split - 1 + horizon for horizon in config.transition_horizons]
    if max(target_frames) >= frames:
        raise ValueError("v66 clip cannot hold all transition horizons")
    binding = build_reliable_core_binding_v65(
        bundle.observation,
        bundle.evidence,
        sequence_index,
        config,
        0,
        split,
    )
    source = bundle.evidence.coordinates[:, split - 1]
    correct_fit = fit_object_motion_models_v66(
        bundle.evidence, binding.selected_core, 0, split, config
    )
    rolled_core = _roll_spatial_tracks(binding.selected_core, config)
    rolled_fit = fit_object_motion_models_v66(
        bundle.evidence, rolled_core, 0, split, config
    )
    shuffled_fit = roll_motion_field_fit_v66(correct_fit)
    coordinates = {
        "persistence": source[:, None].expand(
            -1, len(config.transition_horizons), -1, -1
        ),
        "translation": predict_translation_v66(source, correct_fit),
        "affine": predict_affine_v66(source, correct_fit),
        "motion_field": predict_motion_field_v66(source, correct_fit),
        "rolled_field": predict_motion_field_v66(source, rolled_fit),
        "shuffled_field": predict_motion_field_v66(source, shuffled_fit),
    }
    target_index = torch.as_tensor(
        target_frames, device=source.device, dtype=torch.long
    )
    target_coordinates = bundle.evidence.coordinates.index_select(1, target_index)
    target_visibility = bundle.evidence.visibility.index_select(1, target_index)
    coordinates["oracle_track"] = target_coordinates
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
    visual, visual_valid, visible_tracks = {}, {}, {}
    for name, prediction in coordinates.items():
        dino_prediction = _local_predictions(
            bundle.field.dino, prediction, target_frames, config
        )
        siglip_prediction = _local_predictions(
            bundle.field.siglip, prediction, target_frames, config
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
    for name in (
        "persistence",
        "translation",
        "affine",
        "motion_field",
        "rolled_field",
        "shuffled_field",
    ):
        coordinate[name], coordinate_valid[name] = _coordinate_error(
            coordinates[name],
            target_coordinates,
            target_visibility,
            binding.selected_holdout,
            bundle.evidence.reliability,
        )
    base_valid = binding.selected_valid & visual_valid["persistence"]
    translation_valid = base_valid & correct_fit.translation_valid
    translation_valid = translation_valid & visual_valid["translation"]
    affine_valid = base_valid & correct_fit.affine_valid & visual_valid["affine"]
    field_valid = base_valid & correct_fit.field_valid & visual_valid["motion_field"]
    rolled_valid = field_valid & rolled_fit.field_valid & visual_valid["rolled_field"]
    shuffled_valid = field_valid & shuffled_fit.field_valid
    shuffled_valid = shuffled_valid & visual_valid["shuffled_field"]
    reliability_weight = binding.selected.float()
    reliability_mean = (
        bundle.evidence.reliability * reliability_weight
    ).sum(dim=-1) / reliability_weight.sum(dim=-1).clamp_min(1e-6)
    metrics = {
        "candidate_valid": binding.selected_valid.float(),
        "audit_valid": field_valid.float(),
        "high_change_score": visual["persistence"],
        "component_effective_track_count": binding.selected_effective_track_count,
        "holdout_track_count": (binding.selected_holdout > 0.0).sum(dim=-1).float(),
        "component_reliability": reliability_mean,
        "translation_prefix_error": correct_fit.translation_prefix_error,
        "affine_prefix_error": correct_fit.affine_prefix_error,
        "motion_field_prefix_error": correct_fit.field_prefix_error,
        "persistence_visual_error": visual["persistence"],
        "translation_visual_error": visual["translation"],
        "affine_visual_error": visual["affine"],
        "motion_field_visual_error": visual["motion_field"],
        "rolled_field_visual_error": visual["rolled_field"],
        "shuffled_field_visual_error": visual["shuffled_field"],
        "oracle_track_visual_error": visual["oracle_track"],
        "translation_gain_over_persistence": (
            visual["persistence"] - visual["translation"]
        ),
        "affine_gain_over_persistence": visual["persistence"] - visual["affine"],
        "motion_field_gain_over_persistence": (
            visual["persistence"] - visual["motion_field"]
        ),
        "motion_field_margin_over_translation": (
            visual["translation"] - visual["motion_field"]
        ),
        "motion_field_margin_over_affine": (
            visual["affine"] - visual["motion_field"]
        ),
        "motion_field_margin_over_rolled": (
            visual["rolled_field"] - visual["motion_field"]
        ),
        "motion_field_margin_over_shuffled": (
            visual["shuffled_field"] - visual["motion_field"]
        ),
        "oracle_visual_headroom": (
            visual["persistence"] - visual["oracle_track"]
        ),
        "persistence_coordinate_error": coordinate["persistence"],
        "translation_coordinate_error": coordinate["translation"],
        "affine_coordinate_error": coordinate["affine"],
        "motion_field_coordinate_error": coordinate["motion_field"],
    }
    valid = {
        "candidate_valid": torch.ones_like(base_valid),
        "audit_valid": torch.ones_like(base_valid),
    }
    for name in (
        "high_change_score",
        "component_effective_track_count",
        "holdout_track_count",
        "component_reliability",
        "persistence_visual_error",
        "persistence_coordinate_error",
    ):
        valid[name] = base_valid
    oracle_valid = base_valid & visual_valid["oracle_track"]
    valid["oracle_track_visual_error"] = oracle_valid
    valid["oracle_visual_headroom"] = oracle_valid
    for name in (
        "translation_prefix_error",
        "translation_visual_error",
        "translation_gain_over_persistence",
        "translation_coordinate_error",
    ):
        valid[name] = translation_valid
    for name in (
        "affine_prefix_error",
        "affine_visual_error",
        "affine_gain_over_persistence",
        "affine_coordinate_error",
    ):
        valid[name] = affine_valid
    for name in (
        "motion_field_prefix_error",
        "motion_field_visual_error",
        "motion_field_gain_over_persistence",
        "motion_field_margin_over_translation",
        "motion_field_margin_over_affine",
        "motion_field_coordinate_error",
    ):
        valid[name] = field_valid
    valid["motion_field_margin_over_translation"] = field_valid & translation_valid
    valid["motion_field_margin_over_affine"] = field_valid & affine_valid
    valid["rolled_field_visual_error"] = rolled_valid
    valid["motion_field_margin_over_rolled"] = rolled_valid
    valid["shuffled_field_visual_error"] = shuffled_valid
    valid["motion_field_margin_over_shuffled"] = shuffled_valid
    valid["persistence_coordinate_error"] = (
        base_valid & coordinate_valid["persistence"]
    )
    valid["translation_coordinate_error"] = (
        translation_valid & coordinate_valid["translation"]
    )
    valid["affine_coordinate_error"] = affine_valid & coordinate_valid["affine"]
    valid["motion_field_coordinate_error"] = (
        field_valid & coordinate_valid["motion_field"]
    )
    return HeldMotionFieldAuditBatchV66(metrics=metrics, valid=valid)
