"""Counterfactual transport variants for partitioned object fields."""

from __future__ import annotations

import torch

from .partitioned_object_field import (
    partition_support_js,
    query_partitioned_object_gates,
    render_partitioned_object_field,
)


def render_transport_variants(
    fit,
    coordinates: torch.Tensor,
    budget: int,
    predicted_centers: torch.Tensor,
    predicted_scale_ratio: torch.Tensor,
    predicted_presence: torch.Tensor,
    predicted_feature_delta: torch.Tensor,
    target_centers: torch.Tensor,
    target_scale_ratio: torch.Tensor,
    target_presence: torch.Tensor,
    target_feature_delta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    common = {"local_budget": budget}
    predicted_geometry = {
        "target_object_centers": predicted_centers,
        "object_scale_ratio": predicted_scale_ratio,
    }
    target_geometry = {
        "target_object_centers": target_centers,
        "object_scale_ratio": target_scale_ratio,
    }
    return {
        "persistence": render_partitioned_object_field(fit, coordinates, **common),
        "feature_only": render_partitioned_object_field(
            fit,
            coordinates,
            object_feature_delta=predicted_feature_delta,
            **common,
        ),
        "center_only": render_partitioned_object_field(
            fit,
            coordinates,
            target_object_centers=predicted_centers,
            **common,
        ),
        "scale_only": render_partitioned_object_field(
            fit,
            coordinates,
            object_scale_ratio=predicted_scale_ratio,
            **common,
        ),
        "presence_only": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=predicted_presence,
            **common,
        ),
        "predicted_geometry": render_partitioned_object_field(
            fit, coordinates, **predicted_geometry, **common
        ),
        "predicted_state": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=predicted_presence,
            **predicted_geometry,
            **common,
        ),
        "full": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=predicted_presence,
            object_feature_delta=predicted_feature_delta,
            **predicted_geometry,
            **common,
        ),
        "target_feature_only": render_partitioned_object_field(
            fit,
            coordinates,
            object_feature_delta=target_feature_delta,
            **common,
        ),
        "target_geometry": render_partitioned_object_field(
            fit, coordinates, **target_geometry, **common
        ),
        "target_presence_only": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=target_presence,
            **common,
        ),
        "target_state": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=target_presence,
            **target_geometry,
            **common,
        ),
        "target_state_predicted_feature": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=target_presence,
            object_feature_delta=predicted_feature_delta,
            **target_geometry,
            **common,
        ),
        "predicted_state_target_feature": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=predicted_presence,
            object_feature_delta=target_feature_delta,
            **predicted_geometry,
            **common,
        ),
        "target_state_target_feature": render_partitioned_object_field(
            fit,
            coordinates,
            object_presence=target_presence,
            object_feature_delta=target_feature_delta,
            **target_geometry,
            **common,
        ),
    }


def state_prediction_errors(
    current_center: torch.Tensor,
    current_scale: torch.Tensor,
    current_presence: torch.Tensor,
    predicted_center: torch.Tensor,
    predicted_scale: torch.Tensor,
    predicted_presence: torch.Tensor,
    predicted_feature: torch.Tensor,
    target_center: torch.Tensor,
    target_scale: torch.Tensor,
    target_presence: torch.Tensor,
    target_feature: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def weighted_rms(
        left: torch.Tensor,
        right: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        error = (left.float() - right.float()).square()
        if error.ndim > 1:
            error = error.flatten(1).mean(dim=-1)
        return (
            (error * weight.float()).sum() / weight.float().sum().clamp_min(1e-6)
        ).sqrt()

    tracked = torch.maximum(current_presence, target_presence).clamp_min(1e-3)
    target_visible = target_presence.clamp_min(1e-3)
    all_slots = torch.ones_like(target_presence)

    return {
        "persistence_center_rmse": weighted_rms(current_center, target_center, tracked),
        "predicted_center_rmse": weighted_rms(predicted_center, target_center, tracked),
        "persistence_log_scale_rmse": weighted_rms(
            current_scale.clamp_min(1e-6).log(),
            target_scale.clamp_min(1e-6).log(),
            tracked,
        ),
        "predicted_log_scale_rmse": weighted_rms(
            predicted_scale.clamp_min(1e-6).log(),
            target_scale.clamp_min(1e-6).log(),
            tracked,
        ),
        "predicted_feature_rmse": weighted_rms(
            predicted_feature, target_feature, target_visible
        ),
        "persistence_presence_rmse": weighted_rms(
            current_presence, target_presence, all_slots
        ),
        "predicted_presence_rmse": weighted_rms(
            predicted_presence, target_presence, all_slots
        ),
    }


def transported_support_errors(
    fit,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    target_support: torch.Tensor,
    budget: int,
    predicted_centers: torch.Tensor,
    predicted_scale_ratio: torch.Tensor,
    predicted_presence: torch.Tensor,
    target_centers: torch.Tensor,
    target_scale_ratio: torch.Tensor,
    target_presence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    object_count = fit.reference.geometry.object_centers.shape[0]
    if target_support.shape != (object_count + 1, coordinates.shape[0]):
        raise ValueError("target support must have shape [K+1,N]")
    selected_objects = fit.reference.geometry.group_ids[:-1]
    selected_support = target_support[selected_objects].float()
    residual_scene = (1.0 - selected_support.sum(dim=0)).clamp(0.0, 1.0)
    aligned_target = torch.cat((selected_support, residual_scene[None]), dim=0)
    aligned_target = aligned_target.transpose(0, 1)
    common = {"local_budget": budget}
    predicted_geometry = {
        "target_object_centers": predicted_centers,
        "object_scale_ratio": predicted_scale_ratio,
    }
    target_geometry = {
        "target_object_centers": target_centers,
        "object_scale_ratio": target_scale_ratio,
    }
    gates = {
        "support_persistence_js": query_partitioned_object_gates(
            fit, coordinates, **common
        ),
        "support_predicted_geometry_js": query_partitioned_object_gates(
            fit, coordinates, **predicted_geometry, **common
        ),
        "support_predicted_state_js": query_partitioned_object_gates(
            fit,
            coordinates,
            object_presence=predicted_presence,
            **predicted_geometry,
            **common,
        ),
        "support_target_geometry_js": query_partitioned_object_gates(
            fit, coordinates, **target_geometry, **common
        ),
        "support_target_state_js": query_partitioned_object_gates(
            fit,
            coordinates,
            object_presence=target_presence,
            **target_geometry,
            **common,
        ),
    }
    result = {
        name: partition_support_js(value, aligned_target, valid)
        for name, value in gates.items()
    }
    unselected = torch.ones(object_count, dtype=torch.bool, device=coordinates.device)
    unselected[selected_objects] = False
    omitted = target_support[:-1][unselected].sum(dim=0)
    weight = valid.float()
    result["unrepresented_object_mass"] = (
        omitted * weight
    ).sum() / weight.sum().clamp_min(1.0)
    return result
