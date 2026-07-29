"""Root-gated, locally ungated object residual fields for stable transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .object_centered_residual_field import (
    ResidualFieldFit,
    fit_residual_fields,
    geometric_object_gates,
    object_residual_design,
    render_geometric_residual_field,
)
from .residual_field_linear import RidgeSolution, ridge_solution


@dataclass
class StableResidualFit:
    reference: ResidualFieldFit
    solutions: dict[int, RidgeSolution]


def fit_stable_residual_field(
    tokens,
    slots,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    budgets: tuple[int, ...],
    ridge: float,
    minimum_utility_fraction: float,
    maximum_scene_fraction: float,
) -> StableResidualFit:
    reference = fit_residual_fields(
        tokens,
        slots,
        features,
        coordinates,
        valid,
        budgets=budgets,
        ridge=ridge,
        minimum_utility_fraction=minimum_utility_fraction,
        maximum_scene_fraction=maximum_scene_fraction,
    )
    gates = geometric_object_gates(reference.geometry, coordinates)
    solutions = {}
    for budget in budgets:
        design = object_residual_design(
            reference.geometry,
            coordinates,
            gates,
            budget,
            gate_local_residuals=False,
        )
        solutions[budget] = ridge_solution(design, features, valid, ridge)
    return StableResidualFit(reference=reference, solutions=solutions)


def render_stable_residual_field(
    fit: StableResidualFit,
    coordinates: torch.Tensor,
    *,
    local_budget: int,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_feature_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    if local_budget not in fit.solutions:
        raise ValueError(f"stable residual solution has no budget {local_budget}")
    geometry = fit.reference.geometry
    gates = geometric_object_gates(
        geometry,
        coordinates,
        target_object_centers,
        object_scale_ratio,
    )
    design = object_residual_design(
        geometry,
        coordinates,
        gates,
        local_budget,
        target_object_centers,
        object_scale_ratio,
        gate_local_residuals=False,
    )
    coefficients = fit.solutions[local_budget].coefficients.clone()
    if object_feature_delta is not None:
        expected = (geometry.object_centers.shape[0], coefficients.shape[-1])
        if object_feature_delta.shape != expected:
            raise ValueError(f"object feature delta must have shape {expected}")
        for root_index, group_tensor in enumerate(geometry.group_ids[:-1]):
            coefficients[root_index] += object_feature_delta[int(group_tensor)].float()
    return design @ coefficients


def _feature_rms_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (left.float() - right.float()).square().mean(dim=-1)
    weight = valid.float()
    return ((error * weight).sum() / weight.sum().clamp_min(1.0)).sqrt()


def transport_sensitivity(
    fit: StableResidualFit,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    local_budget: int,
    center_fraction: float,
    scale_log_delta: float,
) -> dict[str, torch.Tensor]:
    if center_fraction <= 0.0 or scale_log_delta <= 0.0:
        raise ValueError("transport perturbations must be positive")
    geometry = fit.reference.geometry
    object_count = geometry.object_centers.shape[0]
    if not bool(geometry.object_active.any()):
        raise RuntimeError("transport sensitivity requires an active object")
    index = torch.arange(object_count, device=coordinates.device, dtype=torch.float32)
    angle = index * (2.0 * math.pi / max(object_count, 1))
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    active = geometry.object_active.float()[:, None]
    extent = geometry.coordinate_maximum - geometry.coordinate_minimum
    center_delta = direction * extent[None] * center_fraction * active
    center_magnitude = (
        center_delta.square()
        .sum(dim=-1)[geometry.object_active]
        .mean()
        .sqrt()
        .clamp_min(1e-8)
    )
    scale_sign = torch.where((index.long() % 2) == 0, 1.0, -1.0)
    log_scale = scale_sign * scale_log_delta * geometry.object_active.float()
    scale_ratio = log_scale.exp()
    scale_magnitude = (
        log_scale[geometry.object_active].square().mean().sqrt().clamp_min(1e-8)
    )

    stable_base = render_stable_residual_field(
        fit, coordinates, local_budget=local_budget
    )
    stable_center = render_stable_residual_field(
        fit,
        coordinates,
        local_budget=local_budget,
        target_object_centers=geometry.object_centers + center_delta,
    )
    stable_scale = render_stable_residual_field(
        fit,
        coordinates,
        local_budget=local_budget,
        object_scale_ratio=scale_ratio,
    )
    gated_base = render_geometric_residual_field(
        fit.reference, coordinates, local_budget=local_budget
    )
    gated_center = render_geometric_residual_field(
        fit.reference,
        coordinates,
        local_budget=local_budget,
        target_object_centers=geometry.object_centers + center_delta,
    )
    gated_scale = render_geometric_residual_field(
        fit.reference,
        coordinates,
        local_budget=local_budget,
        object_scale_ratio=scale_ratio,
    )
    return {
        "stable_center": _feature_rms_difference(stable_center, stable_base, valid)
        / center_magnitude,
        "stable_scale": _feature_rms_difference(stable_scale, stable_base, valid)
        / scale_magnitude,
        "gated_center": _feature_rms_difference(gated_center, gated_base, valid)
        / center_magnitude,
        "gated_scale": _feature_rms_difference(gated_scale, gated_base, valid)
        / scale_magnitude,
    }
