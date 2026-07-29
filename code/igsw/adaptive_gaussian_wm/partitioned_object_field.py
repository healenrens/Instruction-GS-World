"""Normalized object-expert feature fields with transportable support mixtures."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .object_centered_carrier_probe import build_frame_groups, feature_error
from .object_centered_residual_field import (
    ResidualFieldFit,
    ResidualFieldGeometry,
    fit_residual_fields,
    geometric_object_gates,
)
from .residual_field_linear import RidgeSolution, rbf, ridge_solution


@dataclass
class ObjectFeatureExpert:
    group_id: int
    carrier_indices: torch.Tensor
    component_mass: torch.Tensor
    solution: RidgeSolution


@dataclass
class PartitionedBudgetFit:
    experts: tuple[ObjectFeatureExpert, ...]
    prediction: torch.Tensor
    gates: torch.Tensor
    error: torch.Tensor
    support_js: torch.Tensor
    maximum_condition: torch.Tensor
    coefficient_rms: torch.Tensor


@dataclass
class PartitionedObjectFieldFit:
    reference: ResidualFieldFit
    budgets: dict[int, PartitionedBudgetFit]
    geometric_support_js: torch.Tensor


def _canonical_coordinates(
    coordinates: torch.Tensor,
    center: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    difference = coordinates.float() - center.float()
    return torch.linalg.solve(transform.float(), difference[..., None]).squeeze(-1)


def _query_coordinates(
    geometry: ResidualFieldGeometry,
    coordinates: torch.Tensor,
    group: int,
    centers: torch.Tensor,
    scale_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    object_count = geometry.object_centers.shape[0]
    if group == object_count:
        return coordinates.float(), coordinates.new_ones((), dtype=torch.float32)
    transform = geometry.object_transforms[group].float() * scale_ratio[group]
    canonical = _canonical_coordinates(coordinates, centers[group], transform)
    return canonical, torch.linalg.det(transform).abs().clamp_min(1e-8)


def _selected_indices(
    geometry: ResidualFieldGeometry,
    group: int,
    local_budget: int,
) -> torch.Tensor:
    selected_groups = geometry.carrier_group[:local_budget]
    return torch.nonzero(selected_groups == group, as_tuple=False).flatten()


def _expert_design(
    geometry: ResidualFieldGeometry,
    query: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    root = query.new_ones((query.shape[0], 1), dtype=torch.float32)
    if indices.numel() == 0:
        return root
    local = rbf(
        query,
        geometry.relative_center[indices],
        geometry.relative_covariance[indices],
    )
    return torch.cat((root, local), dim=-1)


def _normal_density(
    query: torch.Tensor,
    centers: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    basis = rbf(query, centers, covariance)
    determinant = torch.linalg.det(covariance.float()).clamp_min(1e-12)
    normalizer = 1.0 / (2.0 * math.pi * determinant.sqrt())
    return basis * normalizer[None]


def _density_design(
    geometry: ResidualFieldGeometry,
    query: torch.Tensor,
    group: int,
    indices: torch.Tensor,
    transform_determinant: torch.Tensor,
) -> torch.Tensor:
    object_count = geometry.object_centers.shape[0]
    if group == object_count:
        extent = (geometry.coordinate_maximum - geometry.coordinate_minimum).clamp_min(
            1e-3
        )
        root = query.new_full((query.shape[0], 1), 1.0 / float(extent.prod()))
    else:
        center = query.new_zeros((1, 2), dtype=torch.float32)
        covariance = torch.eye(2, device=query.device, dtype=torch.float32)[None]
        root = _normal_density(query, center, covariance) / transform_determinant
    if indices.numel() == 0:
        return root
    local = _normal_density(
        query,
        geometry.relative_center[indices],
        geometry.relative_covariance[indices],
    )
    if group != object_count:
        local = local / transform_determinant
    return torch.cat((root, local), dim=-1)


def _fit_component_mass(
    density: torch.Tensor,
    support: torch.Tensor,
    iterations: int = 8,
) -> torch.Tensor:
    component_count = density.shape[1]
    mass = density.new_full((component_count,), 1.0 / component_count)
    weight = support.float().clamp_min(0.0)
    for _ in range(iterations):
        weighted_density = density * mass[None]
        responsibility = weighted_density / weighted_density.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        mass = (responsibility * weight[:, None]).sum(dim=0)
        mass = mass.clamp_min(1e-6)
        mass = mass / mass.sum()
    return mass


def partition_support_js(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.float().clamp_min(1e-8)
    target = target.float().clamp_min(1e-8)
    target = target / target.sum(dim=-1, keepdim=True)
    mixture = 0.5 * (prediction + target)
    divergence = 0.5 * (
        (prediction * (prediction.log() - mixture.log())).sum(dim=-1)
        + (target * (target.log() - mixture.log())).sum(dim=-1)
    )
    weight = valid.float()
    return (divergence * weight).sum() / weight.sum().clamp_min(1.0)


def _transport_arguments(
    geometry: ResidualFieldGeometry,
    coordinates: torch.Tensor,
    target_object_centers: torch.Tensor | None,
    object_scale_ratio: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    object_count = geometry.object_centers.shape[0]
    centers = (
        geometry.object_centers
        if target_object_centers is None
        else target_object_centers.float()
    )
    if centers.shape != (object_count, 2):
        raise ValueError("target object centers must have shape [K,2]")
    ratio = (
        coordinates.new_ones(object_count, dtype=torch.float32)
        if object_scale_ratio is None
        else object_scale_ratio.float()
    )
    if ratio.shape != (object_count,):
        raise ValueError("object scale ratio must have shape [K]")
    return centers, ratio.clamp(0.25, 4.0)


def _partition_gates(
    geometry: ResidualFieldGeometry,
    experts: tuple[ObjectFeatureExpert, ...],
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    scale_ratio: torch.Tensor,
    object_presence: torch.Tensor | None,
) -> torch.Tensor:
    scores = []
    for root_index, expert in enumerate(experts):
        query, determinant = _query_coordinates(
            geometry,
            coordinates,
            expert.group_id,
            centers,
            scale_ratio,
        )
        density = _density_design(
            geometry,
            query,
            expert.group_id,
            expert.carrier_indices,
            determinant,
        )
        score = geometry.group_priors[root_index] * (density @ expert.component_mass)
        if object_presence is not None and expert.group_id < object_presence.shape[0]:
            score = score * object_presence[expert.group_id]
        scores.append(score)
    stacked = torch.stack(scores, dim=-1).clamp_min(1e-12)
    return stacked / stacked.sum(dim=-1, keepdim=True)


def _render_budget(
    geometry: ResidualFieldGeometry,
    budget: PartitionedBudgetFit,
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    scale_ratio: torch.Tensor,
    object_presence: torch.Tensor | None,
    object_feature_delta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    object_count = geometry.object_centers.shape[0]
    gates = _partition_gates(
        geometry,
        budget.experts,
        coordinates,
        centers,
        scale_ratio,
        object_presence,
    )
    outputs = []
    for expert in budget.experts:
        query, _ = _query_coordinates(
            geometry,
            coordinates,
            expert.group_id,
            centers,
            scale_ratio,
        )
        design = _expert_design(geometry, query, expert.carrier_indices)
        output = design @ expert.solution.coefficients
        if object_feature_delta is not None and expert.group_id < object_count:
            output = output + object_feature_delta[expert.group_id]
        outputs.append(output)
    stacked = torch.stack(outputs, dim=1)
    return (gates[..., None] * stacked).sum(dim=1), gates


def fit_partitioned_object_field(
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
) -> PartitionedObjectFieldFit:
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
    groups = build_frame_groups(tokens, slots, features, coordinates, valid)
    geometry = reference.geometry
    target = groups.support[geometry.group_ids].transpose(0, 1).float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    geometric = geometric_object_gates(geometry, coordinates)
    geometric_js = partition_support_js(geometric, target, valid)
    centers, ratio = _transport_arguments(geometry, coordinates, None, None)
    results = {}
    for local_budget in budgets:
        experts = []
        for root_index, group_tensor in enumerate(geometry.group_ids):
            group = int(group_tensor)
            indices = _selected_indices(geometry, group, local_budget)
            query, determinant = _query_coordinates(
                geometry, coordinates, group, centers, ratio
            )
            density = _density_design(
                geometry,
                query,
                group,
                indices,
                determinant,
            )
            support = target[:, root_index] * valid.float()
            component_mass = _fit_component_mass(density, support)
            solution = ridge_solution(
                _expert_design(geometry, query, indices),
                features,
                support,
                ridge,
            )
            experts.append(
                ObjectFeatureExpert(group, indices, component_mass, solution)
            )
        experts_tuple = tuple(experts)
        placeholder = PartitionedBudgetFit(
            experts=experts_tuple,
            prediction=features.new_empty(0),
            gates=features.new_empty(0),
            error=features.new_zeros(()),
            support_js=features.new_zeros(()),
            maximum_condition=features.new_zeros(()),
            coefficient_rms=features.new_zeros(()),
        )
        prediction, gates = _render_budget(
            geometry, placeholder, coordinates, centers, ratio, None, None
        )
        conditions = torch.stack(
            [expert.solution.condition_number for expert in experts_tuple]
        )
        coefficients = torch.cat(
            [expert.solution.coefficients.flatten() for expert in experts_tuple]
        )
        results[local_budget] = PartitionedBudgetFit(
            experts=experts_tuple,
            prediction=prediction,
            gates=gates,
            error=feature_error(prediction, features, valid),
            support_js=partition_support_js(gates, target, valid),
            maximum_condition=conditions.max(),
            coefficient_rms=coefficients.square().mean().sqrt(),
        )
    return PartitionedObjectFieldFit(reference, results, geometric_js)


def render_partitioned_object_field(
    fit: PartitionedObjectFieldFit,
    coordinates: torch.Tensor,
    *,
    local_budget: int,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_presence: torch.Tensor | None = None,
    object_feature_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    if local_budget not in fit.budgets:
        raise ValueError(f"partitioned field has no budget {local_budget}")
    geometry = fit.reference.geometry
    object_count = geometry.object_centers.shape[0]
    if object_presence is not None:
        if object_presence.shape != (object_count,):
            raise ValueError("object presence must have shape [K]")
        object_presence = object_presence.float().clamp(0.0, 1.0)
    if object_feature_delta is not None:
        expected = (
            object_count,
            fit.budgets[local_budget].prediction.shape[-1],
        )
        if object_feature_delta.shape != expected:
            raise ValueError(f"object feature delta must have shape {expected}")
    centers, ratio = _transport_arguments(
        geometry, coordinates, target_object_centers, object_scale_ratio
    )
    prediction, _ = _render_budget(
        geometry,
        fit.budgets[local_budget],
        coordinates,
        centers,
        ratio,
        object_presence,
        object_feature_delta,
    )
    return prediction


def query_partitioned_object_gates(
    fit: PartitionedObjectFieldFit,
    coordinates: torch.Tensor,
    *,
    local_budget: int,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_presence: torch.Tensor | None = None,
) -> torch.Tensor:
    if local_budget not in fit.budgets:
        raise ValueError(f"partitioned field has no budget {local_budget}")
    geometry = fit.reference.geometry
    object_count = geometry.object_centers.shape[0]
    if object_presence is not None:
        if object_presence.shape != (object_count,):
            raise ValueError("object presence must have shape [K]")
        object_presence = object_presence.float().clamp(0.0, 1.0)
    centers, ratio = _transport_arguments(
        geometry, coordinates, target_object_centers, object_scale_ratio
    )
    return _partition_gates(
        geometry,
        fit.budgets[local_budget].experts,
        coordinates,
        centers,
        ratio,
        object_presence,
    )
