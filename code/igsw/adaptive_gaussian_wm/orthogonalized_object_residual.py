"""Globally fitted, whitened residual fields over adaptive object partitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .object_centered_carrier_probe import feature_error
from .object_centered_residual_field import ResidualFieldGeometry
from .partitioned_object_field import (
    PartitionedObjectFieldFit,
    fit_partitioned_object_field,
    query_partitioned_object_gates,
)
from .residual_field_linear import rbf


@dataclass
class OrthogonalizedResidualSolution:
    prediction: torch.Tensor
    normalized_coefficients: torch.Tensor
    column_scale: torch.Tensor
    active_columns: torch.Tensor
    error: torch.Tensor
    retained_rank: torch.Tensor
    full_rank: int
    effective_condition: torch.Tensor
    retained_energy: torch.Tensor
    coefficient_rms: torch.Tensor


@dataclass
class OrthogonalizedObjectResidualFit:
    partition: PartitionedObjectFieldFit
    solutions: dict[int, OrthogonalizedResidualSolution]


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


def _group_coordinates(
    geometry: ResidualFieldGeometry,
    coordinates: torch.Tensor,
    group: int,
    centers: torch.Tensor,
    scale_ratio: torch.Tensor,
) -> torch.Tensor:
    object_count = geometry.object_centers.shape[0]
    if group == object_count:
        return coordinates.float()
    transform = geometry.object_transforms[group].float() * scale_ratio[group]
    difference = coordinates.float() - centers[group].float()
    return torch.linalg.solve(transform, difference[..., None]).squeeze(-1)


def _validated_presence(
    geometry: ResidualFieldGeometry,
    presence: torch.Tensor | None,
) -> torch.Tensor | None:
    if presence is None:
        return None
    expected = (geometry.object_centers.shape[0],)
    if presence.shape != expected:
        raise ValueError(f"object presence must have shape {expected}")
    return presence.float().clamp(0.0, 1.0)


def orthogonalized_residual_design(
    fit: OrthogonalizedObjectResidualFit,
    coordinates: torch.Tensor,
    *,
    local_budget: int,
    root_object_centers: torch.Tensor | None = None,
    root_scale_ratio: torch.Tensor | None = None,
    root_presence: torch.Tensor | None = None,
    local_object_centers: torch.Tensor | None = None,
    local_scale_ratio: torch.Tensor | None = None,
    local_presence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_budget not in fit.partition.budgets:
        raise ValueError(f"partition has no budget {local_budget}")
    geometry = fit.partition.reference.geometry
    root_presence = _validated_presence(geometry, root_presence)
    local_presence = _validated_presence(geometry, local_presence)
    root_centers, root_ratio = _transport_arguments(
        geometry, coordinates, root_object_centers, root_scale_ratio
    )
    local_centers, local_ratio = _transport_arguments(
        geometry, coordinates, local_object_centers, local_scale_ratio
    )
    root_gates = query_partitioned_object_gates(
        fit.partition,
        coordinates,
        local_budget=local_budget,
        target_object_centers=root_centers,
        object_scale_ratio=root_ratio,
        object_presence=root_presence,
    )
    local_gates = query_partitioned_object_gates(
        fit.partition,
        coordinates,
        local_budget=local_budget,
        target_object_centers=local_centers,
        object_scale_ratio=local_ratio,
        object_presence=local_presence,
    )
    carrier_count = min(local_budget, geometry.carrier_group.shape[0])
    local_columns = coordinates.new_zeros(
        (coordinates.shape[0], carrier_count), dtype=torch.float32
    )
    for root_index, expert in enumerate(fit.partition.budgets[local_budget].experts):
        indices = expert.carrier_indices
        if indices.numel() == 0:
            continue
        query = _group_coordinates(
            geometry,
            coordinates,
            expert.group_id,
            local_centers,
            local_ratio,
        )
        basis = rbf(
            query,
            geometry.relative_center[indices],
            geometry.relative_covariance[indices],
        )
        local_columns[:, indices] = local_gates[:, root_index : root_index + 1] * basis
    return torch.cat((root_gates, local_columns), dim=-1), root_gates


def _truncated_svd_solution(
    design: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    ridge: float,
    relative_singular_cutoff: float,
    minimum_column_fraction: float,
) -> OrthogonalizedResidualSolution:
    weight = valid.float().sqrt()[:, None]
    weighted_design = design.float() * weight
    weighted_target = target.float() * weight
    raw_scale = weighted_design.square().sum(dim=0).sqrt()
    maximum_scale = raw_scale.max().clamp_min(1e-8)
    active = raw_scale >= maximum_scale * minimum_column_fraction
    if not bool(active.any()):
        raise RuntimeError("orthogonalized residual design has no active columns")
    column_scale = raw_scale.clamp_min(maximum_scale * minimum_column_fraction)
    normalized = weighted_design / column_scale[None]
    normalized[:, ~active] = 0.0
    gram = normalized.transpose(0, 1) @ normalized
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(1)
    cutoff = eigenvalues[0] * relative_singular_cutoff**2
    retained = eigenvalues >= cutoff
    if not bool(retained.any()):
        raise RuntimeError("singular cutoff removed the complete residual basis")
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    right_hand = normalized.transpose(0, 1) @ weighted_target
    projected_target = retained_vectors.transpose(0, 1) @ right_hand
    coefficients = retained_vectors @ (
        projected_target / (retained_values[:, None] + ridge)
    )
    normalized_design = design.float() / column_scale[None]
    normalized_design[:, ~active] = 0.0
    prediction = normalized_design @ coefficients
    energy = retained_values.sum() / eigenvalues.sum().clamp_min(1e-8)
    return OrthogonalizedResidualSolution(
        prediction=prediction,
        normalized_coefficients=coefficients,
        column_scale=column_scale,
        active_columns=active,
        error=feature_error(prediction, target, valid),
        retained_rank=retained.sum(),
        full_rank=int(active.sum()),
        effective_condition=(
            retained_values[0] / retained_values[-1].clamp_min(1e-12)
        ).sqrt(),
        retained_energy=energy,
        coefficient_rms=coefficients[active].square().mean().sqrt(),
    )


def fit_orthogonalized_object_residual(
    tokens,
    slots,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    budgets: tuple[int, ...],
    ridge: float,
    relative_singular_cutoff: float,
    minimum_column_fraction: float,
    minimum_utility_fraction: float,
    maximum_scene_fraction: float,
) -> OrthogonalizedObjectResidualFit:
    partition = fit_partitioned_object_field(
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
    fit = OrthogonalizedObjectResidualFit(partition, {})
    for budget in budgets:
        design, _ = orthogonalized_residual_design(
            fit, coordinates, local_budget=budget
        )
        fit.solutions[budget] = _truncated_svd_solution(
            design,
            features,
            valid,
            ridge=ridge,
            relative_singular_cutoff=relative_singular_cutoff,
            minimum_column_fraction=minimum_column_fraction,
        )
    return fit


def render_orthogonalized_object_residual(
    fit: OrthogonalizedObjectResidualFit,
    coordinates: torch.Tensor,
    *,
    local_budget: int,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_presence: torch.Tensor | None = None,
    object_feature_delta: torch.Tensor | None = None,
    transport_local_residual: bool,
) -> torch.Tensor:
    if local_budget not in fit.solutions:
        raise ValueError(f"orthogonalized field has no budget {local_budget}")
    geometry = fit.partition.reference.geometry
    object_count = geometry.object_centers.shape[0]
    presence = _validated_presence(geometry, object_presence)
    local_arguments = (
        {
            "local_object_centers": target_object_centers,
            "local_scale_ratio": object_scale_ratio,
            "local_presence": presence,
        }
        if transport_local_residual
        else {}
    )
    design, root_gates = orthogonalized_residual_design(
        fit,
        coordinates,
        local_budget=local_budget,
        root_object_centers=target_object_centers,
        root_scale_ratio=object_scale_ratio,
        root_presence=presence,
        **local_arguments,
    )
    solution = fit.solutions[local_budget]
    normalized = design.float() / solution.column_scale[None]
    normalized[:, ~solution.active_columns] = 0.0
    prediction = normalized @ solution.normalized_coefficients
    if object_feature_delta is not None:
        expected = (object_count, prediction.shape[-1])
        if object_feature_delta.shape != expected:
            raise ValueError(f"object feature delta must have shape {expected}")
        object_delta = object_feature_delta[geometry.group_ids[:-1]].float()
        prediction = prediction + root_gates[:, :-1] @ object_delta
    return prediction
