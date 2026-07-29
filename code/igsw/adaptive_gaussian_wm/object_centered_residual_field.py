"""Oracle capacity probe for signed object-centered residual fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .object_centered_carrier_probe import (
    FrameGroups,
    build_frame_groups,
)
from .residual_field_linear import (
    RidgeSolution,
    point_error,
    rbf,
    ridge_solution,
    stabilize_covariance,
)


@dataclass
class ResidualFieldGeometry:
    object_centers: torch.Tensor
    object_transforms: torch.Tensor
    object_active: torch.Tensor
    group_ids: torch.Tensor
    group_priors: torch.Tensor
    coordinate_minimum: torch.Tensor
    coordinate_maximum: torch.Tensor
    carrier_group: torch.Tensor
    relative_center: torch.Tensor
    relative_covariance: torch.Tensor


@dataclass
class ResidualFieldFit:
    geometry: ResidualFieldGeometry
    geometric_solution: RidgeSolution
    global_root_error: torch.Tensor
    oracle_root_error: torch.Tensor
    geometric_root_error: torch.Tensor
    global_errors: dict[int, torch.Tensor]
    oracle_errors: dict[int, torch.Tensor]
    geometric_errors: dict[int, torch.Tensor]
    condition_numbers: dict[str, torch.Tensor]
    coefficient_rms: dict[str, torch.Tensor]
    selected_global_carriers: int
    selected_object_carriers: int
    selected_scene_carriers: int
    local_carriers_per_object: torch.Tensor
    global_stopped_by_utility: bool
    object_stopped_by_utility: bool


def _canonical_coordinates(
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    transforms: torch.Tensor,
) -> torch.Tensor:
    difference = coordinates[None].float() - centers[:, None].float()
    return torch.linalg.solve(
        transforms[:, None].float(), difference[..., None]
    ).squeeze(-1)


def _group_ids(groups: FrameGroups) -> torch.Tensor:
    active = torch.nonzero(groups.object_active, as_tuple=False).flatten()
    scene = active.new_tensor([groups.object_centers.shape[0]])
    return torch.cat((active, scene))


def _oracle_gates(groups: FrameGroups, group_ids: torch.Tensor) -> torch.Tensor:
    gates = groups.support[group_ids].float().transpose(0, 1)
    return gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _group_priors(groups: FrameGroups, group_ids: torch.Tensor) -> torch.Tensor:
    mass = groups.support[group_ids].float().sum(dim=-1)
    return mass / mass.sum().clamp_min(1e-6)


def geometric_object_gates(
    geometry: ResidualFieldGeometry,
    coordinates: torch.Tensor,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
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
    object_ids = geometry.group_ids[:-1]
    transforms = (
        geometry.object_transforms.float() * ratio.clamp(0.25, 4.0)[:, None, None]
    )
    canonical = _canonical_coordinates(
        coordinates, centers[object_ids], transforms[object_ids]
    )
    determinant = torch.linalg.det(transforms[object_ids]).abs().clamp_min(1e-8)
    object_logits = (
        geometry.group_priors[:-1].clamp_min(1e-8).log()[:, None]
        - math.log(2.0 * math.pi)
        - determinant.log()[:, None]
        - 0.5 * canonical.square().sum(dim=-1)
    )
    extent = (geometry.coordinate_maximum - geometry.coordinate_minimum).clamp_min(1e-3)
    scene_logit = geometry.group_priors[-1].clamp_min(1e-8).log() - extent.prod().log()
    scene_logits = scene_logit.expand(1, coordinates.shape[0])
    return torch.cat((object_logits, scene_logits), dim=0).transpose(0, 1).softmax(-1)


def _global_design(
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    covariance: torch.Tensor,
    local_budget: int,
) -> torch.Tensor:
    count = min(local_budget, centers.shape[0])
    root = coordinates.new_ones((coordinates.shape[0], 1), dtype=torch.float32)
    return torch.cat(
        (root, rbf(coordinates, centers[:count], covariance[:count])), dim=1
    )


def object_residual_design(
    geometry: ResidualFieldGeometry,
    coordinates: torch.Tensor,
    gates: torch.Tensor,
    local_budget: int,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    gate_local_residuals: bool = True,
) -> torch.Tensor:
    count = min(local_budget, geometry.carrier_group.shape[0])
    object_count = geometry.object_centers.shape[0]
    centers = (
        geometry.object_centers
        if target_object_centers is None
        else target_object_centers.float()
    )
    ratio = (
        coordinates.new_ones(object_count, dtype=torch.float32)
        if object_scale_ratio is None
        else object_scale_ratio.float()
    )
    transforms = (
        geometry.object_transforms.float() * ratio.clamp(0.25, 4.0)[:, None, None]
    )
    canonical = _canonical_coordinates(coordinates, centers, transforms)
    if count == 0:
        return gates.float()
    local_columns = coordinates.new_zeros(
        (coordinates.shape[0], count), dtype=torch.float32
    )
    selected_groups = geometry.carrier_group[:count]
    for root_index, group_tensor in enumerate(geometry.group_ids):
        group = int(group_tensor)
        indices = torch.nonzero(selected_groups == group, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        query = coordinates if group == object_count else canonical[group]
        basis = rbf(
            query,
            geometry.relative_center[indices],
            geometry.relative_covariance[indices],
        )
        envelope = (
            gates[:, root_index : root_index + 1] if gate_local_residuals else 1.0
        )
        local_columns[:, indices] = envelope * basis
    return torch.cat((gates.float(), local_columns), dim=1)


def _select_global_centers(
    coordinates: torch.Tensor,
    complexity: torch.Tensor,
    valid: torch.Tensor,
    maximum: int,
    minimum_utility_fraction: float,
) -> tuple[torch.Tensor, bool]:
    nearest = coordinates.new_full((coordinates.shape[0],), float("inf"))
    selected = []
    initial = None
    stopped = False
    for _ in range(maximum):
        novelty = torch.ones_like(nearest)
        finite = torch.isfinite(nearest)
        novelty[finite] = 1.0 - torch.exp(-0.5 * nearest[finite] / 0.08**2)
        utility = complexity * valid.float() * novelty
        patch = int(utility.argmax())
        value = utility[patch]
        if initial is None:
            initial = value.clamp_min(1e-8)
        if float(value / initial) < minimum_utility_fraction:
            stopped = True
            break
        center = coordinates[patch].float()
        selected.append(center)
        distance = (coordinates.float() - center).square().sum(dim=-1)
        nearest = torch.minimum(nearest, distance)
    if not selected:
        raise RuntimeError("global residual selector produced no carrier")
    return torch.stack(selected), stopped


def _group_coordinates(groups: FrameGroups, coordinates: torch.Tensor) -> torch.Tensor:
    canonical = _canonical_coordinates(
        coordinates, groups.object_centers, groups.object_transforms
    )
    return torch.cat((canonical, coordinates[None].float()), dim=0)


def _select_object_centers(
    groups: FrameGroups,
    group_coordinates: torch.Tensor,
    complexity: torch.Tensor,
    maximum: int,
    minimum_utility_fraction: float,
    maximum_scene_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    score = groups.support.float() * complexity[None]
    nearest = group_coordinates.square().sum(dim=-1)
    selected_group = []
    selected_center = []
    scene_count = 0
    initial = None
    stopped = False
    for step in range(1, maximum + 1):
        separation = score.new_full((score.shape[0], 1), 0.25)
        separation[-1] = 0.08
        novelty = 1.0 - torch.exp(-0.5 * nearest / separation.square())
        utility = score * novelty
        if scene_count + 1 > maximum_scene_fraction * step + 1e-8:
            utility[-1].zero_()
        flat = utility.flatten()
        index = int(flat.argmax())
        value = flat[index]
        if initial is None:
            initial = value.clamp_min(1e-8)
        if float(value / initial) < minimum_utility_fraction:
            stopped = True
            break
        group = index // complexity.shape[0]
        patch = index % complexity.shape[0]
        center = group_coordinates[group, patch]
        selected_group.append(group)
        selected_center.append(center)
        distance = (group_coordinates[group] - center).square().sum(dim=-1)
        nearest[group] = torch.minimum(nearest[group], distance)
        if group == groups.object_centers.shape[0]:
            scene_count += 1
    if not selected_group:
        raise RuntimeError("object residual selector produced no carrier")
    return (
        torch.tensor(selected_group, device=complexity.device, dtype=torch.long),
        torch.stack(selected_center),
        stopped,
    )


def _fit_covariances(
    group_coordinates: torch.Tensor,
    support: torch.Tensor,
    carrier_group: torch.Tensor,
    seed_center: torch.Tensor,
    complexity: torch.Tensor,
    scene_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = carrier_group.shape[0]
    centers = seed_center.new_zeros((count, 2), dtype=torch.float32)
    covariance = seed_center.new_zeros((count, 2, 2), dtype=torch.float32)
    point_scale = complexity / complexity.mean().clamp_min(1e-8)
    for group_tensor in torch.unique(carrier_group, sorted=True):
        group = int(group_tensor)
        indices = torch.nonzero(carrier_group == group, as_tuple=False).flatten()
        coordinates = group_coordinates[group].float()
        radius = 0.35 if group < scene_group else 0.12
        distance = (
            (coordinates[None] - seed_center[indices, None].float())
            .square()
            .sum(dim=-1)
        )
        weight = support[group][None].float() * torch.exp(-0.5 * distance / radius**2)
        weight = weight * (0.25 + point_scale[None])
        mass = weight.sum(dim=-1).clamp_min(1e-6)
        center = torch.einsum("ln,nc->lc", weight, coordinates) / mass[:, None]
        difference = coordinates[None] - center[:, None]
        matrix = torch.einsum(
            "ln,lni,lnj->lij", weight / mass[:, None], difference, difference
        )
        floor, ceiling = (0.06, 0.80) if group < scene_group else (0.03, 0.35)
        centers[indices] = center
        covariance[indices] = stabilize_covariance(matrix, floor, ceiling)
    return centers, covariance


def fit_residual_fields(
    tokens,
    slots,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    budgets: tuple[int, ...] = (64, 128, 256),
    ridge: float = 1e-4,
    minimum_utility_fraction: float = 1e-3,
    maximum_scene_fraction: float = 0.25,
) -> ResidualFieldFit:
    if tuple(sorted(set(budgets))) != budgets or min(budgets) <= 0:
        raise ValueError("budgets must be unique, increasing, and positive")
    if ridge <= 0.0:
        raise ValueError("ridge regularization must be positive")
    if not 0.0 <= minimum_utility_fraction < 1.0:
        raise ValueError("minimum utility fraction must be in [0,1)")
    if not 0.0 <= maximum_scene_fraction <= 1.0:
        raise ValueError("maximum scene fraction must be in [0,1]")
    groups = build_frame_groups(tokens, slots, features, coordinates, valid)
    group_ids = _group_ids(groups)
    oracle_gates = _oracle_gates(groups, group_ids)
    empty = coordinates.new_zeros((0, 2), dtype=torch.float32)
    empty_covariance = coordinates.new_zeros((0, 2, 2), dtype=torch.float32)
    maximum = max(budgets)
    global_root = ridge_solution(
        _global_design(coordinates, empty, empty_covariance, 0),
        features,
        valid,
        ridge,
    )
    global_centers, global_stopped = _select_global_centers(
        coordinates,
        point_error(global_root.prediction, features),
        valid,
        maximum,
        minimum_utility_fraction,
    )
    global_groups = torch.zeros(
        global_centers.shape[0], device=coordinates.device, dtype=torch.long
    )
    global_center, global_covariance = _fit_covariances(
        coordinates[None].float(),
        valid[None].float(),
        global_groups,
        global_centers,
        point_error(global_root.prediction, features) * valid.float(),
        0,
    )
    coordinate_minimum = coordinates[valid].float().amin(dim=0)
    coordinate_maximum = coordinates[valid].float().amax(dim=0)
    base_geometry = ResidualFieldGeometry(
        object_centers=groups.object_centers,
        object_transforms=groups.object_transforms,
        object_active=groups.object_active,
        group_ids=group_ids,
        group_priors=_group_priors(groups, group_ids),
        coordinate_minimum=coordinate_minimum,
        coordinate_maximum=coordinate_maximum,
        carrier_group=group_ids.new_zeros(0),
        relative_center=empty,
        relative_covariance=empty_covariance,
    )
    geometric_gates = geometric_object_gates(base_geometry, coordinates)
    oracle_root = ridge_solution(oracle_gates, features, valid, ridge)
    geometric_root = ridge_solution(geometric_gates, features, valid, ridge)
    group_coordinates = _group_coordinates(groups, coordinates)
    carrier_group, seed_center, object_stopped = _select_object_centers(
        groups,
        group_coordinates,
        point_error(oracle_root.prediction, features) * valid.float(),
        maximum,
        minimum_utility_fraction,
        maximum_scene_fraction,
    )
    relative_center, relative_covariance = _fit_covariances(
        group_coordinates,
        groups.support,
        carrier_group,
        seed_center,
        point_error(oracle_root.prediction, features) * valid.float(),
        groups.object_centers.shape[0],
    )
    geometry = ResidualFieldGeometry(
        object_centers=base_geometry.object_centers,
        object_transforms=base_geometry.object_transforms,
        object_active=base_geometry.object_active,
        group_ids=base_geometry.group_ids,
        group_priors=base_geometry.group_priors,
        coordinate_minimum=base_geometry.coordinate_minimum,
        coordinate_maximum=base_geometry.coordinate_maximum,
        carrier_group=carrier_group,
        relative_center=relative_center,
        relative_covariance=relative_covariance,
    )
    global_errors = {}
    oracle_errors = {}
    geometric_errors = {}
    conditions = {}
    coefficient_rms = {}
    geometric_solution = geometric_root
    for budget in budgets:
        global_solution = ridge_solution(
            _global_design(coordinates, global_center, global_covariance, budget),
            features,
            valid,
            ridge,
        )
        oracle_solution = ridge_solution(
            object_residual_design(geometry, coordinates, oracle_gates, budget),
            features,
            valid,
            ridge,
        )
        geometric_solution = ridge_solution(
            object_residual_design(geometry, coordinates, geometric_gates, budget),
            features,
            valid,
            ridge,
        )
        for name, solution in (
            ("global", global_solution),
            ("oracle", oracle_solution),
            ("geometric", geometric_solution),
        ):
            conditions[f"{name}_b{budget}"] = solution.condition_number
            coefficient_rms[f"{name}_b{budget}"] = solution.coefficient_rms
        global_errors[budget] = global_solution.error
        oracle_errors[budget] = oracle_solution.error
        geometric_errors[budget] = geometric_solution.error
    object_count = groups.object_centers.shape[0]
    object_counts = torch.stack(
        [(carrier_group == group).sum() for group in range(object_count)]
    )
    return ResidualFieldFit(
        geometry=geometry,
        geometric_solution=geometric_solution,
        global_root_error=global_root.error,
        oracle_root_error=oracle_root.error,
        geometric_root_error=geometric_root.error,
        global_errors=global_errors,
        oracle_errors=oracle_errors,
        geometric_errors=geometric_errors,
        condition_numbers=conditions,
        coefficient_rms=coefficient_rms,
        selected_global_carriers=global_center.shape[0],
        selected_object_carriers=carrier_group.shape[0],
        selected_scene_carriers=int((carrier_group == object_count).sum()),
        local_carriers_per_object=object_counts,
        global_stopped_by_utility=global_stopped,
        object_stopped_by_utility=object_stopped,
    )


def render_geometric_residual_field(
    fit: ResidualFieldFit,
    coordinates: torch.Tensor,
    *,
    local_budget: int,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_feature_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    gates = geometric_object_gates(
        fit.geometry,
        coordinates,
        target_object_centers,
        object_scale_ratio,
    )
    design = object_residual_design(
        fit.geometry,
        coordinates,
        gates,
        local_budget,
        target_object_centers,
        object_scale_ratio,
    )
    coefficients = fit.geometric_solution.coefficients[: design.shape[1]].clone()
    if object_feature_delta is not None:
        expected = (fit.geometry.object_centers.shape[0], coefficients.shape[-1])
        if object_feature_delta.shape != expected:
            raise ValueError(f"object feature delta must have shape {expected}")
        for root_index, group_tensor in enumerate(fit.geometry.group_ids[:-1]):
            coefficients[root_index] += object_feature_delta[int(group_tensor)].float()
    return design @ coefficients
