"""Oracle capacity probe for sparse object-centered feature carriers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FrameGroups:
    support: torch.Tensor
    object_centers: torch.Tensor
    object_transforms: torch.Tensor
    object_active: torch.Tensor
    background: torch.Tensor


@dataclass
class ObjectCenteredCarrierSet:
    background: torch.Tensor
    object_centers: torch.Tensor
    object_transforms: torch.Tensor
    object_active: torch.Tensor
    carrier_group: torch.Tensor
    relative_center: torch.Tensor
    relative_sigma: torch.Tensor
    coefficient: torch.Tensor
    root_mask: torch.Tensor


@dataclass
class CarrierFitResult:
    carriers: ObjectCenteredCarrierSet
    root_error: torch.Tensor
    budget_errors: dict[int, torch.Tensor]
    selected_local_carriers: int
    selected_scene_carriers: int
    local_carriers_per_object: torch.Tensor
    stopped_by_marginal_gain: bool


def feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """DINO feature error used by existing readout evaluations."""
    if prediction.shape != target.shape:
        raise ValueError("feature prediction and target must align")
    if valid.shape != target.shape[:-1]:
        raise ValueError("valid mask must align with feature positions")
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    weight = valid.float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def _validate_single_frame(tokens, slots, features, coordinates, valid) -> None:
    if features.ndim != 2:
        raise ValueError("features must have shape [N,C]")
    if coordinates.shape != (features.shape[0], 2):
        raise ValueError("coordinates must have shape [N,2]")
    if valid.shape != (features.shape[0],):
        raise ValueError("valid must have shape [N]")
    if tokens.assignment.shape[0] != 1 or slots.assignment.shape[0] != 1:
        raise ValueError("carrier probe accepts one sample at a time")
    if tokens.assignment.shape[-1] != features.shape[0]:
        raise ValueError("GPSToken support and feature grid differ")
    if tokens.assignment.shape[1] != slots.assignment.shape[1]:
        raise ValueError("GPSTokens and object assignments differ")


def _grid_spacing(coordinates: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = coordinates[valid].float()
    if values.shape[0] < 2:
        raise ValueError("carrier probe requires at least two valid coordinates")
    unique_x = torch.unique(values[:, 0], sorted=True)
    unique_y = torch.unique(values[:, 1], sorted=True)
    differences = []
    if unique_x.numel() > 1:
        differences.append((unique_x[1:] - unique_x[:-1]).abs().amin())
    if unique_y.numel() > 1:
        differences.append((unique_y[1:] - unique_y[:-1]).abs().amin())
    if not differences:
        return values.new_tensor(0.05)
    return torch.stack(differences).amin().clamp_min(1e-3)


def _stabilized_transform(
    coordinates: torch.Tensor,
    center: torch.Tensor,
    weight: torch.Tensor,
    floor: torch.Tensor,
) -> torch.Tensor:
    normalized = weight / weight.sum().clamp_min(1e-6)
    difference = coordinates.float() - center.float()
    covariance = torch.einsum("n,ni,nj->ij", normalized.float(), difference, difference)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(floor.square())
    covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(-1, -2)
    return torch.linalg.cholesky(covariance)


def build_frame_groups(
    tokens,
    slots,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    minimum_object_mass: float = 2.0,
) -> FrameGroups:
    """Build soft object/scene groups; dense support is discarded after fitting."""
    _validate_single_frame(tokens, slots, features, coordinates, valid)
    active_token = tokens.assignment[0].float() * tokens.activation[0].float()
    token_partition = active_token / active_token.sum(dim=0, keepdim=True).clamp_min(
        1e-6
    )
    object_support = torch.einsum(
        "mn,mk->kn", token_partition, slots.assignment[0].float()
    )
    object_support = object_support * valid[None].float()
    object_support = object_support / object_support.sum(dim=0, keepdim=True).clamp_min(
        1.0
    )
    scene_support = (1.0 - object_support.sum(dim=0)).clamp(0.0, 1.0)
    scene_support = scene_support * valid.float()
    support = torch.cat((object_support, scene_support[None]), dim=0)

    mass = object_support.sum(dim=-1)
    object_active = (mass >= minimum_object_mass) & (slots.activity[0].float() > 1e-3)
    object_support = object_support * object_active[:, None]
    scene_support = (1.0 - object_support.sum(dim=0)).clamp(0.0, 1.0)
    support = torch.cat((object_support, scene_support[None]), dim=0)

    centers = slots.center[0].float()
    spacing = _grid_spacing(coordinates, valid)
    transforms = []
    for index in range(centers.shape[0]):
        weight = object_support[index]
        if object_active[index]:
            transform = _stabilized_transform(
                coordinates, centers[index], weight, 0.75 * spacing
            )
        else:
            transform = (
                torch.eye(2, device=coordinates.device, dtype=torch.float32) * spacing
            )
        transforms.append(transform)
    object_transforms = torch.stack(transforms)

    scene_weight = scene_support
    if float(scene_weight.sum()) < 1.0:
        scene_weight = valid.float()
    background = (features.float() * scene_weight[:, None]).sum(
        dim=0
    ) / scene_weight.sum().clamp_min(1.0)
    return FrameGroups(
        support=support,
        object_centers=centers,
        object_transforms=object_transforms,
        object_active=object_active,
        background=background,
    )


def _canonical_coordinates(
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    transforms: torch.Tensor,
) -> torch.Tensor:
    difference = coordinates[None].float() - centers[:, None].float()
    return torch.linalg.solve(
        transforms[:, None].float(), difference[..., None]
    ).squeeze(-1)


def _basis(
    coordinates: torch.Tensor,
    groups: FrameGroups,
    group: int,
    relative_center: torch.Tensor,
    relative_sigma: torch.Tensor,
) -> torch.Tensor:
    if group < 0:
        difference = coordinates.float() - relative_center.float()
    else:
        canonical = _canonical_coordinates(
            coordinates,
            groups.object_centers[group : group + 1],
            groups.object_transforms[group : group + 1],
        )[0]
        difference = canonical - relative_center.float()
    distance = difference.square().sum(dim=-1)
    return torch.exp(-0.5 * distance / relative_sigma.float().square())


def _fit_coefficient(
    residual: torch.Tensor,
    basis: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = valid.float()
    denominator = (basis.square() * weight).sum().clamp_min(1e-6)
    coefficient = (
        torch.einsum("n,nc->c", basis * weight, residual.float()) / denominator
    )
    updated = residual.float() - basis[:, None] * coefficient[None]
    return coefficient, updated


def _carrier_set(
    groups: FrameGroups,
    carrier_groups: list[int],
    centers: list[torch.Tensor],
    sigmas: list[torch.Tensor],
    coefficients: list[torch.Tensor],
    root_count: int,
) -> ObjectCenteredCarrierSet:
    if not coefficients:
        raise ValueError("carrier fit produced no object roots")
    device = groups.background.device
    return ObjectCenteredCarrierSet(
        background=groups.background,
        object_centers=groups.object_centers,
        object_transforms=groups.object_transforms,
        object_active=groups.object_active,
        carrier_group=torch.tensor(carrier_groups, device=device, dtype=torch.long),
        relative_center=torch.stack(centers),
        relative_sigma=torch.stack(sigmas),
        coefficient=torch.stack(coefficients),
        root_mask=torch.arange(len(coefficients), device=device) < root_count,
    )


def render_carriers(
    carriers: ObjectCenteredCarrierSet,
    coordinates: torch.Tensor,
    *,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_feature_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query a carrier set without using a dense object assignment map."""
    object_count = carriers.object_centers.shape[0]
    centers = (
        carriers.object_centers
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
    ratio = ratio.clamp(0.25, 4.0)
    transforms = carriers.object_transforms.float() * ratio[:, None, None]
    canonical = _canonical_coordinates(coordinates, centers, transforms)
    result = carriers.background[None].expand(coordinates.shape[0], -1).clone()
    for group in range(object_count):
        selected = carriers.carrier_group == group
        if not bool(selected.any()):
            continue
        difference = (
            canonical[group, :, None] - carriers.relative_center[selected][None].float()
        )
        sigma = carriers.relative_sigma[selected].float()
        basis = torch.exp(-0.5 * difference.square().sum(dim=-1) / sigma[None].square())
        result = result + basis @ carriers.coefficient[selected].float()
    scene = carriers.carrier_group < 0
    if bool(scene.any()):
        difference = (
            coordinates[:, None].float() - carriers.relative_center[scene][None].float()
        )
        sigma = carriers.relative_sigma[scene].float()
        basis = torch.exp(-0.5 * difference.square().sum(dim=-1) / sigma[None].square())
        result = result + basis @ carriers.coefficient[scene].float()
    if object_feature_delta is not None:
        expected = (object_count, result.shape[-1])
        if object_feature_delta.shape != expected:
            raise ValueError(f"object feature delta must have shape {expected}")
        gate = torch.exp(-0.5 * canonical.square().sum(dim=-1)).transpose(0, 1)
        gate = gate * carriers.object_active[None].float()
        gate = gate / (1.0 + gate.sum(dim=-1, keepdim=True))
        result = result + gate @ object_feature_delta.float()
    return result


def fit_object_centered_carriers(
    tokens,
    slots,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    extra_budgets: tuple[int, ...] = (8, 16, 32, 64),
    allow_scene_carriers: bool,
    minimum_marginal_gain: float = 1e-5,
    object_local_sigma: float = 0.35,
    scene_local_sigma: float = 0.15,
) -> CarrierFitResult:
    """Greedily allocate local residual carriers under an explicit budget."""
    if not extra_budgets or min(extra_budgets) <= 0:
        raise ValueError("extra carrier budgets must be positive")
    if tuple(sorted(set(extra_budgets))) != extra_budgets:
        raise ValueError("extra carrier budgets must be unique and increasing")
    if minimum_marginal_gain < 0.0:
        raise ValueError("minimum marginal gain must be non-negative")
    groups = build_frame_groups(tokens, slots, features, coordinates, valid)
    prediction = groups.background[None].expand_as(features).clone()
    residual = features.float() - prediction
    carrier_groups: list[int] = []
    relative_centers: list[torch.Tensor] = []
    relative_sigmas: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []

    canonical = _canonical_coordinates(
        coordinates, groups.object_centers, groups.object_transforms
    )
    for group in range(groups.object_centers.shape[0]):
        if not bool(groups.object_active[group]):
            continue
        center = coordinates.new_zeros(2, dtype=torch.float32)
        sigma = coordinates.new_tensor(1.0, dtype=torch.float32)
        basis = _basis(coordinates, groups, group, center, sigma)
        coefficient, residual = _fit_coefficient(residual, basis, valid)
        prediction = features.float() - residual
        carrier_groups.append(group)
        relative_centers.append(center)
        relative_sigmas.append(sigma)
        coefficients.append(coefficient)
    root_count = len(coefficients)
    if root_count == 0:
        raise ValueError("carrier probe found no active object root")
    root_error = feature_error(prediction, features, valid)
    initial_error = root_error.clamp_min(1e-8)

    budget_errors: dict[int, torch.Tensor] = {}
    object_counts = torch.zeros(
        groups.object_centers.shape[0], device=features.device, dtype=torch.long
    )
    scene_count = 0
    stopped = False
    maximum = max(extra_budgets)
    for local_index in range(1, maximum + 1):
        residual_energy = residual.square().mean(dim=-1)
        eligible_support = groups.support.clone()
        if not allow_scene_carriers:
            eligible_support[-1].zero_()
        score = eligible_support * residual_energy[None] * valid[None].float()
        for group in range(groups.object_centers.shape[0]):
            selected = [
                relative_centers[index]
                for index, value in enumerate(carrier_groups)
                if value == group
            ]
            if selected:
                distance = (
                    (canonical[group, :, None] - torch.stack(selected)[None])
                    .square()
                    .sum(dim=-1)
                    .amin(dim=-1)
                )
                score[group] = score[group] * (
                    1.0 - torch.exp(-0.5 * distance / 0.20**2)
                )
        scene_selected = [
            relative_centers[index]
            for index, value in enumerate(carrier_groups)
            if value < 0
        ]
        if scene_selected:
            distance = (
                (coordinates[:, None].float() - torch.stack(scene_selected)[None])
                .square()
                .sum(dim=-1)
                .amin(dim=-1)
            )
            score[-1] = score[-1] * (
                1.0 - torch.exp(-0.5 * distance / scene_local_sigma**2)
            )
        flat_index = int(score.reshape(-1).argmax())
        if float(score.reshape(-1)[flat_index]) <= 0.0:
            stopped = True
            break
        group = flat_index // features.shape[0]
        patch = flat_index % features.shape[0]
        if group == groups.object_centers.shape[0]:
            stored_group = -1
            center = coordinates[patch].float()
            sigma = coordinates.new_tensor(scene_local_sigma, dtype=torch.float32)
        else:
            stored_group = group
            center = canonical[group, patch].float()
            sigma = coordinates.new_tensor(object_local_sigma, dtype=torch.float32)
        basis = _basis(coordinates, groups, stored_group, center, sigma)
        coefficient, candidate_residual = _fit_coefficient(residual, basis, valid)
        candidate_prediction = features.float() - candidate_residual
        current_error = feature_error(prediction, features, valid)
        candidate_error = feature_error(candidate_prediction, features, valid)
        relative_gain = (current_error - candidate_error) / initial_error
        if float(relative_gain) < minimum_marginal_gain:
            stopped = True
            break
        residual = candidate_residual
        prediction = candidate_prediction
        carrier_groups.append(stored_group)
        relative_centers.append(center)
        relative_sigmas.append(sigma)
        coefficients.append(coefficient)
        if stored_group < 0:
            scene_count += 1
        else:
            object_counts[stored_group] += 1
        if local_index in extra_budgets:
            budget_errors[local_index] = feature_error(prediction, features, valid)

    final_error = feature_error(prediction, features, valid)
    for budget in extra_budgets:
        budget_errors.setdefault(budget, final_error)
    carriers = _carrier_set(
        groups,
        carrier_groups,
        relative_centers,
        relative_sigmas,
        coefficients,
        root_count,
    )
    return CarrierFitResult(
        carriers=carriers,
        root_error=root_error,
        budget_errors=budget_errors,
        selected_local_carriers=len(coefficients) - root_count,
        selected_scene_carriers=scene_count,
        local_carriers_per_object=object_counts,
        stopped_by_marginal_gain=stopped,
    )
