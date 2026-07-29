"""Oracle probe for anisotropic object-centered attention carriers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .object_centered_carrier_probe import (
    FrameGroups,
    build_frame_groups,
    feature_error,
)


@dataclass
class AttentionCarrierSet:
    object_centers: torch.Tensor
    object_transforms: torch.Tensor
    object_active: torch.Tensor
    carrier_group: torch.Tensor
    relative_center: torch.Tensor
    relative_covariance: torch.Tensor
    feature: torch.Tensor
    confidence: torch.Tensor
    root_mask: torch.Tensor


@dataclass
class AttentionCarrierFit:
    carriers: AttentionCarrierSet
    root_error: torch.Tensor
    budget_errors: dict[int, torch.Tensor]
    selected_local_carriers: int
    selected_scene_carriers: int
    local_carriers_per_object: torch.Tensor
    stopped_by_utility: bool


def _canonical_coordinates(
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    transforms: torch.Tensor,
) -> torch.Tensor:
    difference = coordinates[None].float() - centers[:, None].float()
    return torch.linalg.solve(
        transforms[:, None].float(), difference[..., None]
    ).squeeze(-1)


def _group_coordinates(
    groups: FrameGroups,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    objects = _canonical_coordinates(
        coordinates, groups.object_centers, groups.object_transforms
    )
    return torch.cat((objects, coordinates[None].float()), dim=0)


def _weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("...n,nc->...c", weight.float(), value.float()) / weight.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)


def _stabilize_covariance(
    covariance: torch.Tensor,
    floor: float,
    ceiling: float,
) -> torch.Tensor:
    covariance = 0.5 * (covariance.float() + covariance.float().transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp(floor**2, ceiling**2)
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)


def _scene_root(
    groups: FrameGroups,
    features: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support = groups.support[-1].float()
    mass = support.sum().clamp_min(1e-6)
    center = (support[:, None] * coordinates.float()).sum(dim=0) / mass
    difference = coordinates.float() - center
    covariance = torch.einsum("n,ni,nj->ij", support / mass, difference, difference)
    covariance = _stabilize_covariance(covariance, 0.05, 2.0)
    feature = (support[:, None] * features.float()).sum(dim=0) / mass
    return center, covariance, feature


def _root_carriers(
    groups: FrameGroups,
    features: torch.Tensor,
    coordinates: torch.Tensor,
) -> AttentionCarrierSet:
    carrier_group = []
    center = []
    covariance = []
    value = []
    confidence = []
    for group in range(groups.object_centers.shape[0]):
        if not bool(groups.object_active[group]):
            continue
        support = groups.support[group].float()
        carrier_group.append(group)
        center.append(coordinates.new_zeros(2, dtype=torch.float32))
        covariance.append(torch.eye(2, device=coordinates.device, dtype=torch.float32))
        value.append(_weighted_mean(features, support))
        confidence.append(coordinates.new_tensor(1.0, dtype=torch.float32))
    scene_center, scene_covariance, scene_feature = _scene_root(
        groups, features, coordinates
    )
    carrier_group.append(-1)
    center.append(scene_center)
    covariance.append(scene_covariance)
    value.append(scene_feature)
    confidence.append(coordinates.new_tensor(1.0, dtype=torch.float32))
    count = len(carrier_group)
    return AttentionCarrierSet(
        object_centers=groups.object_centers,
        object_transforms=groups.object_transforms,
        object_active=groups.object_active,
        carrier_group=torch.tensor(
            carrier_group, device=coordinates.device, dtype=torch.long
        ),
        relative_center=torch.stack(center),
        relative_covariance=torch.stack(covariance),
        feature=torch.stack(value),
        confidence=torch.stack(confidence),
        root_mask=torch.ones(count, device=coordinates.device, dtype=torch.bool),
    )


def _carrier_subset(
    carriers: AttentionCarrierSet,
    local_budget: int | None,
) -> torch.Tensor:
    roots = torch.nonzero(carriers.root_mask, as_tuple=False).flatten()
    local = torch.nonzero(~carriers.root_mask, as_tuple=False).flatten()
    if local_budget is not None:
        local = local[:local_budget]
    return torch.cat((roots, local))


def _group_readout(
    query: torch.Tensor,
    carrier_center: torch.Tensor,
    carrier_covariance: torch.Tensor,
    carrier_feature: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = query[:, None].float() - carrier_center[None].float()
    cholesky = torch.linalg.cholesky(carrier_covariance.float())
    whitened = torch.linalg.solve_triangular(
        cholesky,
        difference.permute(1, 2, 0),
        upper=False,
    )
    distance = whitened.square().sum(dim=1).transpose(0, 1)
    logits = (
        -0.5 * distance.clamp_max(160.0)
        + confidence.float().clamp_min(1e-6).log()[None]
    )
    attention = logits.softmax(dim=1)
    feature = attention @ carrier_feature.float()
    evidence = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
    return feature, evidence


def render_attention_carriers(
    carriers: AttentionCarrierSet,
    coordinates: torch.Tensor,
    *,
    local_budget: int | None = None,
    target_object_centers: torch.Tensor | None = None,
    object_scale_ratio: torch.Tensor | None = None,
    object_feature_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render with object-internal attention and object/scene normalization."""
    selected = _carrier_subset(carriers, local_budget)
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
    if object_feature_delta is not None and object_feature_delta.shape != (
        object_count,
        carriers.feature.shape[-1],
    ):
        raise ValueError("object feature delta must have shape [K,C]")
    transforms = (
        carriers.object_transforms.float() * ratio.clamp(0.25, 4.0)[:, None, None]
    )
    canonical = _canonical_coordinates(coordinates, centers, transforms)
    group_features = []
    group_evidence = []
    for group in range(object_count):
        indices = selected[carriers.carrier_group[selected] == group]
        if indices.numel() == 0:
            continue
        value = carriers.feature[indices].float()
        if object_feature_delta is not None:
            value = value + object_feature_delta[group].float()[None]
        feature, evidence = _group_readout(
            canonical[group],
            carriers.relative_center[indices],
            carriers.relative_covariance[indices],
            value,
            carriers.confidence[indices],
        )
        group_features.append(feature)
        group_evidence.append(evidence)
    scene = selected[carriers.carrier_group[selected] < 0]
    if scene.numel() == 0:
        raise ValueError("attention carriers require a scene root")
    feature, evidence = _group_readout(
        coordinates.float(),
        carriers.relative_center[scene],
        carriers.relative_covariance[scene],
        carriers.feature[scene],
        carriers.confidence[scene],
    )
    group_features.append(feature)
    group_evidence.append(evidence)
    stacked_feature = torch.stack(group_features, dim=1)
    stacked_evidence = torch.stack(group_evidence, dim=1)
    group_attention = stacked_evidence.softmax(dim=1)
    return torch.einsum("ng,ngc->nc", group_attention, stacked_feature)


def _point_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (prediction.float() - target.float()).square().mean(dim=-1) + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )


def _select_local_centers(
    groups: FrameGroups,
    group_coordinates: torch.Tensor,
    root_prediction: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
    maximum: int,
    minimum_utility_fraction: float,
    maximum_scene_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    complexity = _point_error(root_prediction, features) * valid.float()
    base_score = groups.support.float() * complexity[None]
    nearest = group_coordinates.square().sum(dim=-1)
    scene_root = _scene_root(groups, features, group_coordinates[-1])[0]
    nearest[-1] = (group_coordinates[-1] - scene_root).square().sum(dim=-1)
    selected_group = []
    selected_center = []
    selected_utility = []
    scene_count = 0
    initial_utility = None
    stopped = False
    for step in range(1, maximum + 1):
        separation = base_score.new_full((base_score.shape[0], 1), 0.25)
        separation[-1] = 0.08
        novelty = 1.0 - torch.exp(-0.5 * nearest / separation.square().clamp_min(1e-6))
        utility = base_score * novelty
        scene_allowed = scene_count + 1 <= maximum_scene_fraction * step + 1e-8
        if not scene_allowed:
            utility[-1].zero_()
        flat = utility.reshape(-1)
        index = int(flat.argmax())
        value = flat[index]
        if initial_utility is None:
            initial_utility = value.clamp_min(1e-8)
        if float(value / initial_utility) < minimum_utility_fraction:
            stopped = True
            break
        group = index // features.shape[0]
        patch = index % features.shape[0]
        center = group_coordinates[group, patch]
        selected_group.append(group)
        selected_center.append(center)
        selected_utility.append(value)
        distance = (group_coordinates[group] - center).square().sum(dim=-1)
        nearest[group] = torch.minimum(nearest[group], distance)
        if group == groups.object_centers.shape[0]:
            scene_count += 1
    if not selected_group:
        raise ValueError("attention carrier selector produced no local carrier")
    return (
        torch.tensor(selected_group, device=features.device, dtype=torch.long),
        torch.stack(selected_center),
        torch.stack(selected_utility),
        stopped,
    )


def _fit_local_attributes(
    groups: FrameGroups,
    group_coordinates: torch.Tensor,
    features: torch.Tensor,
    selected_group: torch.Tensor,
    seed_center: torch.Tensor,
    utility: torch.Tensor,
    point_complexity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = selected_group.shape[0]
    centers: list[torch.Tensor | None] = [None] * count
    covariances: list[torch.Tensor | None] = [None] * count
    values: list[torch.Tensor | None] = [None] * count
    confidences: list[torch.Tensor | None] = [None] * count
    utility_scale = utility / utility.mean().clamp_min(1e-8)
    point_scale = point_complexity / point_complexity.mean().clamp_min(1e-8)
    scene_group = groups.object_centers.shape[0]
    for group in torch.unique(selected_group, sorted=True).tolist():
        indices = torch.nonzero(selected_group == group, as_tuple=False).flatten()
        coordinates = group_coordinates[group]
        radius = 0.35 if group < scene_group else 0.12
        distance = (coordinates[None] - seed_center[indices, None]).square().sum(dim=-1)
        support = groups.support[group][None].float()
        weight = support * torch.exp(-0.5 * distance / radius**2)
        weight = weight * (0.25 + point_scale[None])
        mass = weight.sum(dim=-1).clamp_min(1e-6)
        refined = torch.einsum("ln,nc->lc", weight, coordinates.float())
        refined = refined / mass[:, None]
        difference = coordinates[None].float() - refined[:, None]
        covariance = torch.einsum(
            "ln,lni,lnj->lij", weight / mass[:, None], difference, difference
        )
        floor, ceiling = (0.06, 0.80) if group < scene_group else (0.03, 0.35)
        covariance = _stabilize_covariance(covariance, floor, ceiling)
        feature = torch.einsum("ln,nc->lc", weight, features.float())
        feature = feature / mass[:, None]
        confidence = mass / mass.mean().clamp_min(1e-6)
        confidence = confidence * utility_scale[indices]
        confidence = confidence.clamp(0.1, 10.0)
        for offset, destination in enumerate(indices.tolist()):
            centers[destination] = refined[offset]
            covariances[destination] = covariance[offset]
            values[destination] = feature[offset]
            confidences[destination] = confidence[offset]
    if any(value is None for value in centers + covariances + values + confidences):
        raise RuntimeError("local carrier fitting left an uninitialized attribute")
    return (
        torch.stack([value for value in centers if value is not None]),
        torch.stack([value for value in covariances if value is not None]),
        torch.stack([value for value in values if value is not None]),
        torch.stack([value for value in confidences if value is not None]),
    )


def fit_attention_carriers(
    tokens,
    slots,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    extra_budgets: tuple[int, ...] = (64, 128, 256),
    minimum_utility_fraction: float = 1e-3,
    maximum_scene_fraction: float = 0.25,
) -> AttentionCarrierFit:
    if tuple(sorted(set(extra_budgets))) != extra_budgets or min(extra_budgets) <= 0:
        raise ValueError(
            "extra carrier budgets must be unique, increasing, and positive"
        )
    if not 0.0 <= minimum_utility_fraction < 1.0:
        raise ValueError("minimum utility fraction must be in [0,1)")
    if not 0.0 <= maximum_scene_fraction <= 1.0:
        raise ValueError("maximum scene fraction must be in [0,1]")
    groups = build_frame_groups(tokens, slots, features, coordinates, valid)
    roots = _root_carriers(groups, features, coordinates)
    root_prediction = render_attention_carriers(roots, coordinates, local_budget=0)
    root_error = feature_error(root_prediction, features, valid)
    group_coordinates = _group_coordinates(groups, coordinates)
    selected_group, seed_center, utility, stopped = _select_local_centers(
        groups,
        group_coordinates,
        root_prediction,
        features,
        valid,
        max(extra_budgets),
        minimum_utility_fraction,
        maximum_scene_fraction,
    )
    local_center, local_covariance, local_feature, local_confidence = (
        _fit_local_attributes(
            groups,
            group_coordinates,
            features,
            selected_group,
            seed_center,
            utility,
            _point_error(root_prediction, features) * valid.float(),
        )
    )
    object_count = groups.object_centers.shape[0]
    stored_group = torch.where(
        selected_group == object_count,
        selected_group.new_full((), -1),
        selected_group,
    )
    local_count = selected_group.shape[0]
    carriers = AttentionCarrierSet(
        object_centers=roots.object_centers,
        object_transforms=roots.object_transforms,
        object_active=roots.object_active,
        carrier_group=torch.cat((roots.carrier_group, stored_group)),
        relative_center=torch.cat((roots.relative_center, local_center)),
        relative_covariance=torch.cat((roots.relative_covariance, local_covariance)),
        feature=torch.cat((roots.feature, local_feature)),
        confidence=torch.cat((roots.confidence, local_confidence)),
        root_mask=torch.cat(
            (
                roots.root_mask,
                torch.zeros(local_count, device=features.device, dtype=torch.bool),
            )
        ),
    )
    budget_errors = {}
    for budget in extra_budgets:
        prediction = render_attention_carriers(
            carriers, coordinates, local_budget=budget
        )
        budget_errors[budget] = feature_error(prediction, features, valid)
    object_counts = torch.stack(
        [(stored_group == group).sum() for group in range(object_count)]
    )
    return AttentionCarrierFit(
        carriers=carriers,
        root_error=root_error,
        budget_errors=budget_errors,
        selected_local_carriers=local_count,
        selected_scene_carriers=int((stored_group < 0).sum()),
        local_carriers_per_object=object_counts,
        stopped_by_utility=stopped,
    )
