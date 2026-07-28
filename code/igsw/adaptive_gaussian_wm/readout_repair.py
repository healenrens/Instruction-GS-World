"""Causal current-frame anchors and health metrics for Gaussian readout."""
from __future__ import annotations

import math

import torch

from .change_objectives import dense_feature_loss
from .decoder import GaussianReadoutState, feature_loss_coverage
from .gaussian_math import mahalanobis_squared_from_precision, precision_2d


def first_query(state: GaussianReadoutState) -> GaussianReadoutState:
    """Select one identical current-state query without changing rank."""
    return GaussianReadoutState(
        feature=state.feature[:, :1],
        center=state.center[:, :1],
        covariance=state.covariance[:, :1],
        depth_order=state.depth_order[:, :1],
        opacity=state.opacity[:, :1],
        activation=state.activation[:, :1],
        rgb=None if state.rgb is None else state.rgb[:, :1],
        background_feature=(
            None
            if state.background_feature is None
            else state.background_feature[:, :1]
        ),
    )


def direct_current_state(tokens) -> GaussianReadoutState:
    """Build the allocator-only Gaussian state used to locate interface error."""
    return GaussianReadoutState(
        feature=tokens.decoded_features[:, None],
        center=tokens.center[:, None],
        covariance=tokens.covariance[:, None],
        depth_order=tokens.depth_order[:, None],
        opacity=tokens.opacity[:, None],
        activation=tokens.activation[:, None],
        background_feature=None,
    )


def current_readout_regularization(
    state: GaussianReadoutState,
    tokens,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep the learned correction local while permitting held-set gains."""
    current = first_query(state)
    micro_count = tokens.latent.shape[1]
    if current.feature.shape[2] % micro_count:
        raise ValueError("readout component count must divide by GPSToken count")
    children = current.feature.shape[2] // micro_count

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.repeat_interleave(children, dim=1).float()

    base_covariance = expand(tokens.covariance)
    readout_covariance = current.covariance[:, 0].float()
    base_eigenvalues = torch.linalg.eigvalsh(base_covariance).clamp_min(1e-6)
    readout_eigenvalues = torch.linalg.eigvalsh(
        readout_covariance
    ).clamp_min(1e-6)
    feature = (
        current.feature[:, 0].float() - expand(tokens.decoded_features)
    ).square().mean()
    center = (
        current.center[:, 0].float() - expand(tokens.center)
    ).square().mean() / 0.25**2
    covariance = (
        readout_eigenvalues.log() - base_eigenvalues.log()
    ).square().mean()
    depth = (
        current.depth_order[:, 0].float() - expand(tokens.depth_order)
    ).square().mean()
    opacity = (
        torch.logit(current.opacity[:, 0].float(), eps=1e-4)
        - torch.logit(expand(tokens.opacity), eps=1e-4)
    ).square().mean()
    activation = (
        torch.logit(current.activation[:, 0].float(), eps=1e-4)
        - torch.logit(expand(tokens.activation), eps=1e-4)
    ).square().mean()
    total = (
        0.1 * feature
        + center
        + 0.25 * covariance
        + 0.01 * depth
        + 0.01 * opacity
        + 0.01 * activation
    )
    return total, {
        "readout_residual_feature_energy": feature,
        "readout_residual_center_energy": center,
        "readout_residual_covariance_energy": covariance,
        "readout_residual_depth_energy": depth,
        "readout_residual_opacity_energy": opacity,
        "readout_residual_activation_energy": activation,
    }


def current_readout_objective(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct frozen current DINO targets without reading future values."""
    state = first_query(output["current_gaussian_readout"])
    coordinates = batch["history_coordinates"][:, -1:]
    target = batch["history_features"][:, -1:]
    valid = batch["history_valid"][:, -1:]
    prediction, coverage = model.gaussian_readout.splat_features(
        state,
        coordinates,
    )
    loss_coverage = feature_loss_coverage(state, coverage)
    reconstruction = dense_feature_loss(prediction, target, valid, loss_coverage)
    valid_weight = valid.float()
    coverage_penalty = reconstruction * 0.0
    if state.background_feature is None:
        coverage_penalty = (
            (torch.relu(0.05 - coverage.float()) / 0.05).square()
            * valid_weight
        ).sum() / valid_weight.sum().clamp_min(1.0)
    anchor = reconstruction + 0.1 * coverage_penalty
    regularization, parts = current_readout_regularization(
        state,
        output["history_token_states"][-1],
    )
    parts.update(
        readout_current_reconstruction=reconstruction,
        readout_current_coverage_penalty=coverage_penalty,
        readout_current_coverage=(
            ((coverage > 1e-4) & valid).float().sum()
            / valid.float().sum().clamp_min(1.0)
        ),
    )
    return anchor, regularization, parts


@torch.no_grad()
def gaussian_state_health(
    state: GaussianReadoutState,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Measure kernel collapse and whether multiple Gaussians contribute."""
    current = first_query(state)
    covariance = current.covariance.float()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-8)
    condition = eigenvalues[..., 1] / eigenvalues[..., 0]
    difference = coordinates[:, :, None].float() - current.center[..., None, :].float()
    distance = mahalanobis_squared_from_precision(
        precision_2d(covariance),
        difference,
    )
    weight = torch.exp(-0.5 * distance)
    weight = weight * current.opacity.squeeze(-1)[..., None].float()
    weight = weight * current.activation.squeeze(-1)[..., None].float()
    order = torch.softmax(
        current.depth_order.squeeze(-1).float(), dim=2
    )
    weight = weight * order[..., None] * current.center.shape[2]
    coverage = weight.sum(dim=2)
    mixture = weight / coverage[:, :, None].clamp_min(1e-6)
    effective = mixture.square().sum(dim=2).clamp_min(1e-8).reciprocal()
    entropy = -(mixture.clamp_min(1e-8) * mixture.clamp_min(1e-8).log()).sum(dim=2)
    valid_weight = valid.float()
    denominator = valid_weight.sum().clamp_min(1.0)
    opacity = current.opacity.float()
    activation = current.activation.float()
    opacity_saturated = ((opacity < 0.01) | (opacity > 0.99)).float()
    return {
        f"{prefix}_covariance_min_eigenvalue": eigenvalues[..., 0].mean(),
        f"{prefix}_covariance_max_eigenvalue": eigenvalues[..., 1].mean(),
        f"{prefix}_covariance_condition": condition.mean(),
        f"{prefix}_opacity_mean": opacity.mean(),
        f"{prefix}_activation_mean": activation.mean(),
        f"{prefix}_opacity_saturation_fraction": opacity_saturated.mean(),
        f"{prefix}_active_token_fraction": (activation > 0.5).float().mean(),
        f"{prefix}_effective_components": (
            effective * valid_weight
        ).sum() / denominator,
        f"{prefix}_mixture_entropy_fraction": (
            entropy * valid_weight
        ).sum() / denominator / math.log(max(current.center.shape[2], 2)),
        f"{prefix}_max_component_weight": (
            mixture.max(dim=2).values * valid_weight
        ).sum() / denominator,
        f"{prefix}_coverage_mean": (
            coverage * valid_weight
        ).sum() / denominator,
    }
