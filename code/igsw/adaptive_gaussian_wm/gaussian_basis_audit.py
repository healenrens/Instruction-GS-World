"""Deterministic held-data capacity audit for hierarchical Gaussian bases."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _seeds(count: int, device: torch.device) -> torch.Tensor:
    if count == 1:
        return torch.zeros(1, 2, device=device)
    angle = 2.0 * math.pi * torch.arange(count, device=device) / count
    return 0.7 * torch.stack((angle.cos(), angle.sin()), dim=-1)


def _moments(
    distribution: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.einsum("bmn,bnd->bmd", distribution, coordinates)
    difference = coordinates[:, None] - mean[:, :, None]
    covariance = torch.einsum(
        "bmn,bmni,bmnj->bmij",
        distribution,
        difference,
        difference,
    )
    identity = torch.eye(2, device=coordinates.device)
    return mean, covariance + 1e-4 * identity


def fit_gaussian_basis(
    active_assignment: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    children: int,
    iterations: int = 8,
) -> torch.Tensor:
    """Fit a per-parent Gaussian mixture and return support [B,M,N]."""
    target = active_assignment.float() * valid[:, None].float()
    parent_mass = target.sum(dim=-1, keepdim=True)
    distribution = target / parent_mass.clamp_min(1e-6)
    parent_mean, parent_covariance = _moments(distribution, coordinates.float())
    parent_cholesky = torch.linalg.cholesky(parent_covariance)
    seed = _seeds(children, coordinates.device)
    mean = parent_mean[:, :, None] + 0.5 * torch.einsum(
        "bmij,lj->bmli", parent_cholesky, seed
    )
    covariance = parent_covariance[:, :, None] / math.sqrt(children)
    covariance = covariance.expand(-1, -1, children, -1, -1).clone()
    mixture = target.new_full(
        (*target.shape[:2], children), 1.0 / children
    )
    identity = torch.eye(2, device=target.device)
    for _ in range(iterations):
        precision = torch.linalg.inv(covariance)
        difference = coordinates[:, None, None].float() - mean[:, :, :, None]
        distance = torch.einsum(
            "bmlni,bmlij,bmlnj->bmln",
            difference,
            precision,
            difference,
        )
        log_determinant = torch.logdet(covariance).unsqueeze(-1)
        log_joint = (
            mixture.clamp_min(1e-8).log().unsqueeze(-1)
            - 0.5 * (distance + log_determinant)
        )
        posterior = log_joint.softmax(dim=2) * distribution[:, :, None]
        child_mass = posterior.sum(dim=-1).clamp_min(1e-6)
        mixture = child_mass / child_mass.sum(dim=2, keepdim=True)
        mean = torch.einsum(
            "bmln,bnd->bmld", posterior, coordinates.float()
        ) / child_mass[..., None]
        difference = coordinates[:, None, None].float() - mean[:, :, :, None]
        covariance = torch.einsum(
            "bmln,bmlni,bmlnj->bmlij",
            posterior,
            difference,
            difference,
        ) / child_mass[..., None, None]
        covariance = covariance + 1e-4 * identity

    precision = torch.linalg.inv(covariance)
    difference = coordinates[:, None, None].float() - mean[:, :, :, None]
    distance = torch.einsum(
        "bmlni,bmlij,bmlnj->bmln", difference, precision, difference
    )
    normalizer = covariance.det().clamp_min(1e-8).sqrt().unsqueeze(-1)
    density = mixture[..., None] * torch.exp(-0.5 * distance) / normalizer
    density = density.sum(dim=2) * valid[:, None].float()
    density = density / density.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return density * parent_mass


def render_basis(
    support: torch.Tensor,
    decoded_features: torch.Tensor,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    foreground = torch.einsum(
        "bmn,bmc->bnc", support.float(), decoded_features.float()
    )
    coverage = support.sum(dim=1).clamp(0.0, 1.0)
    rendered = foreground + (1.0 - coverage)[..., None] * background[:, None]
    return rendered, coverage


def sample_feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    weight = valid.float()
    return (error * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def sample_support_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction_coverage = prediction.sum(dim=1).clamp(0.0, 1.0)
    target_coverage = target.sum(dim=1).clamp(0.0, 1.0)
    predicted_parent = prediction / prediction.sum(dim=1, keepdim=True).clamp_min(
        1e-7
    )
    target_parent = target / target.sum(dim=1, keepdim=True).clamp_min(1e-7)
    midpoint = 0.5 * (predicted_parent + target_parent)
    divergence = 0.5 * (
        target_parent.clamp_min(1e-7)
        * (target_parent.clamp_min(1e-7).log() - midpoint.log())
        + predicted_parent.clamp_min(1e-7)
        * (predicted_parent.clamp_min(1e-7).log() - midpoint.log())
    ).sum(dim=1)
    foreground = valid.float() * target_coverage
    js = (divergence * foreground).sum(dim=1) / foreground.sum(dim=1).clamp_min(
        1.0
    )
    coverage = (
        (prediction_coverage - target_coverage).square() * valid.float()
    ).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
    return js, coverage


def audit_batch(
    tokens,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    child_counts: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, torch.Tensor]:
    active_assignment = tokens.assignment.float() * tokens.activation.float()
    background = (features.float() * valid[..., None].float()).sum(dim=1)
    background = background / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    result = {
        "token": sample_feature_error(
            tokens.reconstructed_features, features, valid
        )
    }
    for children in child_counts:
        support = fit_gaussian_basis(
            active_assignment, coordinates, valid, children
        )
        rendered, _ = render_basis(support, tokens.decoded_features, background)
        js, coverage = sample_support_error(support, active_assignment, valid)
        result[f"feature_{children}"] = sample_feature_error(
            rendered, features, valid
        )
        result[f"support_js_{children}"] = js
        result[f"coverage_{children}"] = coverage
    return result
