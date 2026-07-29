"""Numerically stable linear algebra for residual-field capacity probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .object_centered_carrier_probe import feature_error


@dataclass
class RidgeSolution:
    prediction: torch.Tensor
    coefficients: torch.Tensor
    error: torch.Tensor
    condition_number: torch.Tensor
    coefficient_rms: torch.Tensor


def point_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().mean(dim=-1) + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )


def stabilize_covariance(
    covariance: torch.Tensor,
    floor: float,
    ceiling: float,
) -> torch.Tensor:
    covariance = 0.5 * (covariance.float() + covariance.float().transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp(floor**2, ceiling**2)
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)


def ridge_solution(
    design: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ridge: float,
) -> RidgeSolution:
    weight = valid.float().sqrt()[:, None]
    weighted_design = design.float() * weight
    weighted_target = target.float() * weight
    column_scale = weighted_design.square().sum(dim=0).sqrt().clamp_min(1e-6)
    normalized = weighted_design / column_scale[None]
    gram = normalized.transpose(0, 1) @ normalized
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    cholesky, info = torch.linalg.cholesky_ex(regularized)
    if bool((info != 0).any()):
        raise RuntimeError(f"ridge system is not positive definite: {info.tolist()}")
    right = normalized.transpose(0, 1) @ weighted_target
    scaled_coefficients = torch.cholesky_solve(right, cholesky)
    coefficients = scaled_coefficients / column_scale[:, None]
    prediction = design.float() @ coefficients
    eigenvalues = torch.linalg.eigvalsh(regularized)
    condition = eigenvalues[-1] / eigenvalues[0].clamp_min(1e-12)
    return RidgeSolution(
        prediction=prediction,
        coefficients=coefficients,
        error=feature_error(prediction, target, valid),
        condition_number=condition,
        coefficient_rms=coefficients.square().mean().sqrt(),
    )


def rbf(
    coordinates: torch.Tensor,
    centers: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    if centers.numel() == 0:
        return coordinates.new_zeros((coordinates.shape[0], 0), dtype=torch.float32)
    difference = coordinates[:, None].float() - centers[None].float()
    cholesky = torch.linalg.cholesky(covariance.float())
    whitened = torch.linalg.solve_triangular(
        cholesky, difference.permute(1, 2, 0), upper=False
    )
    distance = whitened.square().sum(dim=1).transpose(0, 1)
    return torch.exp(-0.5 * distance.clamp_max(160.0))
