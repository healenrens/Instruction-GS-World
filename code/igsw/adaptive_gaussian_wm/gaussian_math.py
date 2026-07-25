"""Numerically stable Gaussian geometry operations."""
from __future__ import annotations

import torch


def precision_2d(covariance: torch.Tensor) -> torch.Tensor:
    """Invert symmetric 2x2 covariance with a determinant safety floor."""
    if covariance.shape[-2:] != (2, 2):
        raise ValueError("Gaussian covariance must be two-dimensional")
    symmetric = 0.5 * (covariance.float() + covariance.float().transpose(-1, -2))
    xx = symmetric[..., 0, 0]
    xy = symmetric[..., 0, 1]
    yy = symmetric[..., 1, 1]
    determinant = (xx * yy - xy.square()).clamp_min(1e-8)
    precision = torch.stack((yy, -xy, -xy, xx), dim=-1)
    return precision.reshape(*covariance.shape[:-2], 2, 2) / determinant[
        ..., None, None
    ]


def mahalanobis_squared_from_precision(
    precision: torch.Tensor,
    difference: torch.Tensor,
) -> torch.Tensor:
    """Compute bounded squared distances from an analytic precision matrix."""
    if precision.shape[-2:] != (2, 2) or difference.shape[-1] != 2:
        raise ValueError("Gaussian geometry must be two-dimensional")
    if difference.shape[:-2] != precision.shape[:-2]:
        raise ValueError("precision and difference leading dimensions differ")
    dx = difference[..., 0].float()
    dy = difference[..., 1].float()
    distance = (
        precision[..., 0, 0, None] * dx.square()
        + 2.0 * precision[..., 0, 1, None] * dx * dy
        + precision[..., 1, 1, None] * dy.square()
    )
    return distance.clamp(0.0, 80.0)
