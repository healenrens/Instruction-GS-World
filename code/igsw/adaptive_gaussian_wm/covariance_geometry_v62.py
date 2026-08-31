"""FP32 spatial moments and Mahalanobis geometry for v62 object fields."""

from __future__ import annotations

import torch


def _regularize_covariance(covariance, floor):
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eye = torch.eye(2, device=covariance.device, dtype=torch.float32)
    return covariance + float(floor) * eye


def carrier_spatial_moments_v62(coordinates, assignment, covariance_floor):
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        coordinates = coordinates.float()
        assignment = assignment.float()
        center = torch.einsum("bkp,bpd->bkd", assignment, coordinates)
        offset = coordinates[:, None] - center[:, :, None]
        covariance = torch.einsum("bkp,bkpi,bkpj->bkij", assignment, offset, offset)
        covariance = _regularize_covariance(covariance, covariance_floor)
    return center, covariance


def object_spatial_moments_v62(coordinates, weight, covariance_floor):
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        coordinates = coordinates.float()
        weight = weight.float()
        total = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        center = (coordinates * weight[..., None]).sum(dim=1) / total
        offset = coordinates - center[:, None]
        covariance = torch.einsum("bp,bpi,bpj->bij", weight, offset, offset)
        covariance = covariance / total[..., None]
        covariance = _regularize_covariance(covariance, covariance_floor)
    return center, covariance


def mahalanobis_squared_v62(offset, covariance):
    with torch.autocast(device_type=offset.device.type, enabled=False):
        offset = offset.float()
        covariance = 0.5 * (covariance.float() + covariance.float().transpose(-1, -2))
        factor = torch.linalg.cholesky(covariance)
        whitened = torch.linalg.solve_triangular(
            factor[:, None], offset[..., None], upper=False
        )
        squared = whitened[..., 0].square().sum(dim=-1)
    return squared
