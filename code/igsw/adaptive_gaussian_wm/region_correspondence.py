"""Owner-aware causal correspondence between persistent and observed regions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig


@dataclass
class RegionAssociation:
    matrix: torch.Tensor
    confidence: torch.Tensor
    aligned_feature: torch.Tensor
    aligned_center: torch.Tensor
    aligned_covariance: torch.Tensor
    aligned_owner: torch.Tensor
    aligned_relative_center: torch.Tensor
    aligned_presence: torch.Tensor
    aligned_visibility: torch.Tensor
    aligned_identity: torch.Tensor
    aligned_detail_latent: torch.Tensor
    aligned_detail_valid: torch.Tensor
    aligned_detail_gate: torch.Tensor


class CausalRegionCorrespondence(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.temperature = config.correspondence_temperature
        self.identity_weight = nn.Parameter(torch.tensor(1.0))
        self.geometry_weight = nn.Parameter(torch.tensor(1.0))
        self.owner_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, previous, observation) -> RegionAssociation:
        if previous.feature.shape != observation.feature.shape:
            raise ValueError("region correspondence feature shapes differ")
        observed = observation.presence > 1e-4

        def observed_value(value: torch.Tensor) -> torch.Tensor:
            mask = observed
            while mask.ndim < value.ndim:
                mask = mask[..., None]
            return torch.where(mask, value, torch.zeros_like(value))

        identity = torch.einsum(
            "brd,bsd->brs",
            F.normalize(previous.identity_key.float(), dim=-1),
            F.normalize(observed_value(observation.identity_key).float(), dim=-1),
        )
        geometry = (
            previous.relative_center[:, :, None]
            - observed_value(observation.relative_center)[:, None]
        ).square().sum(dim=-1)
        owner = torch.einsum(
            "bro,bso->brs",
            previous.owner.float(),
            observed_value(observation.owner).float(),
        )
        score = (
            self.identity_weight.abs() * identity
            - self.geometry_weight.abs() * geometry
            + self.owner_weight.abs() * owner
        ) / self.temperature
        score = score.masked_fill(
            observation.presence[:, None] < 1e-4,
            torch.finfo(score.dtype).min,
        )
        matrix = torch.softmax(score, dim=-1).to(previous.feature.dtype)
        confidence = matrix.max(dim=-1).values * previous.presence

        def align(value: torch.Tensor) -> torch.Tensor:
            return torch.einsum(
                "brs,bs...->br...", matrix, observed_value(value)
            )

        return RegionAssociation(
            matrix=matrix,
            confidence=confidence,
            aligned_feature=align(observation.feature),
            aligned_center=align(observation.center),
            aligned_covariance=align(observation.covariance),
            aligned_owner=align(observation.owner),
            aligned_relative_center=align(observation.relative_center),
            aligned_presence=align(observation.presence),
            aligned_visibility=align(observation.visibility),
            aligned_identity=align(observation.identity_key),
            aligned_detail_latent=align(observation.detail_latent),
            aligned_detail_valid=align(observation.detail_valid),
            aligned_detail_gate=align(observation.detail_gate),
        )
