"""Scale-free object geometry derived from image-plane Gaussian supports."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig
from .gpstoken import GPSTokenState


RELATION_DIM = 6


@dataclass
class ObjectGeometryState:
    center: torch.Tensor
    relative_scale: torch.Tensor
    relative_disparity: torch.Tensor
    relations: torch.Tensor


def gaussian_linear_scale(covariance: torch.Tensor) -> torch.Tensor:
    """Return ellipse linear support; global image scaling remains factored out."""
    if covariance.shape[-2:] != (2, 2):
        raise ValueError("covariance must end with [2,2]")
    determinant = torch.linalg.det(covariance.float()).clamp_min(1e-12)
    return determinant.pow(0.25).to(covariance.dtype)


def pairwise_relative_geometry(
    center: torch.Tensor,
    relative_scale: torch.Tensor,
    relative_disparity: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Build translation- and scale-normalized ordered object relations."""
    if center.shape[:-1] != relative_scale.shape:
        raise ValueError("center and relative_scale must share object axes")
    if relative_disparity.shape != relative_scale.shape:
        raise ValueError("relative_disparity must match relative_scale")
    if visibility.shape != relative_scale.shape:
        raise ValueError("visibility must match relative_scale")
    if center.shape[-1] != 2:
        raise ValueError("center must end with two image coordinates")

    center_delta = center.unsqueeze(-3) - center.unsqueeze(-2)
    scale_i = relative_scale.unsqueeze(-1)
    scale_j = relative_scale.unsqueeze(-2)
    scale_mean = 0.5 * (scale_i + scale_j)
    normalized_center = center_delta / scale_mean.clamp_min(1e-4)[..., None]
    log_scale_ratio = (
        scale_j.clamp_min(1e-6).log() - scale_i.clamp_min(1e-6).log()
    )[..., None]
    disparity_i = relative_disparity.unsqueeze(-1)
    disparity_j = relative_disparity.unsqueeze(-2)
    disparity_delta = (disparity_j - disparity_i)[..., None]
    depth_order = torch.tanh(5.0 * disparity_delta)
    pair_visibility = (
        visibility.unsqueeze(-1) * visibility.unsqueeze(-2)
    )[..., None]
    return torch.cat(
        (
            normalized_center,
            log_scale_ratio,
            disparity_delta,
            depth_order,
            pair_visibility,
        ),
        dim=-1,
    )


def pool_object_geometry(
    tokens: GPSTokenState,
    assignment: torch.Tensor,
    visibility: torch.Tensor,
) -> ObjectGeometryState:
    if assignment.shape[:2] != tokens.latent.shape[:2]:
        raise ValueError("assignment and GPSTokens do not align")
    if visibility.shape != assignment.shape[:1] + assignment.shape[2:]:
        raise ValueError("visibility must have shape [B,K]")
    weight = assignment * tokens.activation
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    center = torch.einsum("bmk,bmd->bkd", weight, tokens.center)
    token_scale = gaussian_linear_scale(tokens.covariance)
    relative_scale = torch.exp(
        torch.einsum(
            "bmk,bm->bk",
            weight,
            token_scale.clamp_min(1e-6).log(),
        )
    )
    relative_disparity = torch.einsum(
        "bmk,bm->bk",
        weight,
        tokens.depth_order.squeeze(-1),
    )
    relations = pairwise_relative_geometry(
        center,
        relative_scale,
        relative_disparity,
        visibility,
    )
    return ObjectGeometryState(
        center=center,
        relative_scale=relative_scale,
        relative_disparity=relative_disparity,
        relations=relations,
    )


class RelativeGeometryEncoder(nn.Module):
    """Encode pair relations for memory prediction and attention bias."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        hidden = config.memory_relation_dim
        self.relation = nn.Sequential(
            nn.Linear(RELATION_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, config.object_dim),
        )
        self.attention_bias = nn.Sequential(
            nn.Linear(RELATION_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.heads),
        )

    def forward(self, relations: torch.Tensor) -> torch.Tensor:
        if relations.shape[-1] != RELATION_DIM:
            raise ValueError("relations have an unexpected feature dimension")
        encoded = self.relation(relations)
        pair_weight = relations[..., -1:].to(encoded.dtype)
        pooled = (encoded * pair_weight).sum(dim=-2)
        pooled = pooled / pair_weight.sum(dim=-2).clamp_min(1e-6)
        return self.summary(pooled)

    def bias(self, relations: torch.Tensor) -> torch.Tensor:
        if relations.shape[-1] != RELATION_DIM:
            raise ValueError("relations have an unexpected feature dimension")
        return self.attention_bias(relations).movedim(-1, -3)
