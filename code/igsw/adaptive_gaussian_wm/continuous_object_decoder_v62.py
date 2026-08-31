"""Compositional continuous decoder for v62 object carriers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .covariance_geometry_v62 import mahalanobis_squared_v62


@dataclass(frozen=True)
class DecodedObjectFieldV62:
    support_logits: torch.Tensor
    dino: torch.Tensor
    siglip: torch.Tensor
    visibility_logits: torch.Tensor
    carrier_weights: torch.Tensor


class ContinuousObjectDecoderV62(nn.Module):
    """Decode a single object only at requested continuous image coordinates."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.state_dim
        self.carrier_value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.coordinate = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.field = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.support = nn.Linear(dim, 1)
        self.dino = nn.Linear(dim, config.semantic_dim)
        self.siglip = nn.Linear(dim, config.semantic_dim)
        self.visibility = nn.Linear(dim, 1)

    def forward(self, state, coordinates: torch.Tensor) -> DecodedObjectFieldV62:
        coordinates = coordinates.float()
        offset = coordinates[:, :, None] - state.center[:, None].float()
        squared = mahalanobis_squared_v62(offset, state.covariance)
        presence = state.presence.float().clamp(1e-4, 1.0 - 1e-4)
        basis_logits = -0.5 * squared + torch.logit(presence)[:, None]
        carrier_weights = torch.softmax(basis_logits, dim=-1)
        carrier_values = self.carrier_value(state.carriers)
        mixed = torch.einsum("bpk,bkd->bpd", carrier_weights, carrier_values)
        hidden = mixed + self.coordinate(coordinates)
        hidden = hidden + self.field(hidden)
        carrier_visibility = torch.logit(
            state.visibility.float().clamp(1e-4, 1.0 - 1e-4)
        )
        visibility_logits = self.visibility(hidden)[..., 0]
        visibility_logits = visibility_logits + torch.einsum(
            "bpk,bk->bp", carrier_weights, carrier_visibility
        )
        return DecodedObjectFieldV62(
            support_logits=self.support(hidden)[..., 0],
            dino=F.normalize(self.dino(hidden).float(), dim=-1, eps=1e-6),
            siglip=F.normalize(self.siglip(hidden).float(), dim=-1, eps=1e-6),
            visibility_logits=visibility_logits,
            carrier_weights=carrier_weights,
        )
