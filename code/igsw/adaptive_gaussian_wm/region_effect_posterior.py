"""Continuous latent effect inferred only from compact root/region transitions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig


class RegionEffectPosterior(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.region_dim
        self.config = config
        self.root_delta = nn.Linear(config.object_dim, dim)
        self.region_delta = nn.Linear(config.region_dim, dim)
        self.owner_identity = nn.Parameter(
            torch.randn(config.region_owners, dim) / dim**0.5
        )
        self.root_identity = nn.Parameter(
            torch.randn(config.object_slots, dim) / dim**0.5
        )
        layer = nn.TransformerEncoderLayer(
            dim,
            config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transition_encoder = nn.TransformerEncoder(layer, 4)
        self.effect_queries = nn.Parameter(
            torch.randn(config.action_tokens, dim) / dim**0.5
        )
        self.query_attention = nn.MultiheadAttention(
            dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(dim)
        self.direction = nn.Linear(dim, config.action_dim)
        self.magnitude = nn.Linear(dim, 1)

    def _owner_pool(
        self,
        delta: torch.Tensor,
        owner: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        weight = owner * presence[..., None]
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bro,brd->bod", weight, delta)

    def forward(
        self,
        current_roots: torch.Tensor,
        future_roots: torch.Tensor,
        current_regions,
        future_regions,
    ) -> torch.Tensor:
        if current_roots.ndim != 3 or future_roots.shape != current_roots.shape:
            raise ValueError("posterior roots must have matching [B,K,D] shapes")
        root_tokens = (
            self.root_delta(future_roots - current_roots) + self.root_identity
        )
        region_change = future_regions.feature - current_regions.feature
        owner = 0.5 * (current_regions.owner + future_regions.owner)
        presence = torch.minimum(
            current_regions.presence, future_regions.presence
        )
        owner_tokens = self.region_delta(
            self._owner_pool(region_change, owner, presence)
        ) + self.owner_identity
        transition = self.transition_encoder(
            torch.cat((root_tokens, owner_tokens), dim=1)
        )
        queries = self.effect_queries[None].expand(len(transition), -1, -1)
        encoded = self.query_attention(
            queries, transition, transition, need_weights=False
        )[0]
        encoded = self.output_norm(encoded + queries)
        direction = F.normalize(self.direction(encoded), dim=-1, eps=1e-6)
        magnitude = torch.sigmoid(self.magnitude(encoded))
        return direction * magnitude
