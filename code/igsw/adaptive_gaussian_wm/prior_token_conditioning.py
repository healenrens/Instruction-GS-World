"""Object-query cross-attention over exact frozen-Qwen instruction tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig


class PriorTokenConditioner(nn.Module):
    """Add a zero-initialized token-language residual to object queries."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        if config.condition_dim <= 0:
            raise ValueError("token-conditioned Prior requires condition_dim")
        self.condition_dim = config.condition_dim
        self.model_dim = config.model_dim
        self.token_input = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def pooled_semantic(
        self,
        token_features: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        if token_features.ndim != 3:
            raise ValueError("instruction tokens must have shape [U,L,D]")
        if (
            token_features.shape[-1] != self.condition_dim
            or token_valid.shape != token_features.shape[:2]
        ):
            raise ValueError("instruction token bank shape mismatch")
        weights = token_valid.to(token_features.dtype)
        pooled = (
            token_features * weights[..., None]
        ).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        projected = self.token_input(pooled.float())
        return self.output_norm(projected)

    def forward(
        self,
        object_queries: torch.Tensor,
        token_features: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        if object_queries.ndim != 3:
            raise ValueError("object queries must have shape [B,K,D]")
        expected = (
            object_queries.shape[0],
            token_features.shape[1],
            self.condition_dim,
        )
        if token_features.shape != expected:
            raise ValueError(
                f"instruction tokens must have shape {expected}"
            )
        if token_valid.shape != token_features.shape[:2]:
            raise ValueError("instruction token mask must have shape [B,L]")
        if token_valid.dtype != torch.bool or not bool(
            token_valid.any(dim=1).all()
        ):
            raise ValueError("each instruction must contain a valid token")
        tokens = self.token_input(token_features.float()).to(
            object_queries.dtype
        )
        attended = self.attention(
            object_queries,
            tokens,
            tokens,
            key_padding_mask=~token_valid,
            need_weights=False,
        )[0]
        return torch.tanh(self.gate) * self.output_norm(attended)
