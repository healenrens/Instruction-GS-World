"""Compose bounded short and tail effects without exposing explicit motion labels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig


class LatentEffectComposer(nn.Module):
    """Map two continuous effect sets to one bounded transition effect."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        hidden = config.model_dim
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.input = nn.Linear(config.action_dim * 4, hidden, bias=False)
        self.identity = nn.Parameter(
            torch.randn(config.action_tokens, hidden) / hidden**0.5
        )
        self.attention = nn.MultiheadAttention(
            hidden,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.direction = nn.Linear(hidden, config.action_dim, bias=False)
        self.magnitude = nn.Linear(hidden, 1, bias=False)

    def forward(
        self,
        short_effect: torch.Tensor,
        tail_effect: torch.Tensor,
    ) -> torch.Tensor:
        expected = (*short_effect.shape[:-2], self.action_tokens, self.action_dim)
        if short_effect.shape != expected or tail_effect.shape != expected:
            raise ValueError("effect composer expects matching [...,A,D] tensors")
        shape = short_effect.shape
        short = short_effect.reshape(-1, self.action_tokens, self.action_dim)
        tail = tail_effect.reshape_as(short)
        hidden = self.input(
            torch.cat((short, tail, tail - short, short * tail), dim=-1)
        )
        hidden = hidden + self.identity[None]
        attended = self.attention(hidden, hidden, hidden, need_weights=False)[0]
        hidden = self.norm(hidden + attended - self.identity[None])
        direction = F.normalize(self.direction(hidden), dim=-1, eps=1e-6)
        amplitude = torch.tanh(short.norm(dim=-1) + tail.norm(dim=-1))
        magnitude = torch.sigmoid(self.magnitude(hidden).squeeze(-1)) * amplitude
        return (direction * magnitude[..., None]).reshape(shape)
