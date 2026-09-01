"""Deterministic object-transition effect posterior for v62 E1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ObjectEffectV62:
    value: torch.Tensor


def zero_object_effect_v62(effect: ObjectEffectV62) -> ObjectEffectV62:
    return ObjectEffectV62(value=torch.zeros_like(effect.value))


def state_tokens_without_identity_v62(state):
    return torch.cat(
        (
            state.carriers,
            state.center,
            state.covariance.flatten(-2),
            state.presence[..., None],
            state.visibility[..., None],
        ),
        dim=-1,
    )


class DeterministicObjectEffectPosteriorV62(nn.Module):
    """Read source and target states without explicit center-delta concatenation."""

    def __init__(self, config):
        super().__init__()
        dim = config.state_dim
        self.config = config
        self.state_input = nn.Sequential(nn.LayerNorm(dim + 8), nn.Linear(dim + 8, dim))
        self.source_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.target_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.time_input = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.queries = nn.Parameter(torch.randn(config.effect_factors, dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            dim, config.effect_heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, config.effect_dim),
            nn.Tanh(),
        )

    def forward(self, source, target, delta_seconds) -> ObjectEffectV62:
        state_dtype = self.state_input[1].weight.dtype
        source_tokens = self.state_input(
            state_tokens_without_identity_v62(source).to(dtype=state_dtype)
        )
        target_tokens = self.state_input(
            state_tokens_without_identity_v62(target).to(dtype=state_dtype)
        )
        context = torch.cat(
            (source_tokens + self.source_type, target_tokens + self.target_type), dim=1
        )
        time_value = torch.log1p(delta_seconds.float())[:, None]
        time = self.time_input(
            time_value.to(dtype=self.time_input[0].weight.dtype)
        )[:, None]
        queries = self.queries[None].expand(len(context), -1, -1) + time
        hidden, _ = self.attention(queries, context, context, need_weights=False)
        return ObjectEffectV62(value=self.output(hidden))
