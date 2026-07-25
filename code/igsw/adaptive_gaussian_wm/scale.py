"""Signed physical-gap scaling and scale-modulated Transformer blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


def signed_gap_scale(delta_time: torch.Tensor, reference: float) -> torch.Tensor:
    """Map physical time gaps to a signed, dimensionless logarithmic scale."""
    if reference <= 0.0:
        raise ValueError("reference must be positive")
    return torch.sign(delta_time) * torch.log1p(delta_time.abs() / reference)


def inverse_signed_gap_scale(scale: torch.Tensor, reference: float) -> torch.Tensor:
    """Recover physical gaps from the signed logarithmic scale."""
    if reference <= 0.0:
        raise ValueError("reference must be positive")
    return torch.sign(scale) * torch.expm1(scale.abs()) * reference


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + torch.tanh(scale)) + shift


class ScaleModulatedBlock(nn.Module):
    """Full-attention block whose residual branches are modulated by gap scale."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.modulation = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        with torch.no_grad():
            self.modulation[-1].bias[dim * 2 : dim * 3].fill_(0.1)
            self.modulation[-1].bias[dim * 5 : dim * 6].fill_(0.1)

    def forward(
        self,
        tokens: torch.Tensor,
        scale: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        extra_modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scale.shape != tokens.shape[:2]:
            raise ValueError(
                f"scale must have shape {tokens.shape[:2]}, got {scale.shape}"
            )
        params = self.modulation(scale[..., None])
        if extra_modulation is not None:
            if extra_modulation.shape != (*tokens.shape[:2], tokens.shape[-1] * 6):
                raise ValueError("extra modulation must have shape [B,L,6D]")
            params = params + extra_modulation
        params = params.chunk(6, dim=-1)
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = params
        attention_input = _modulate(
            self.norm_attention(tokens),
            attn_shift,
            attn_scale,
        )
        attended = self.attention(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        tokens = tokens + torch.tanh(attn_gate) * attended
        mlp_input = _modulate(self.norm_mlp(tokens), mlp_shift, mlp_scale)
        return tokens + torch.tanh(mlp_gate) * self.mlp(mlp_input)
