"""Numerically bounded normalization for learned feature vectors."""

from __future__ import annotations

import torch


def stable_unit_normalize(
    value: torch.Tensor,
    minimum_norm: float = 0.1,
) -> torch.Tensor:
    """Return unit vectors while bounding gradients of degenerate inputs."""
    if minimum_norm <= 0.0:
        raise ValueError("minimum normalization norm must be positive")
    value_float = value.float()
    scale = value_float.abs().amax(dim=-1, keepdim=True).clamp_min(minimum_norm)
    norm = scale * (value_float / scale).square().sum(dim=-1, keepdim=True).sqrt()
    return value_float / norm.clamp_min(minimum_norm)


def stable_rms_normalize(
    value: torch.Tensor,
    minimum_rms: float = 0.1,
) -> torch.Tensor:
    """Bound recurrent state scale without amplifying a near-zero state."""
    if minimum_rms <= 0.0:
        raise ValueError("minimum RMS must be positive")
    value_float = value.float()
    scale = value_float.abs().amax(dim=-1, keepdim=True).clamp_min(minimum_rms)
    rms = scale * (value_float / scale).square().mean(dim=-1, keepdim=True).sqrt()
    return (value_float / rms.clamp_min(minimum_rms)).to(value.dtype)
