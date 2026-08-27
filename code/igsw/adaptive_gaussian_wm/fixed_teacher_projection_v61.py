"""Parameter-free teacher projections for the v61 comparison space."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fixed_group_projection_v61(features: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Compress contiguous feature groups without introducing trainable targets."""

    input_dim = features.shape[-1]
    if input_dim % output_dim:
        raise ValueError(
            f"v61 fixed projection requires {input_dim} divisible by {output_dim}"
        )
    group = input_dim // output_dim
    projected = features.float().reshape(*features.shape[:-1], output_dim, group)
    projected = projected.mean(dim=-1) * group**0.5
    return F.normalize(projected, dim=-1, eps=1e-6)
