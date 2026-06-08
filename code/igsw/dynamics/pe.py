"""3D Fourier positional encoding of Gaussian centers.

NeRF-style γ(p) = [p, sin(2^k π p), cos(2^k π p) for k=0..L-1] applied per xyz.
This is the ONLY source of spatial information for the dynamics tokens, which
keeps the model permutation-equivariant and count-agnostic (the recommendation
from briefs C and D). Output dim = 3 * (1 + 2L).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierPE3D(nn.Module):
    def __init__(self, num_freqs: int = 10, include_input: bool = True, max_freq_log2: float | None = None):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1
        freqs = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)  # [L]
        self.register_buffer("freqs", freqs * torch.pi, persistent=False)

    @property
    def out_dim(self) -> int:
        return 3 * ((1 if self.include_input else 0) + 2 * self.num_freqs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz [...,3] -> [..., out_dim]."""
        out = [xyz] if self.include_input else []
        ang = xyz[..., None] * self.freqs  # [...,3,L]
        s = torch.sin(ang).flatten(-2)     # [...,3L]
        c = torch.cos(ang).flatten(-2)
        out.extend([s, c])
        return torch.cat(out, dim=-1)
