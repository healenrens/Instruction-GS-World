"""Tokenize a GaussianSet into transformer input features.

Each Gaussian -> a token assembled from:
    FourierPE3D(normalized μ)  ⊕  q(4)  ⊕  log s(3)  ⊕  logit σ(1)  ⊕  c(3)  [⊕ f]
Positions are normalized (center / radius of the current set) ONLY for the PE
feature so frequencies are meaningful regardless of the per-scene gauge; the
actual μ update downstream stays in world units.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..gaussians.types import GaussianSet
from .pe import FourierPE3D


def normalize_centers(means: torch.Tensor, eps: float = 1e-6):
    """means [B,N,3] -> (normalized, center[B,1,3], radius[B,1,1])."""
    center = means.mean(dim=1, keepdim=True)
    radius = (means - center).norm(dim=-1).amax(dim=1).clamp_min(eps)[:, None, None]
    return (means - center) / radius, center, radius


class GaussianTokenizer(nn.Module):
    def __init__(self, d_model: int, num_freqs: int = 10, feature_dim: int = 0):
        super().__init__()
        self.pe = FourierPE3D(num_freqs=num_freqs)
        self.feature_dim = feature_dim
        in_dim = self.pe.out_dim + 4 + 3 + 1 + 3 + feature_dim
        self.in_dim = in_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, gs_batch: list[GaussianSet]) -> tuple[torch.Tensor, torch.Tensor]:
        """list of B GaussianSets with the SAME N -> tokens [B,N,d_model], mask not needed.

        For variable N use pad+mask (see model). Here all sets share N (downsampled).
        """
        means = torch.stack([g.means for g in gs_batch], dim=0)        # [B,N,3]
        quats = torch.stack([g.quats for g in gs_batch], dim=0)
        log_s = torch.stack([g.log_scales for g in gs_batch], dim=0)
        logit_o = torch.stack([g.opacity_logits for g in gs_batch], dim=0)[..., None]  # [B,N,1]
        cols = torch.stack([g.colors for g in gs_batch], dim=0)
        norm_means, _, _ = normalize_centers(means)
        feats = [self.pe(norm_means), quats, log_s, logit_o, cols]
        if self.feature_dim > 0:
            feats.append(torch.stack([g.features for g in gs_batch], dim=0))
        x = torch.cat(feats, dim=-1)
        return self.proj(x)

    def tokenize_tensors(
        self, means, quats, log_s, logit_o, cols, features=None
    ) -> torch.Tensor:
        """Batched tensor path: means[B,N,3], quats[B,N,4], log_s[B,N,3],
        logit_o[B,N], cols[B,N,3], features[B,N,D]."""
        norm_means, _, _ = normalize_centers(means)
        feats = [self.pe(norm_means), quats, log_s, logit_o[..., None], cols]
        if self.feature_dim > 0 and features is not None:
            feats.append(features)
        return self.proj(torch.cat(feats, dim=-1))
