"""Independent frozen-checkpoint probe from compact regions back to DINO patches."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig


class CompactStateProbe(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.object_slots = config.object_slots
        self.root_projection = nn.Linear(config.object_dim, config.region_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(config.region_dim),
            nn.Linear(config.region_dim, config.region_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.region_dim * 2, config.feature_dim),
        )

    @staticmethod
    def _gaussian_assignment(
        center: torch.Tensor,
        covariance: torch.Tensor,
        coordinates: torch.Tensor,
        presence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        identity = torch.eye(2, device=center.device, dtype=torch.float32)
        precision = torch.linalg.inv(covariance.float() + 1e-4 * identity)
        difference = coordinates[:, None].float() - center[:, :, None].float()
        distance = torch.einsum(
            "brni,brij,brnj->brn", difference, precision, difference
        )
        logits = -0.5 * distance + presence.float().clamp_min(1e-6).log()[
            ..., None
        ]
        logits = logits.masked_fill(
            ~valid[:, None], torch.finfo(logits.dtype).min
        )
        assignment = logits.softmax(dim=1)
        return assignment * valid[:, None].to(assignment.dtype)

    def forward(
        self,
        region_feature: torch.Tensor,
        root_slots: torch.Tensor,
        owner: torch.Tensor,
        center: torch.Tensor,
        covariance: torch.Tensor,
        coordinates: torch.Tensor,
        presence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        object_owner = owner[..., : self.object_slots]
        root_context = torch.einsum(
            "brk,bkd->brd", object_owner, self.root_projection(root_slots)
        )
        decoded = self.decoder(region_feature + root_context)
        assignment = self._gaussian_assignment(
            center, covariance, coordinates, presence, valid
        )
        return torch.einsum("brn,brd->bnd", assignment, decoded.float())
