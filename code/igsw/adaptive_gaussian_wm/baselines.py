"""Non-object baselines used by feasibility experiments."""
from __future__ import annotations

import torch
import torch.nn as nn


class FlatFeaturePredictor(nn.Module):
    """Per-grid deterministic baseline without GPSTokens or object slots."""

    def __init__(self, feature_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        history_features: torch.Tensor,
        future_coordinates: torch.Tensor,
        future_scale: torch.Tensor,
    ) -> torch.Tensor:
        current = history_features[:, -1]
        pooled = current.mean(dim=1)
        _, future_count, grid_count = future_coordinates.shape[:3]
        current = current[:, None].expand(-1, future_count, -1, -1)
        pooled = pooled[:, None, None].expand(-1, future_count, grid_count, -1)
        scale = future_scale[:, :, None, None].expand(-1, -1, grid_count, -1)
        inputs = torch.cat(
            (current, pooled, future_coordinates, scale),
            dim=-1,
        )
        return current + self.network(inputs)
