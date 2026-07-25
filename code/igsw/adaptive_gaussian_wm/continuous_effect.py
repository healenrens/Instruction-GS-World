"""Continuous bounded latent effects inferred from future object state."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_posterior import weighted_slot_pool
from .config import AdaptiveGaussianWMConfig


class ContinuousEffectPosterior(nn.Module):
    """Compress a transition into latent directions and bounded magnitudes."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.model_dim
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.current_input = nn.Linear(config.object_dim, dim)
        self.future_input = nn.Linear(config.object_dim, dim)
        self.activity_input = nn.Linear(1, dim)
        self.history_input = nn.Linear(config.object_dim * 2, dim)
        self.gap_input = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.queries = nn.Parameter(
            torch.randn(config.action_tokens, dim) / dim**0.5
        )
        self.attention = nn.MultiheadAttention(
            dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.direction = nn.Linear(dim, config.action_dim)
        self.magnitude = nn.Linear(dim, 1)

    def forward(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
        future_slots: torch.Tensor,
        future_activity: torch.Tensor,
        future_scale: torch.Tensor,
        history_centers: torch.Tensor | None = None,
        future_centers: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        current_object_rgb: torch.Tensor | None = None,
        future_object_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del history_centers, future_centers
        del current_object_rgb, future_object_rgb
        if condition is not None:
            raise ValueError("continuous effect core is language-free")
        current = history_slots[:, -1]
        batch, future_count, object_count, _ = future_slots.shape
        current_expanded = current[:, None].expand(
            -1, future_count, -1, -1
        )
        object_tokens = (
            self.current_input(current_expanded)
            + self.future_input(future_slots)
            + self.activity_input(future_activity[..., None])
        )
        pooled = weighted_slot_pool(history_slots, history_activity)
        history = self.history_input(
            torch.cat((pooled.mean(dim=1), pooled[:, -1]), dim=-1)
        )
        queries = (
            self.queries[None, None]
            + history[:, None, None]
            + self.gap_input(future_scale[..., None])[:, :, None]
        )
        encoded = self.attention(
            queries.reshape(
                batch * future_count,
                self.action_tokens,
                -1,
            ),
            object_tokens.reshape(
                batch * future_count,
                object_count,
                -1,
            ),
            object_tokens.reshape(
                batch * future_count,
                object_count,
                -1,
            ),
            need_weights=False,
        )[0]
        encoded = self.output_norm(
            encoded
            + queries.reshape(
                batch * future_count,
                self.action_tokens,
                -1,
            )
        )
        direction = F.normalize(self.direction(encoded), dim=-1, eps=1e-6)
        magnitude = torch.sigmoid(self.magnitude(encoded))
        return (direction * magnitude).reshape(
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
