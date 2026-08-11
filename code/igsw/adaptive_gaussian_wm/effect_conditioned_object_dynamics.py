"""Object dynamics whose residual is exactly gated by a continuous effect."""

from __future__ import annotations

import torch
import torch.nn as nn

from .v44_config import TemporalObjectSetConfig


class EffectConditionedObjectDynamics(nn.Module):
    """Predict only effect-conditioned object changes; zero effect is identity."""

    def __init__(self, config: TemporalObjectSetConfig):
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.object_input = nn.Sequential(
            nn.Linear(config.semantic_dim + config.dynamic_dim + 8, dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(dim),
        )
        self.action_input = nn.Linear(config.action_dim, dim, bias=False)
        self.action_identity = nn.Parameter(
            torch.randn(config.action_tokens, dim) / dim**0.5
        )
        self.cross_attention = nn.MultiheadAttention(
            dim, config.heads, dropout=config.dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            dim,
            config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.time = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dynamic_delta = nn.Linear(dim, config.dynamic_dim)
        self.geometry_delta = nn.Linear(dim, 4)
        self.lifecycle_delta = nn.Linear(dim, 2)

    def forward(
        self,
        current: dict[str, torch.Tensor],
        effect: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        count = self.config.object_slots
        expected = (len(effect), self.config.action_tokens, self.config.action_dim)
        if effect.shape != expected:
            raise ValueError(f"effect shape {tuple(effect.shape)} != {expected}")
        if delta_time.shape != (len(effect),):
            raise ValueError("dynamics delta time must have shape [B]")
        semantic = current["semantic"][:, :count]
        dynamic = current["dynamic"][:, :count]
        center = current["center"][:, :count].float()
        log_scale = current["log_scale"][:, :count].float()
        presence = current["presence"][:, :count].float()
        visibility = current["visibility"][:, :count].float()
        geometry = torch.cat(
            (
                center,
                log_scale,
                presence[..., None],
                visibility[..., None],
                center.square(),
            ),
            dim=-1,
        )
        objects = self.object_input(
            torch.cat((semantic, dynamic, geometry.to(dynamic.dtype)), dim=-1)
        )
        time = torch.stack(
            (delta_time.float(), torch.log1p(delta_time.float())), dim=-1
        )
        objects = objects + self.time(time).to(objects.dtype)[:, None]
        actions = self.action_input(effect) + self.action_identity[None]
        conditioned = self.cross_attention(
            objects, actions, actions, need_weights=False
        )[0]
        hidden = self.interaction(objects + conditioned)
        gate = effect.float().norm(dim=-1).mean(dim=-1).clamp(0.0, 1.0)
        gate = gate[:, None, None]
        dynamic_delta = torch.tanh(self.dynamic_delta(hidden).float()) * gate
        geometry_delta = torch.tanh(self.geometry_delta(hidden).float()) * gate
        lifecycle_delta = self.lifecycle_delta(hidden).float() * gate
        presence_logits = torch.logit(presence.clamp(1e-4, 1 - 1e-4))
        visibility_logits = torch.logit(visibility.clamp(1e-4, 1 - 1e-4))
        return {
            "semantic": semantic,
            "dynamic": dynamic + dynamic_delta.to(dynamic.dtype),
            "center": (center + 0.5 * geometry_delta[..., :2]).clamp(-1.5, 1.5),
            "log_scale": (log_scale + 0.25 * geometry_delta[..., 2:]).clamp(-3.0, 0.7),
            "presence": torch.sigmoid(presence_logits + lifecycle_delta[..., 0]),
            "visibility": torch.sigmoid(visibility_logits + lifecycle_delta[..., 1]),
        }
