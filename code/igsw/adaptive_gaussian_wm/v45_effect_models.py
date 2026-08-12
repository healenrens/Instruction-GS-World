"""Continuous video and image-goal effects without unit-norm saturation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v45_config import PredictiveObjectTubeConfig


class _EffectReadout(nn.Module):
    def __init__(self, config: PredictiveObjectTubeConfig):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(config.action_tokens, config.model_dim) / config.model_dim**0.5
        )
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.raw_effect = nn.Linear(config.model_dim, config.action_dim)
        self.log_scale = nn.Parameter(
            torch.full((config.action_tokens, config.action_dim), -1.5)
        )

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        queries = self.queries[None].expand(len(transitions), -1, -1)
        attended = self.attention(
            queries, transitions, transitions, need_weights=False
        )[0]
        hidden = self.norm(queries + attended)
        scale = F.softplus(self.log_scale.float())
        effect = torch.tanh(self.raw_effect(hidden).float()) * scale[None]
        return effect.to(hidden.dtype)


def _transition_features(
    state: dict[str, torch.Tensor], object_slots: int
) -> torch.Tensor:
    semantic = state["semantic"][:, :, :object_slots].float()
    dynamic = state["dynamic"][:, :, :object_slots].float()
    center = state["center"][:, :, :object_slots].float()
    scale = state["log_scale"][:, :, :object_slots].float()
    presence = state["presence"][:, :, :object_slots].float()
    visibility = state["visibility"][:, :, :object_slots].float()
    semantic_affinity = F.cosine_similarity(
        semantic[:, 1:], semantic[:, :-1], dim=-1, eps=1e-6
    )[..., None]
    return torch.cat(
        (
            dynamic[:, :-1],
            dynamic[:, 1:] - dynamic[:, :-1],
            center[:, 1:] - center[:, :-1],
            scale[:, 1:] - scale[:, :-1],
            (presence[:, 1:] - presence[:, :-1])[..., None],
            (visibility[:, 1:] - visibility[:, :-1])[..., None],
            semantic_affinity,
        ),
        dim=-1,
    )


class VideoEffectPosterior(nn.Module):
    """Explain an observed object-tube trajectory with a continuous effect."""

    def __init__(self, config: PredictiveObjectTubeConfig):
        super().__init__()
        self.config = config
        input_dim = config.dynamic_dim * 2 + 7
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.model_dim),
            nn.GELU(approximate="tanh"),
        )
        self.object_identity = nn.Parameter(
            torch.randn(config.object_slots, config.model_dim) / config.model_dim**0.5
        )
        self.time = nn.Sequential(
            nn.Linear(2, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            config.model_dim,
            config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.readout = _EffectReadout(config)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        transition = _transition_features(state, self.config.object_slots)
        batch, steps, objects = transition.shape[:3]
        if steps < 1:
            raise ValueError("effect posterior requires at least two object states")
        position = torch.linspace(0.0, 1.0, steps, device=transition.device)
        time = self.time(torch.stack((position, position.square()), dim=-1))
        hidden = self.input(transition)
        hidden = hidden + self.object_identity[None, None] + time[None, :, None]
        hidden = self.encoder(hidden.reshape(batch, steps * objects, -1))
        return self.readout(hidden)


class ImageGoalEffectPredictor(nn.Module):
    """Select an effect from current and desired image-encoded Object Sets."""

    def __init__(self, config: PredictiveObjectTubeConfig):
        super().__init__()
        self.config = config
        input_dim = config.semantic_dim * 2 + config.dynamic_dim * 2 + 8
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.model_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.object_identity = nn.Parameter(
            torch.randn(config.object_slots, config.model_dim) / config.model_dim**0.5
        )
        layer = nn.TransformerEncoderLayer(
            config.model_dim,
            config.heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.readout = _EffectReadout(config)

    def forward(
        self,
        current: dict[str, torch.Tensor],
        goal: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        count = self.config.object_slots
        current_semantic = current["semantic"][:, :count].float()
        goal_semantic = goal["semantic"][:, :count].float()
        current_dynamic = current["dynamic"][:, :count].float()
        goal_dynamic = goal["dynamic"][:, :count].float()
        geometry = torch.cat(
            (
                goal["center"][:, :count].float()
                - current["center"][:, :count].float(),
                goal["log_scale"][:, :count].float()
                - current["log_scale"][:, :count].float(),
                current["presence"][:, :count, None].float(),
                goal["presence"][:, :count, None].float(),
                current["visibility"][:, :count, None].float(),
                goal["visibility"][:, :count, None].float(),
            ),
            dim=-1,
        )
        inputs = torch.cat(
            (
                current_semantic,
                goal_semantic,
                current_dynamic,
                goal_dynamic - current_dynamic,
                geometry,
            ),
            dim=-1,
        )
        hidden = self.encoder(self.input(inputs) + self.object_identity[None])
        return self.readout(hidden)
