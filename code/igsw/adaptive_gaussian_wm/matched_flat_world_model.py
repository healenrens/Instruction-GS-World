"""Capacity-matched unstructured DINO+RGB latent sequence baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .action_embedding import bounded_action_embedding
from .scale import ScaleModulatedBlock


CANONICAL_FEATURE_DIM = 3
CANONICAL_RGB_DIM = 3
CANONICAL_ACTION_DIM = CANONICAL_FEATURE_DIM + CANONICAL_RGB_DIM


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float().clamp(1e-4, 1.0 - 1e-4))


class MatchedFlatLatentWorldModel(nn.Module):
    """Perceiver-style baseline without object assignment or Gaussian readout."""

    def __init__(
        self,
        feature_dim: int,
        rgb_channels: int,
        model_dim: int,
        state_tokens: int,
        action_tokens: int,
        action_dim: int,
        action_residual_dim: int,
        dynamics_layers: int,
        heads: int,
        dropout: float = 0.0,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        positive = (
            feature_dim,
            rgb_channels,
            model_dim,
            state_tokens,
            action_tokens,
            action_dim,
            action_residual_dim,
            dynamics_layers,
            heads,
        )
        if min(positive) <= 0 or model_dim % heads:
            raise ValueError("matched flat architecture dimensions are invalid")
        if rgb_channels != 3:
            raise ValueError("matched flat baseline requires RGB channels")
        if action_dim != CANONICAL_ACTION_DIM + action_residual_dim:
            raise ValueError("matched flat action must use canonical 6D plus residual")
        self.feature_dim = feature_dim
        self.rgb_channels = rgb_channels
        self.model_dim = model_dim
        self.state_tokens = state_tokens
        self.action_tokens = action_tokens
        self.action_dim = action_dim
        self.action_residual_dim = action_residual_dim
        self.checkpoint_blocks = checkpoint_blocks

        self.history_input = nn.Linear(feature_dim + 3, model_dim)
        self.history_queries = nn.Parameter(
            torch.randn(state_tokens, model_dim) / model_dim**0.5
        )
        self.history_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.history_output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_dim * 2, model_dim),
        )

        self.posterior_input = nn.Linear(
            feature_dim * 2 + rgb_channels * 2 + 2,
            model_dim,
        )
        self.posterior_queries = nn.Parameter(
            torch.randn(action_tokens, model_dim) / model_dim**0.5
        )
        self.posterior_gap = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.posterior_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.posterior_output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_dim * 2, action_residual_dim),
            nn.LayerNorm(action_residual_dim),
        )
        row = torch.arange(feature_dim, dtype=torch.float32)[:, None] + 1
        column = torch.arange(CANONICAL_FEATURE_DIM, dtype=torch.float32)[None] + 1
        projection = torch.sin(row * column * 0.017)
        self.register_buffer(
            "feature_effect_projection",
            projection / projection.norm(dim=0, keepdim=True),
        )

        self.action_input = nn.Linear(CANONICAL_ACTION_DIM, model_dim, bias=False)
        self.residual_action_input = nn.Linear(
            action_residual_dim,
            model_dim,
            bias=False,
        )
        self.action_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gap_input = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.dynamics = nn.ModuleList(
            ScaleModulatedBlock(model_dim, heads, dropout)
            for _ in range(dynamics_layers)
        )

        self.readout_input = nn.Linear(
            feature_dim + rgb_channels + 3,
            model_dim,
        )
        self.readout_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_readout_output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_dim * 2, feature_dim),
        )
        self.rgb_readout_output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_dim, rgb_channels),
        )

    def encode_history(
        self,
        history_features: torch.Tensor,
        history_coordinates: torch.Tensor,
        history_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch, history_count, grid_count, feature_dim = history_features.shape
        if history_coordinates.shape != (batch, history_count, grid_count, 2):
            raise ValueError("matched flat history coordinates differ")
        if history_scale.shape != (batch, history_count):
            raise ValueError("matched flat history scale must have shape [B,H]")
        scale = history_scale[:, :, None, None].expand(
            -1,
            -1,
            grid_count,
            -1,
        )
        inputs = torch.cat(
            (history_features, history_coordinates, scale),
            dim=-1,
        ).reshape(batch, history_count * grid_count, feature_dim + 3)
        tokens = self.history_input(inputs)
        queries = self.history_queries[None].expand(batch, -1, -1)
        attended = self.history_attention(
            queries,
            tokens,
            tokens,
            need_weights=False,
        )[0]
        state = queries + attended
        return state + self.history_output(state)

    def posterior(
        self,
        current_features: torch.Tensor,
        future_features: torch.Tensor,
        current_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
        future_coordinates: torch.Tensor,
        future_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch, future_count, grid_count, feature_dim = future_features.shape
        if current_features.shape != (batch, grid_count, feature_dim):
            raise ValueError("matched flat posterior current feature grid differs")
        if current_rgb.shape != (batch, grid_count, self.rgb_channels):
            raise ValueError("matched flat posterior current RGB grid differs")
        if future_rgb.shape != (
            batch,
            future_count,
            grid_count,
            self.rgb_channels,
        ):
            raise ValueError("matched flat posterior future RGB grid differs")
        if future_coordinates.shape != (batch, future_count, grid_count, 2):
            raise ValueError("matched flat posterior coordinates differ")
        if future_scale.shape != (batch, future_count):
            raise ValueError("matched flat posterior scale must have shape [B,Q]")
        current_feature_query = current_features[:, None].expand(
            -1,
            future_count,
            -1,
            -1,
        )
        current_rgb_query = current_rgb[:, None].expand(
            -1,
            future_count,
            -1,
            -1,
        )
        tokens = self.posterior_input(
            torch.cat(
                (
                    current_feature_query,
                    future_features,
                    current_rgb_query,
                    future_rgb,
                    future_coordinates,
                ),
                dim=-1,
            )
        ).reshape(batch * future_count, grid_count, self.model_dim)
        queries = (
            self.posterior_queries[None, None]
            + self.posterior_gap(future_scale[..., None])[:, :, None]
        ).reshape(batch * future_count, self.action_tokens, self.model_dim)
        attended, attention = self.posterior_attention(
            queries,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        encoded = queries + attended
        residual = self.posterior_output(encoded)

        normalized_change = F.normalize(future_features.float(), dim=-1) - F.normalize(
            current_feature_query.float(),
            dim=-1,
        )
        feature_effect = normalized_change @ self.feature_effect_projection.float()
        rgb_effect = _stable_logit(future_rgb) - _stable_logit(current_rgb_query)
        attention_float = attention.float()
        canonical_feature = torch.tanh(
            attention_float
            @ feature_effect.detach().reshape(
                batch * future_count,
                grid_count,
                CANONICAL_FEATURE_DIM,
            )
            / 0.25
        )
        canonical_rgb = torch.tanh(
            attention_float
            @ rgb_effect.detach().reshape(
                batch * future_count,
                grid_count,
                CANONICAL_RGB_DIM,
            )
        )
        canonical = torch.cat((canonical_feature, canonical_rgb), dim=-1).to(
            residual.dtype
        )
        return torch.cat((canonical, residual), dim=-1).reshape(
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        )

    def predict_states(
        self,
        history_state: torch.Tensor,
        future_scale: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch, future_count = future_scale.shape
        if history_state.shape != (batch, self.state_tokens, self.model_dim):
            raise ValueError("matched flat history state layout differs")
        if actions.shape != (
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        ):
            raise ValueError("matched flat action layout differs")
        state = history_state[:, None].expand(-1, future_count, -1, -1).reshape(
            batch * future_count,
            self.state_tokens,
            self.model_dim,
        )
        canonical = bounded_action_embedding(
            self.action_input,
            actions[..., :CANONICAL_ACTION_DIM],
        )
        residual = bounded_action_embedding(
            self.residual_action_input,
            actions[..., CANONICAL_ACTION_DIM:],
        )
        action_tokens = (canonical + residual).reshape(
            batch * future_count,
            self.action_tokens,
            self.model_dim,
        )
        action_context = self.action_attention(
            state,
            action_tokens,
            action_tokens,
            need_weights=False,
        )[0]
        state = state + action_context + self.gap_input(
            future_scale[..., None]
        ).reshape(batch * future_count, 1, self.model_dim)
        scale = future_scale[:, :, None].expand(
            -1,
            -1,
            self.state_tokens,
        ).reshape(batch * future_count, self.state_tokens)
        for block in self.dynamics:
            if self.training and self.checkpoint_blocks:
                state = checkpoint(
                    lambda value, gap, layer=block: layer(value, gap),
                    state,
                    scale,
                    use_reentrant=False,
                )
            else:
                state = block(state, scale)
        return state.reshape(
            batch,
            future_count,
            self.state_tokens,
            self.model_dim,
        )

    def readout(
        self,
        current_features: torch.Tensor,
        current_rgb: torch.Tensor,
        future_coordinates: torch.Tensor,
        future_scale: torch.Tensor,
        future_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, future_count, grid_count = future_coordinates.shape[:3]
        if current_features.shape != (batch, grid_count, self.feature_dim):
            raise ValueError("matched flat readout current feature grid differs")
        if current_rgb.shape != (batch, grid_count, self.rgb_channels):
            raise ValueError("matched flat readout current RGB grid differs")
        current_feature_query = current_features[:, None].expand(
            -1,
            future_count,
            -1,
            -1,
        )
        current_rgb_query = current_rgb[:, None].expand(
            -1,
            future_count,
            -1,
            -1,
        )
        scale = future_scale[:, :, None, None].expand(
            -1,
            -1,
            grid_count,
            -1,
        )
        queries = self.readout_input(
            torch.cat(
                (
                    current_feature_query,
                    current_rgb_query,
                    future_coordinates,
                    scale,
                ),
                dim=-1,
            )
        ).reshape(batch * future_count, grid_count, self.model_dim)
        states = future_state.reshape(
            batch * future_count,
            self.state_tokens,
            self.model_dim,
        )
        attended = self.readout_attention(
            queries,
            states,
            states,
            need_weights=False,
        )[0]
        encoded = queries + attended
        feature_residual = self.feature_readout_output(encoded).reshape(
            batch,
            future_count,
            grid_count,
            self.feature_dim,
        )
        rgb_logit_residual = self.rgb_readout_output(encoded).reshape(
            batch,
            future_count,
            grid_count,
            self.rgb_channels,
        )
        feature_prediction = current_feature_query + feature_residual
        rgb_prediction = torch.sigmoid(
            _stable_logit(current_rgb_query) + rgb_logit_residual.float()
        )
        return feature_prediction, rgb_prediction

    def dynamics_readout(
        self,
        history_state: torch.Tensor,
        current_features: torch.Tensor,
        current_rgb: torch.Tensor,
        future_coordinates: torch.Tensor,
        future_scale: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        future_state = self.predict_states(history_state, future_scale, actions)
        return self.readout(
            current_features,
            current_rgb,
            future_coordinates,
            future_scale,
            future_state,
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_coordinates: torch.Tensor,
        history_scale: torch.Tensor,
        history_rgb: torch.Tensor,
        future_features: torch.Tensor,
        future_coordinates: torch.Tensor,
        future_scale: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_rgb.shape[:3] != history_features.shape[:3]:
            raise ValueError("matched flat history DINO and RGB grids differ")
        if future_rgb.shape[:3] != future_features.shape[:3]:
            raise ValueError("matched flat future DINO and RGB grids differ")
        history_state = self.encode_history(
            history_features,
            history_coordinates,
            history_scale,
        )
        actions = self.posterior(
            history_features[:, -1],
            future_features,
            history_rgb[:, -1],
            future_rgb,
            future_coordinates,
            future_scale,
        )
        posterior_feature, posterior_rgb = self.dynamics_readout(
            history_state,
            history_features[:, -1],
            history_rgb[:, -1],
            future_coordinates,
            future_scale,
            actions,
        )
        history_feature, history_rgb_prediction = self.dynamics_readout(
            history_state,
            history_features[:, -1],
            history_rgb[:, -1],
            future_coordinates,
            future_scale,
            torch.zeros_like(actions),
        )
        return {
            "history_feature_prediction": history_feature,
            "posterior_feature_prediction": posterior_feature,
            "history_rgb_grid_prediction": history_rgb_prediction,
            "posterior_rgb_grid_prediction": posterior_rgb,
            "posterior_actions": actions,
            "history_state": history_state,
        }
