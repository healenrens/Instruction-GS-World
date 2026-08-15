"""Posterior latent effects and effect-conditioned object-state dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from .stable_normalization import stable_rms_normalize
from .v49_config import TrajectoryObjectStateConfig


def _state_tokens(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            state["identity"],
            state["dynamic"],
            state["center"].float(),
            state["log_scale"].float()[..., None],
            state["presence"].float()[..., None],
            state["visibility"].float()[..., None],
        ),
        dim=-1,
    )


class TrajectoryEffectPosterior(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        self.config = config
        token_dim = config.state_dim + 5
        self.pair_input = nn.Sequential(
            nn.LayerNorm(2 * token_dim),
            nn.Linear(2 * token_dim, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.effect_queries = nn.Parameter(
            torch.empty(1, config.effect_factors, config.state_dim)
        )
        nn.init.normal_(self.effect_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            config.state_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.effect_dim),
        )

    def forward(
        self,
        source: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pair = torch.cat((_state_tokens(source), _state_tokens(target)), dim=-1)
        context = self.pair_input(pair)
        query = self.effect_queries.expand(len(pair), -1, -1)
        attended, _ = self.cross_attention(
            query, context, context, need_weights=False
        )
        return torch.tanh(self.output(query + attended))


class EffectConditionedObjectDynamics(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        self.config = config
        token_dim = config.state_dim + 5
        self.state_input = nn.Linear(token_dim, config.state_dim)
        self.effect_input = nn.Linear(
            config.effect_factors * config.effect_dim, config.state_dim
        )
        self.time_input = nn.Sequential(
            nn.Linear(5, config.state_dim),
            nn.SiLU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.state_dim,
            nhead=config.heads,
            dim_feedforward=4 * config.state_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.dynamics_layers
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, config.dynamic_dim + 5),
        )

    def forward(
        self,
        source: dict[str, torch.Tensor],
        effect: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if effect.shape[1:] != (
            self.config.effect_factors,
            self.config.effect_dim,
        ):
            raise ValueError("v49 latent effect shape differs")
        dt = delta_time.float().clamp_min(0.0)
        time = torch.stack(
            (
                torch.log1p(dt),
                torch.sin(dt),
                torch.cos(dt),
                torch.sin(0.25 * dt),
                torch.cos(0.25 * dt),
            ),
            dim=-1,
        )
        hidden = self.state_input(_state_tokens(source))
        hidden = hidden + self.effect_input(effect.flatten(1))[:, None]
        hidden = hidden + self.time_input(time)[:, None]
        hidden = self.transformer(hidden)
        update = self.output(hidden)
        dynamic, center, log_scale, presence, visibility = update.split(
            (self.config.dynamic_dim, 2, 1, 1, 1), dim=-1
        )
        gate = torch.tanh(effect.float().square().mean(dim=(1, 2)).sqrt())
        gate = gate[:, None, None]
        predicted_dynamic = stable_rms_normalize(
            source["dynamic"].float() + gate * 0.5 * torch.tanh(dynamic.float())
        )
        return {
            "identity": source["identity"],
            "dynamic": predicted_dynamic.to(source["dynamic"].dtype),
            "center": (
                source["center"].float() + gate * 0.5 * torch.tanh(center.float())
            ).clamp(-1.25, 1.25),
            "log_scale": (
                source["log_scale"].float()
                + gate.squeeze(-1) * 0.25 * torch.tanh(log_scale.squeeze(-1).float())
            ).clamp(-3.0, 0.7),
            "presence": (
                source["presence"].float()
                + gate.squeeze(-1) * 0.25 * torch.tanh(presence.squeeze(-1).float())
            ).clamp(0.0, 1.0),
            "visibility": (
                source["visibility"].float()
                + gate.squeeze(-1) * 0.25 * torch.tanh(visibility.squeeze(-1).float())
            ).clamp(0.0, 1.0),
        }

