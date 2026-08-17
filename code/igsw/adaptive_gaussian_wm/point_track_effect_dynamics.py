"""Continuous latent effects and effect-conditioned object-state dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stable_normalization import stable_rms_normalize
from .point_track_object_state import normalize_support_shape


def state_tokens(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            state["identity"],
            state["dynamic"],
            state["center"].float(),
            state["log_scale"].float()[..., None],
            state["support_shape"].float(),
            state["presence"].float()[..., None],
            state["visibility"].float()[..., None],
        ),
        dim=-1,
    )


class PointTrackEffectPosterior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pair_input = nn.Sequential(
            nn.LayerNorm(2 * config.state_token_dim),
            nn.Linear(2 * config.state_token_dim, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.queries = nn.Parameter(torch.empty(1, config.effect_factors, config.state_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            config.state_dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.effect_dim),
        )

    def forward(self, source, target):
        context = self.pair_input(torch.cat((state_tokens(source), state_tokens(target)), dim=-1))
        query = self.queries.expand(len(context), -1, -1)
        attended, _ = self.cross_attention(query, context, context, need_weights=False)
        return torch.tanh(self.output(query + attended))


class PointTrackEffectDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_input = nn.Linear(config.state_token_dim, config.state_dim)
        self.effect_input = nn.Linear(config.effect_factors * config.effect_dim, config.state_dim)
        self.time_input = nn.Sequential(
            nn.Linear(5, config.state_dim), nn.SiLU(), nn.Linear(config.state_dim, config.state_dim)
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
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.dynamics_layers)
        update_dim = config.dynamic_dim + 2 + 1 + config.support_shape_dim + 2
        self.output = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, update_dim),
        )

    def forward(self, source, effect, delta_time):
        if effect.shape[1:] != (self.config.effect_factors, self.config.effect_dim):
            raise ValueError("v51 latent effect shape differs")
        dt = delta_time.float().clamp_min(0.0)
        time = torch.stack(
            (
                torch.log1p(dt), torch.sin(dt), torch.cos(dt),
                torch.sin(0.25 * dt), torch.cos(0.25 * dt),
            ),
            dim=-1,
        )
        hidden = self.state_input(state_tokens(source))
        hidden = hidden + self.effect_input(effect.flatten(1))[:, None]
        hidden = hidden + self.time_input(time)[:, None]
        update = self.output(self.transformer(hidden))
        dynamic, center, scale, shape, presence, visibility = update.split(
            (self.config.dynamic_dim, 2, 1, 3, 1, 1), dim=-1
        )
        gate = torch.tanh(effect.float().square().mean(dim=(1, 2)).sqrt())[:, None, None]
        scalar_gate = gate.squeeze(-1)
        dynamic_candidate = stable_rms_normalize(
            source["dynamic"].float() + 0.5 * torch.tanh(dynamic.float())
        )
        shape_direction = F.normalize(
            source["support_shape"][..., 1:].float() + 0.25 * torch.tanh(shape[..., 1:]),
            dim=-1,
            eps=1e-4,
        )
        shape_candidate = torch.cat(
            (
                (source["support_shape"][..., :1].float() + 0.25 * torch.tanh(shape[..., :1])).clamp(0.0, 2.0),
                shape_direction,
            ),
            dim=-1,
        )
        return {
            "identity": source["identity"],
            "dynamic": torch.lerp(
                source["dynamic"].float(), dynamic_candidate.float(), gate.float()
            ).to(source["dynamic"].dtype),
            "center": torch.lerp(
                source["center"].float(),
                (source["center"].float() + 0.5 * torch.tanh(center.float())).clamp(-1.25, 1.25),
                gate.float(),
            ),
            "log_scale": torch.lerp(
                source["log_scale"].float(),
                (source["log_scale"].float() + 0.25 * torch.tanh(scale.squeeze(-1))).clamp(-3.0, 0.7),
                scalar_gate.float(),
            ),
            "support_shape": normalize_support_shape(
                torch.lerp(
                    source["support_shape"].float(),
                    shape_candidate.float(),
                    gate.float(),
                )
            ),
            "presence": torch.lerp(
                source["presence"].float(),
                (source["presence"].float() + 0.25 * torch.tanh(presence.squeeze(-1))).clamp(0.0, 1.0),
                scalar_gate.float(),
            ),
            "visibility": torch.lerp(
                source["visibility"].float(),
                (source["visibility"].float() + 0.25 * torch.tanh(visibility.squeeze(-1))).clamp(0.0, 1.0),
                scalar_gate.float(),
            ),
        }
