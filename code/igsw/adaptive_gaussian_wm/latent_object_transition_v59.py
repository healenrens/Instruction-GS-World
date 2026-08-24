"""Latent-effect posterior and effect-conditioned single-object Dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributed_statistics import roll_batch_with_grad
from .query_persistent_state_v58 import QueryPersistentStateEncoder


@dataclass(frozen=True)
class ObjectTransitionPrediction:
    semantic: torch.Tensor
    geometry: torch.Tensor
    visibility_logits: torch.Tensor


class TimeCondition(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(4, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, delta_seconds: torch.Tensor) -> torch.Tensor:
        scale = torch.log1p(delta_seconds.float())
        features = torch.stack(
            (scale, scale.square(), torch.sin(scale), torch.cos(scale)), dim=-1
        )
        return self.project(features)


class ObjectTargetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.semantic = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.model_dim),
        )
        self.geometry = nn.Sequential(
            nn.LayerNorm(config.target_geometry_dim + 1),
            nn.Linear(config.target_geometry_dim + 1, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.output = nn.LayerNorm(config.model_dim)

    def forward(self, semantic, geometry, visibility):
        auxiliary = torch.cat((geometry.float(), visibility[..., None].float()), dim=-1)
        return self.output(self.semantic(semantic.float()) + self.geometry(auxiliary))


class ContinuousObjectEffectPosterior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target = ObjectTargetEncoder(config)
        self.time = TimeCondition(config.model_dim)
        self.posterior = nn.Sequential(
            nn.LayerNorm(4 * config.model_dim),
            nn.Linear(4 * config.model_dim, 2 * config.model_dim),
            nn.GELU(),
            nn.Linear(2 * config.model_dim, config.effect_factors * config.effect_dim),
        )

    def forward(self, target):
        source = self.target(
            target.source_semantic,
            target.source_geometry,
            target.source_visibility,
        )
        future = self.target(
            target.future_semantic,
            target.future_geometry,
            target.future_visibility,
        )
        source = source[:, None].expand_as(future)
        time = self.time(target.delta_seconds)
        features = torch.cat((source, future, future - source, time), dim=-1)
        effect = self.posterior(features)
        effect = effect.reshape(
            *effect.shape[:2], self.config.effect_factors, self.config.effect_dim
        )
        return torch.tanh(effect)


class TransitionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiheadAttention(
            config.model_dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(config.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, 4 * config.model_dim),
            nn.GELU(),
            nn.Linear(4 * config.model_dim, config.model_dim),
        )

    def forward(self, tokens):
        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class EffectConditionedObjectDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        geometry_dim = 2 + 3 + 1
        self.source_semantic = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.model_dim),
        )
        self.source_identity = nn.Sequential(
            nn.LayerNorm(config.identity_dim),
            nn.Linear(config.identity_dim, config.model_dim),
        )
        self.source_geometry = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.effect = nn.Linear(config.effect_dim, config.model_dim, bias=False)
        self.effect_position = nn.Parameter(
            torch.randn(config.effect_factors, config.model_dim) * 0.02
        )
        self.time = TimeCondition(config.model_dim)
        self.blocks = nn.ModuleList(
            TransitionBlock(config) for _ in range(config.transition_layers)
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.semantic = nn.Linear(config.model_dim, config.patch_dim)
        self.geometry = nn.Linear(config.model_dim, config.target_geometry_dim)
        self.visibility = nn.Linear(config.model_dim, 1)

    @staticmethod
    def _source_auxiliary(source):
        covariance = source.covariance[:, -1]
        return torch.stack(
            (
                source.center[:, -1, 0],
                source.center[:, -1, 1],
                covariance[:, 0, 0],
                covariance[:, 1, 1],
                covariance[:, 0, 1],
                source.visibility[:, -1],
            ),
            dim=-1,
        )

    def forward(self, source, effect, delta_seconds):
        batch, horizons = delta_seconds.shape
        source_token = (
            self.source_semantic(source.pooled_semantic[:, -1].float())
            + self.source_identity(source.identity.float())
            + self.source_geometry(self._source_auxiliary(source).float())
        )
        source_token = source_token[:, None].expand(batch, horizons, -1)
        effect_token = self.effect(effect.float()) + self.effect_position
        time_token = self.time(delta_seconds)
        tokens = torch.cat(
            (source_token[:, :, None], effect_token, time_token[:, :, None]), dim=2
        )
        flat = tokens.flatten(0, 1)
        for block in self.blocks:
            flat = block(flat)
        hidden = self.output_norm(flat[:, 0]).reshape(batch, horizons, -1)
        semantic = F.normalize(self.semantic(hidden), dim=-1, eps=1e-6)
        return ObjectTransitionPrediction(
            semantic=semantic,
            geometry=self.geometry(hidden).float(),
            visibility_logits=self.visibility(hidden).squeeze(-1).float(),
        )


class QueryObjectTransitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = QueryPersistentStateEncoder(config)
        self.encoder.requires_grad_(False)
        self.posterior = ContinuousObjectEffectPosterior(config)
        self.dynamics = EffectConditionedObjectDynamics(config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_source(self, patches, coordinates, valid, frame_times, query_coordinate):
        with torch.no_grad():
            return self.encoder(
                patches, coordinates, valid, frame_times, query_coordinate
            )

    def forward(
        self,
        patches,
        coordinates,
        valid,
        frame_times,
        query_coordinate,
        transition_target,
    ):
        source = self.encode_source(
            patches, coordinates, valid, frame_times, query_coordinate
        )
        effect = self.posterior(transition_target)
        correct = self.dynamics(source, effect, transition_target.delta_seconds)
        zero = self.dynamics(
            source, torch.zeros_like(effect), transition_target.delta_seconds
        )
        shuffled_effect = roll_batch_with_grad(effect.detach())
        shuffled = self.dynamics(
            source, shuffled_effect, transition_target.delta_seconds
        )
        return {
            "source": source,
            "effect": effect,
            "correct": correct,
            "zero": zero,
            "shuffled": shuffled,
        }
