"""Continuous posterior effect and function-space object Dynamics for v67."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_predictive_state_v67 import (
    DecodedPredictiveFieldV67,
    PredictiveObjectCodeV67,
    PredictiveObjectFieldDecoderV67,
    gaussian_rate_v67,
)


@dataclass(frozen=True)
class ContinuousObjectEffectV67:
    sample: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor
    rate: torch.Tensor


@dataclass(frozen=True)
class ObjectFieldOperatorOutputV67:
    code: PredictiveObjectCodeV67
    field: DecodedPredictiveFieldV67


def zero_effect_v67(effect: ContinuousObjectEffectV67) -> ContinuousObjectEffectV67:
    zeros = torch.zeros_like(effect.sample)
    return ContinuousObjectEffectV67(
        sample=zeros,
        mean=zeros,
        log_variance=torch.zeros_like(effect.log_variance),
        rate=torch.zeros_like(effect.rate),
    )


def shuffled_effect_v67(effect: ContinuousObjectEffectV67) -> ContinuousObjectEffectV67:
    if len(effect.sample) > 1:
        index = torch.roll(torch.arange(len(effect.sample), device=effect.sample.device), 1)
        def select(value):
            return value.index_select(0, index)
    else:
        def select(value):
            return value.roll(1, dims=1)
    return ContinuousObjectEffectV67(
        sample=select(effect.sample),
        mean=select(effect.mean),
        log_variance=select(effect.log_variance),
        rate=select(effect.rate),
    )


class ContinuousEffectPosteriorV67(nn.Module):
    """Explain an observed object-field transition without explicit motion labels."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.code_dim
        self.pair = nn.Sequential(
            nn.LayerNorm(width * 3 + 1),
            nn.Linear(width * 3 + 1, width * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(width * 2, width),
        )
        layer = nn.TransformerEncoderLayer(
            width,
            config.operator_heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, num_layers=2)
        self.posterior = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.effect_dim * 2),
        )
        self.composer = nn.Sequential(
            nn.LayerNorm(config.effect_dim * 2),
            nn.Linear(config.effect_dim * 2, config.effect_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.effect_dim * 2, config.effect_dim * 2),
        )

    def forward(
        self,
        source: PredictiveObjectCodeV67,
        target: PredictiveObjectCodeV67,
        delta_seconds: torch.Tensor,
        sample: bool = True,
    ) -> ContinuousObjectEffectV67:
        time = torch.log1p(delta_seconds.float())[:, None, None]
        time = time.expand(-1, source.mean.shape[1], -1)
        pair = torch.cat(
            (source.mean, target.mean, target.mean - source.mean, time), dim=-1
        )
        hidden = self.pair(pair.to(dtype=self.pair[1].weight.dtype))
        hidden = self.interaction(hidden)
        mean, log_variance = self.posterior(hidden).chunk(2, dim=-1)
        mean = mean.float()
        log_variance = log_variance.float().clamp(-6.0, 2.0)
        value = mean
        if sample:
            value = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        return ContinuousObjectEffectV67(
            sample=value,
            mean=mean,
            log_variance=log_variance,
            rate=gaussian_rate_v67(mean, log_variance),
        )

    def compose(
        self,
        first: ContinuousObjectEffectV67,
        second: ContinuousObjectEffectV67,
    ) -> ContinuousObjectEffectV67:
        encoded = self.composer(
            torch.cat((first.mean, second.mean), dim=-1).to(
                dtype=self.composer[1].weight.dtype
            )
        )
        mean, log_variance = encoded.chunk(2, dim=-1)
        mean = mean.float()
        log_variance = log_variance.float().clamp(-6.0, 2.0)
        return ContinuousObjectEffectV67(
            sample=mean,
            mean=mean,
            log_variance=log_variance,
            rate=gaussian_rate_v67(mean, log_variance),
        )


class ObjectFieldOperatorV67(nn.Module):
    """Neural operator mapping one continuous object state to another."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.code_dim
        self.effect = nn.Linear(config.effect_dim, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.update = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        layer = nn.TransformerEncoderLayer(
            dim,
            config.operator_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, config.operator_layers)
        self.identity_update = nn.Linear(dim, config.identity_dim)
        self.dynamic_update = nn.Linear(dim, config.dynamic_dim)
        self.log_variance_update = nn.Linear(dim, dim)
        self.decoder = PredictiveObjectFieldDecoderV67(config)

    def transition_code(
        self,
        source: PredictiveObjectCodeV67,
        effect: ContinuousObjectEffectV67,
        delta_seconds: torch.Tensor,
    ) -> PredictiveObjectCodeV67:
        source_value = source.mean.to(dtype=self.effect.weight.dtype)
        effect_value = self.effect(effect.sample.to(dtype=self.effect.weight.dtype))
        time = self.time(
            torch.log1p(delta_seconds.float())[:, None].to(dtype=self.time[0].weight.dtype)
        )[:, None]
        time = time.expand(-1, source_value.shape[1], -1)
        hidden = self.update(torch.cat((source_value, effect_value, time), dim=-1))
        hidden = self.interaction(hidden + source_value)
        identity = source.mean[..., : self.config.identity_dim]
        identity = identity + 0.10 * self.identity_update(hidden).float()
        dynamic = source.mean[..., self.config.identity_dim :]
        dynamic = dynamic + self.dynamic_update(hidden).float()
        mean = torch.cat((identity, dynamic), dim=-1)
        log_variance = (
            source.log_variance + 0.25 * torch.tanh(self.log_variance_update(hidden).float())
        ).clamp(-6.0, 2.0)
        return PredictiveObjectCodeV67(
            sample=mean,
            mean=mean,
            log_variance=log_variance,
            identity=F.normalize(identity, dim=-1, eps=1e-6),
            dynamic=dynamic,
            rate=gaussian_rate_v67(mean, log_variance),
            point_mean=source.point_mean,
            point_log_variance=source.point_log_variance,
            point_rate=source.point_rate,
            support_mass=source.support_mass,
        )

    def forward(
        self,
        source: PredictiveObjectCodeV67,
        effect: ContinuousObjectEffectV67,
        delta_seconds: torch.Tensor,
        anchor_coordinates: torch.Tensor,
        output_coordinates: torch.Tensor,
        output_scales: torch.Tensor,
    ) -> ObjectFieldOperatorOutputV67:
        code = self.transition_code(source, effect, delta_seconds)
        field = self.decoder(
            code.mean,
            anchor_coordinates,
            output_coordinates,
            output_scales,
        )
        return ObjectFieldOperatorOutputV67(code=code, field=field)
