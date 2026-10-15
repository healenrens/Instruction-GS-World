"""Matched global, object-step, and persistent object-effect transitions."""

import torch
from torch import nn
from torch.nn import functional as F

from .state import AttentionBlock, ObjectState, position_features


class EffectPosterior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.local = nn.Sequential(nn.LayerNorm(d*3+4), nn.Linear(d*3+4, d), nn.GELU())
        self.transition = AttentionBlock(d, config.heads)
        self.distribution = nn.Linear(d, config.effect_dim*2)
        self.boundary = (nn.Sequential(nn.Linear(d+config.effect_dim, d), nn.GELU(), nn.Linear(d, 1))
                         if config.effect_mode == "object_segment" else None)

    def forward(self, source, target, previous, first=False, deterministic=False):
        b, k, r, d = source.tokens.shape
        before, after = source.tokens, target.tokens
        geometry = torch.cat((source.centers, target.centers-source.centers), -1)
        hidden = self.local(torch.cat((before, after, after-before, geometry), -1).float())
        if self.config.effect_mode == "global_step":
            hidden = self.transition(source.tokens[:, :, 0], hidden.flatten(1, 2))
        else:
            hidden = self.transition(source.tokens[:, :, :1].reshape(b*k, 1, d),
                                     hidden.reshape(b*k, r, d)).reshape(b, k, d)
        mean, logvar = self.distribution(hidden).float().chunk(2, -1)
        logvar = logvar.clamp(-6, 2)
        candidate = mean if deterministic else mean + torch.randn_like(mean)*(.5*logvar).exp()
        probability = (self.boundary(torch.cat((hidden.float(), previous.float()), -1)).squeeze(-1).sigmoid()
                       if self.boundary is not None else mean.new_ones((b, k)))
        if self.config.effect_mode == "object_segment" and not first:
            hard = (probability > (.5 if deterministic else torch.rand_like(probability))).float()
            boundary = hard + probability - probability.detach()
        else:
            boundary = torch.ones_like(probability)
        value = previous + boundary[..., None]*(candidate-previous)
        kl = .5 * (mean.square() + logvar.exp() - logvar - 1).mean(-1)
        return {"value": value, "mean": mean, "logvar": logvar,
                "boundary": boundary, "boundary_probability": probability, "kl": kl}


class ObjectDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.effect = nn.Linear(config.effect_dim, d)
        self.position = nn.Linear(18, d)
        self.elapsed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.local = nn.ModuleList(AttentionBlock(d, config.heads) for _ in range(config.dynamics_layers))
        self.interaction = nn.ModuleList(AttentionBlock(d, config.heads) for _ in range(config.dynamics_layers))
        self.feature_delta = nn.Linear(d, d)
        self.center_delta = nn.Linear(d, 2)
        nn.init.normal_(self.feature_delta.weight, std=1e-3)
        nn.init.zeros_(self.feature_delta.bias)
        nn.init.normal_(self.center_delta.weight, std=1e-3)
        nn.init.zeros_(self.center_delta.bias)

    def forward(self, source, effect, time):
        b, k, r, d = source.tokens.shape
        dt = time-source.time
        value = source.tokens + self.position(position_features(source.centers))
        value = value + self.elapsed(dt[:, None])[:, None, None]
        effect = self.effect(effect.float())
        for local, interaction in zip(self.local, self.interaction):
            if self.config.effect_mode == "global_step":
                value = local(value.flatten(1, 2), effect).reshape(b, k, r, d)
            else:
                value = local(value.reshape(b*k, r, d), effect.reshape(b*k, 1, d)).reshape(b, k, r, d)
            value = interaction(value.flatten(1, 2)).reshape(b, k, r, d)
        tokens = source.tokens + dt[:, None, None, None]*self.feature_delta(value.float())
        centers = source.centers + dt[:, None, None, None]*self.center_delta(value.float())
        return ObjectState(tokens, centers, time)
