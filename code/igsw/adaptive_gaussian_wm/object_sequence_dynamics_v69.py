"""Continuous plan effects and object-token sequence prediction, with no future images."""

import math

import torch
from torch import nn

from .object_video_attention_v69 import AttentionBlockV69, position_features
from .query_object_video_encoder_v69 import ObjectVideoStateV69


class ObjectSequencePosteriorV69(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(torch.randn(config.effect_tokens, config.width) / math.sqrt(config.width))
        self.time = nn.Linear(9, config.width)
        self.layers = nn.ModuleList(AttentionBlockV69(config.width, config.heads, cross=True) for _ in range(config.posterior_layers))
        self.distribution = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, config.effect_dim*2))

    def forward(self, source, targets, deterministic=False, target_valid=None):
        b, k, r, d = source.tokens.shape
        # Only latent tokens and elapsed time enter the posterior; no RGB or measured displacements.
        frames = [source, *targets]
        tokens = torch.stack([s.tokens for s in frames], 2)
        times = torch.stack([s.time-source.time for s in frames], 1)
        tokens = tokens + self.time(position_features(times[..., None]))[:, None, :, None]
        context = tokens.reshape(b*k, len(frames)*r, d)
        bias = None
        if target_valid is not None:
            frames_valid = torch.cat((target_valid.new_ones((b, 1)), target_valid), 1)
            keys_valid = frames_valid[:, None, :, None].expand(b, k, len(frames), r).reshape(b*k, -1)
            bias = tokens.new_zeros((b*k, self.config.effect_tokens, len(frames)*r)).masked_fill(~keys_valid[:, None], -torch.inf)
        hidden = self.queries[None].expand(b*k, -1, -1) + source.tokens[:, :, 0].reshape(b*k, 1, d)
        for layer in self.layers:
            hidden, _ = layer(hidden, context, bias)
        mean, logvar = self.distribution(hidden.float()).chunk(2, -1)
        logvar = logvar.clamp(-6, 2)
        noise = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)
        sampled = mean + noise * (.5*logvar).exp()
        kl = .5 * (mean.square() + logvar.exp() - logvar - 1)
        shape = (b, k, self.config.effect_tokens, self.config.effect_dim)
        return {"value": sampled.tanh().reshape(shape), "mean": mean.reshape(shape), "logvar": logvar.reshape(shape),
                "kl": kl.reshape(shape), "noise": noise.reshape(shape)}


class ObjectSequenceDynamicsV69(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.effect = nn.Linear(config.effect_dim, d)
        self.geometry = nn.Linear(18, d)
        self.time = nn.Sequential(nn.Linear(18, d), nn.SiLU(), nn.Linear(d, d))
        self.interaction = nn.ModuleList(AttentionBlockV69(d, config.heads) for _ in range(config.dynamics_layers))
        self.condition = nn.ModuleList(AttentionBlockV69(d, config.heads, cross=True) for _ in range(config.dynamics_layers))
        self.feature_delta = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.center_delta = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def advance(self, state, effect, time):
        b, k, r, d = state.tokens.shape
        elapsed = time - state.time
        condition = self.time(position_features(torch.stack((time, elapsed), -1)))
        value = state.tokens + self.geometry(position_features(state.centers)) + condition[:, None, None]
        # Preserve the effect-to-query association: a bag of untagged effects is permutation invariant.
        encoded_effect = (self.effect(effect.float()) + state.tokens[:, :, :1]).flatten(1, 2)
        object_mask = state.query_valid[:, :, None].expand(-1, -1, r).flatten(1)
        bias = torch.zeros((b, k*r, k*r), device=value.device).masked_fill(~object_mask[:, None], -torch.inf)
        effect_mask = state.query_valid[:, :, None].expand(-1, -1, effect.shape[2]).flatten(1)
        cross_bias = torch.zeros((b, k*r, effect_mask.shape[1]), device=value.device).masked_fill(~effect_mask[:, None], -torch.inf)
        for interaction, conditioning in zip(self.interaction, self.condition):
            flat, _ = interaction(value.flatten(1, 2), bias=bias)
            flat, _ = conditioning(flat, encoded_effect, cross_bias)
            value = flat.reshape(b, k, r, d)
        scale = elapsed[:, None, None, None]
        tokens = state.tokens + self.feature_delta(value.float()) * scale
        tokens = torch.cat((state.tokens[:, :, :1], tokens[:, :, 1:]), 2)
        centers = state.centers + self.center_delta(value.float()) * scale
        return ObjectVideoStateV69(tokens, centers, time, state.query_valid)

    def forward(self, source, effect, times, rollout=True):
        state, states = source, []
        for time in times.unbind(1):
            state = self.advance(state if rollout else source, effect, time)
            states.append(state)
        return states
