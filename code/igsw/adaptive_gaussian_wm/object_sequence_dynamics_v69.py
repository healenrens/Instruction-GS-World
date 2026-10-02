"""Continuous plan effects and object-token sequence prediction, with no future images."""

import math
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .object_video_attention_v69 import AttentionBlockV69, position_features
from .query_object_video_encoder_v69 import ObjectVideoStateV69


class ObjectSequencePosteriorV69(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.posterior_hidden_width
        self.input_projection = nn.Identity() if d == config.width else nn.Linear(config.width, d)
        self.queries = nn.Parameter(torch.randn(config.effect_tokens, d) / math.sqrt(d))
        self.time = nn.Linear(9, d)
        self.geometry = None if config.architecture == "pretrained_query_object_video_sequence_v1" else nn.Linear(36, d)
        self.layers = nn.ModuleList(AttentionBlockV69(d, config.posterior_attention_heads, cross=True) for _ in range(config.posterior_layers))
        self.distribution = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.effect_dim*2))

    def forward(self, source, targets, deterministic=False, target_valid=None):
        b, k, r, _ = source.tokens.shape
        d = self.config.posterior_hidden_width
        # Read estimated state geometry, never tracker coordinates or measured flow.
        frames = [source, *targets]
        tokens = torch.stack([s.tokens for s in frames], 2)
        tokens = self.input_projection(tokens if d == self.config.width else tokens.float())
        centers = torch.stack([s.centers for s in frames], 2)
        root_delta = centers[:, :, :, :1] - source.centers[:, :, None, :1]
        local = centers - centers[:, :, :, :1]
        relative_geometry = torch.cat((root_delta.expand_as(local), local), -1)
        times = torch.stack([s.time-source.time for s in frames], 1)
        tokens = tokens + self.time(position_features(times[..., None]))[:, None, :, None]
        if self.geometry is not None:
            tokens = tokens + float(self.config.posterior_geometry) * self.geometry(position_features(relative_geometry))
        context = tokens.reshape(b*k, len(frames)*r, d)
        bias = None
        if target_valid is not None:
            frames_valid = torch.cat((target_valid.new_ones((b, 1)), target_valid), 1)
            keys_valid = frames_valid[:, None, :, None].expand(b, k, len(frames), r).reshape(b*k, -1)
            bias = tokens.new_zeros((b*k, self.config.effect_tokens, len(frames)*r)).masked_fill(~keys_valid[:, None], -torch.inf)
        source_root = source.tokens[:, :, 0]
        source_root = self.input_projection(source_root if d == self.config.width else source_root.float())
        hidden = self.queries[None].expand(b*k, -1, -1) + source_root.reshape(b*k, 1, d)
        for layer in self.layers:
            hidden, _ = layer(hidden, context, bias)
        mean, logvar = self.distribution(hidden.float()).chunk(2, -1)
        logvar = logvar.clamp(-6, 2)
        noise = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)
        sampled = mean + noise * (.5*logvar).exp()
        logvar_fp32 = logvar.float()
        kl = .5 * (mean.float().square() + (torch.expm1(logvar_fp32)-logvar_fp32))
        shape = (b, k, self.config.effect_tokens, self.config.effect_dim)
        return {"value": sampled.tanh().reshape(shape), "mean": mean.reshape(shape), "logvar": logvar.reshape(shape),
                "kl": kl.reshape(shape), "noise": noise.reshape(shape)}


class ObjectSequenceDynamicsV69(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.dynamics_hidden_width
        self.input_projection = nn.Identity() if d == config.width else nn.Linear(config.width, d)
        self.effect = nn.Linear(config.effect_dim, d)
        self.geometry = nn.Linear(18, d)
        self.time = nn.Sequential(nn.Linear(18, d), nn.SiLU(), nn.Linear(d, d))
        self.interaction = nn.ModuleList(AttentionBlockV69(d, config.dynamics_attention_heads) for _ in range(config.dynamics_layers))
        self.condition = nn.ModuleList(AttentionBlockV69(d, config.dynamics_attention_heads, cross=True) for _ in range(config.dynamics_layers))
        self.feature_delta = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.width))
        self.center_delta = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def condition_local(self, value, effect, block_index):
        """Query k reads only effect k before any cross-query interaction."""
        b, k, r, d = value.shape
        local_effect = self.effect(effect.float()).reshape(b*k, effect.shape[2], d)
        conditioned, _ = self.condition[block_index](value.reshape(b*k, r, d), local_effect)
        return conditioned.reshape(b, k, r, d)

    def interaction_block(self, value, effect, bias, block_index):
        b, k, r, d = value.shape
        value = self.condition_local(value, effect, block_index)
        flat, _ = self.interaction[block_index](value.flatten(1, 2), bias=bias)
        return flat.reshape(b, k, r, d)

    def advance(self, state, effect, time):
        b, k, r, _ = state.tokens.shape
        elapsed = time - state.time
        condition = self.time(position_features(torch.stack((time, elapsed), -1)))
        state_tokens = state.tokens if self.config.dynamics_hidden_width == self.config.width else state.tokens.float()
        value = self.input_projection(state_tokens) + self.geometry(position_features(state.centers)) + condition[:, None, None]
        object_mask = state.query_valid[:, :, None].expand(-1, -1, r).flatten(1)
        bias = torch.zeros((b, k*r, k*r), device=value.device).masked_fill(~object_mask[:, None], -torch.inf)
        for index in range(len(self.interaction)):
            block = partial(self.interaction_block, block_index=index)
            if self.config.dynamics_checkpoint_blocks and self.training and torch.is_grad_enabled():
                value = checkpoint(block, value, effect, bias, use_reentrant=False, preserve_rng_state=True)
            else:
                value = block(value, effect, bias)
        scale = elapsed[:, None, None, None]
        tokens = state.tokens + self.feature_delta(value.float()) * scale
        tokens = torch.cat((state.tokens[:, :, :1], tokens[:, :, 1:]), 2)
        centers = state.centers + self.center_delta(value.float()) * scale
        valid = state.query_valid[:, :, None, None]
        tokens = torch.where(valid, tokens, state.tokens)
        centers = torch.where(valid, centers, state.centers)
        return ObjectVideoStateV69(tokens, centers, time, state.query_valid)

    def forward(self, source, effect, times, rollout=True):
        state, states = source, []
        for time in times.unbind(1):
            state = self.advance(state if rollout else source, effect, time)
            states.append(state)
        return states
