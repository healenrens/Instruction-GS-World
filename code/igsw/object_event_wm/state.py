"""Competitive observation binding and shared local feature rendering."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def coordinates(side, device):
    axis = torch.linspace(-1+1/side, 1-1/side, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), -1).flatten(0, 1)


def position_features(x):
    frequencies = x.new_tensor([1., 2., 4., 8.]) * math.pi
    angles = x[..., None] * frequencies
    return torch.cat((x, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)


@dataclass
class ObjectState:
    tokens: torch.Tensor
    centers: torch.Tensor
    time: torch.Tensor

    def detach(self):
        return ObjectState(self.tokens.detach(), self.centers.detach(), self.time.detach())


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*4),
                                        nn.GELU(), nn.Linear(width*4, width))

    def forward(self, x, context=None, context_valid=None):
        query = self.norm(x)
        keys = query if context is None else self.norm(context)
        padding = None if context_valid is None else ~context_valid
        x = x + self.attention(query, keys, keys, key_padding_mask=padding, need_weights=False)[0]
        return x + self.feedforward(x)


class ObjectStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, d))
        self.spatial = nn.Linear(18, d)
        self.initial_tokens = nn.Parameter(torch.randn(config.slots, config.local_tokens, d)/math.sqrt(d))
        self.queries = nn.Linear(d, d, bias=False)
        self.keys = nn.Linear(d, d, bias=False)
        self.values = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.update = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.memory = AttentionBlock(d, config.heads)
        self.output_norm = nn.LayerNorm(d)

    def initial(self, batch, time):
        tokens = self.initial_tokens[None].expand(batch, -1, -1, -1)
        centers = tokens.new_zeros((*tokens.shape[:-1], 2))
        return ObjectState(tokens, centers, time)

    def observe(self, state, features, xy, valid, time):
        b, k, r, d = state.tokens.shape
        content = self.input(features.float()) + self.spatial(position_features(xy))[None]
        keys, values = self.keys(content), self.values(content)
        tokens = state.tokens
        for _ in range(self.config.observation_iterations):
            queries = self.queries(self.norm(tokens))
            scores = torch.einsum("bkrd,bnd->bkrn", queries, keys).float()/math.sqrt(d)
            # Object competition, then local carriers divide the object's observations.
            ownership = scores.logsumexp(2).softmax(1) * valid[:, None].float()
            local = scores.softmax(2) * ownership[:, :, None]
            weights = local / local.sum(-1, keepdim=True).clamp_min(1e-8)
            evidence = torch.einsum("bkrn,bnd->bkrd", weights.to(values.dtype), values)
            joined = torch.cat((self.norm(tokens.float()), evidence.float()), -1)
            candidate = self.update(joined).float()
            gate = self.gate(joined).float()
            # Correct the persistent state instead of accumulating its own unbounded projection.
            tokens = (1-gate)*tokens.float() + gate*candidate
        centers = torch.einsum("bkrn,nd->bkrd", weights, xy.float())
        tokens = self.output_norm(self.memory(tokens.reshape(b*k, r, d))).reshape(b, k, r, d)
        observed = valid.any(-1)[:, None, None, None]
        return ObjectState(torch.where(observed, tokens, state.tokens),
                           torch.where(observed, centers, state.centers), time)

    def forward(self, features, xy, valid, times):
        state = self.initial(len(features), times[:, 0])
        states = []
        for frame in range(features.shape[1]):
            state = self.observe(state, features[:, frame], xy, valid[:, frame], times[:, frame])
            states.append(state)
        return states


class ObjectFeatureRenderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.local_radius = nn.Parameter(torch.tensor(-1.))
        self.field = nn.Sequential(nn.LayerNorm(d*2+18), nn.Linear(d*2+18, d),
                                   nn.GELU(), nn.Linear(d, config.feature_dim+1))

    def forward(self, state, xy):
        root = state.tokens[:, :, 0]
        center = state.centers[:, :, 0]
        relative = xy[None, None] - center[:, :, None]
        distance = (xy[None, None, :, None]-state.centers[:, :, None]).square().sum(-1)
        weights = (-distance/(F.softplus(self.local_radius.float()).square()+1e-4)).softmax(-1)
        local = torch.einsum("bknr,bkrd->bknd", weights.to(state.tokens.dtype), state.tokens)
        roots = root[:, :, None].expand(-1, -1, len(xy), -1)
        fields = self.field(torch.cat((roots.float(), local.float(), position_features(relative)), -1))
        features, logits = fields[..., :-1], fields[..., -1]
        masks = logits.float().softmax(1)
        reconstruction = (features.float()*masks[..., None]).sum(1)
        return reconstruction, masks
