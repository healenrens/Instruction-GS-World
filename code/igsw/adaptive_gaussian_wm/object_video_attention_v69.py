"""Pre-norm attention blocks without dtype-dependent MHA inference fast paths."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def position_features(value, bands=4):
    frequency = 2.0 ** torch.arange(bands, device=value.device, dtype=torch.float32) * math.pi
    phase = value.float()[..., None] * frequency
    return torch.cat((value.float(), phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


def masked_probability_v69(logits, valid):
    masked = logits.float().masked_fill(~valid, -torch.inf)
    maximum = masked.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    mass = (masked-maximum).exp()
    return mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)


class AttentionV69(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, query, context, bias=None, return_attention=False):
        def heads(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.q(query)), heads(self.k(context)), heads(self.v(context))
        mask = None if bias is None else bias[:, None].to(q.dtype)
        if return_attention or torch.are_deterministic_algorithms_enabled():
            logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            if bias is not None:
                logits = logits + bias[:, None].float()
            attention = logits.softmax(-1)
            value = torch.matmul(attention.to(v.dtype), v)
        else:
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attention = None
        value = self.out(value.transpose(1, 2).flatten(-2))
        return value, attention.mean(1) if return_attention else None


class AttentionBlockV69(nn.Module):
    def __init__(self, width, heads, cross=False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width) if cross else None
        self.attention = AttentionV69(width, heads)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))

    def forward(self, value, context=None, bias=None, return_attention=False):
        normalized = self.norm(value.float())
        context = normalized if context is None else self.context_norm(context.float())
        update, attention = self.attention(normalized, context, bias, return_attention)
        value = value + update
        return value + self.ffn(value.float()), attention
