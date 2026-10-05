"""Query-equivariant continuous effect expert; denoising tau is not video time."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vlm_conditioner_v70 import parameter_counts


class EffectAttentionV70(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, query, context, context_valid=None):
        b, length, width = query.shape
        q = self.q(query).reshape(b, length, self.heads, -1).transpose(1, 2)
        k = self.k(context).reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        v = self.v(context).reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        mask = None if context_valid is None else context_valid[:, None, None].bool()
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(output.transpose(1, 2).reshape(b, length, width))


class EffectExpertBlockV70(nn.Module):
    """Local 4->144 history attention, VLM attention, global 64-token attention."""

    def __init__(self, width=1024, heads=16, ffn_dim=4096):
        super().__init__()
        self.local_norm = nn.LayerNorm(width)
        self.local_attention = EffectAttentionV70(width, heads)
        self.vlm_norm = nn.LayerNorm(width)
        self.vlm_attention = EffectAttentionV70(width, heads)
        self.global_norm = nn.LayerNorm(width)
        self.global_attention = EffectAttentionV70(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, ffn_dim, bias=False)
        self.up = nn.Linear(width, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, width, bias=False)

    def forward(self, x, history_tokens, vlm_tokens, vlm_valid, query_valid):
        b, k, roles, width = x.shape
        valid = query_valid[:, :, None, None]
        local = self.local_norm(x).reshape(b * k, roles, width)
        x = x + self.local_attention(
            local, history_tokens.reshape(b * k, -1, width)
        ).reshape(b, k, roles, width)
        x = x.masked_fill(~valid, 0)
        flat = x.reshape(b, k * roles, width)
        flat = flat + self.vlm_attention(self.vlm_norm(flat), vlm_tokens, vlm_valid)
        token_valid = query_valid[:, :, None].expand(-1, -1, roles).reshape(b, k * roles)
        flat = flat.masked_fill(~token_valid[..., None], 0)
        flat = flat + self.global_attention(self.global_norm(flat), self.global_norm(flat), token_valid)
        norm = self.ffn_norm(flat)
        flat = flat + self.down(F.silu(self.gate(norm)) * self.up(norm))
        return flat.reshape(b, k, roles, width).masked_fill(~valid, 0)


class ContinuousEffectExpertV70(nn.Module):
    """Four role embeddings shared by all queries; no query-index embeddings.

    Defaults: 12 blocks, width 1024, 16 heads, SwiGLU intermediate 4096.
    History tokens [B,T,K,9,512], centers [B,T,K,9,2] are the frozen
    parent's native-normalized coordinates; times [B,T] are real seconds.
    ``encode_condition`` is differentiable and is called inside model.forward
    in training. Its projected tensors can be cached for inference.
    """

    def __init__(self, num_blocks=12, width=1024, heads=16, ffn_dim=4096,
                 activation_checkpointing=True):
        super().__init__()
        self.width = width
        self.activation_checkpointing = activation_checkpointing
        self.state_adapter = nn.Linear(512, width)
        self.center_adapter = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.video_time_adapter = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.vlm_adapter = nn.Linear(2560, width)
        self.effect_input = nn.Linear(64, width)
        self.role_embeddings = nn.Parameter(torch.randn(4, width) * 0.02)
        # Integer indices survive FSDP buffer casting; compute frequencies in FP32.
        self.register_buffer("tau_indices", torch.arange(64), persistent=False)
        self.tau_adapter = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            EffectExpertBlockV70(width, heads, ffn_dim) for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(width)
        self.effect_output = nn.Linear(width, 64)

    def encode_condition(self, vlm_condition, history):
        dtype = self.state_adapter.weight.dtype
        valid = history["query_valid"].bool()
        state_valid = valid[:, None, :, None, None]
        tokens = history["tokens"].to(dtype).masked_fill(~state_valid, 0)
        centers = history["centers"].to(dtype).masked_fill(~state_valid, 0)
        times = (history["times"] - history["times"][:, -1:]).to(dtype)
        state = (self.state_adapter(tokens) + self.center_adapter(centers)
                 + self.video_time_adapter(times[..., None])[:, :, None, None])
        b, t, k, local, width = state.shape
        state = state.permute(0, 2, 1, 3, 4).reshape(b, k, t * local, width)
        state = state.masked_fill(~valid[:, :, None, None], 0)
        vlm_valid = vlm_condition["attention_mask"].bool()
        vlm_tokens = vlm_condition["last_hidden_state"].to(dtype).masked_fill(~vlm_valid[..., None], 0)
        return {"history_tokens": state, "vlm_tokens": self.vlm_adapter(vlm_tokens),
                "vlm_attention_mask": vlm_valid, "query_valid": valid}

    def forward(self, u, tau, condition):
        b = u.shape[0]
        tau = tau.to(device=u.device, dtype=torch.float32).reshape(-1).expand(b)
        frequencies = torch.exp(-math.log(10000) * self.tau_indices.float() / 64)
        angles = tau[:, None] * frequencies[None] * 1000
        tau_features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        tau_embedding = self.tau_adapter(tau_features.to(self.effect_input.weight.dtype))
        x = (self.effect_input(u.to(self.effect_input.weight.dtype))
             + self.role_embeddings[None, None] + tau_embedding[:, None, None])
        x = x.masked_fill(~condition["query_valid"][:, :, None, None], 0)
        args = (condition["history_tokens"], condition["vlm_tokens"],
                condition["vlm_attention_mask"], condition["query_valid"])
        for block in self.blocks:
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, *args, use_reentrant=False)
            else:
                x = block(x, *args)
        output = self.effect_output(self.output_norm(x))
        return output.masked_fill(~condition["query_valid"][:, :, None, None], 0)

    def parameter_inventory(self):
        return {"expert": parameter_counts(self), "effect_blocks": parameter_counts(self.blocks)}
