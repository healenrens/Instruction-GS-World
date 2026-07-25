"""Conditioned transformer for Gaussian dynamics (DiT-style, AdaLN-Zero).

Each block:
    x = x + gate_sa  * SelfAttn ( modulate(LN(x), shift_sa, scale_sa) )
    x = x + gate_ca  * CrossAttn( modulate(LN(x), shift_ca, scale_ca), lang )
    x = x + gate_mlp * MLP     ( modulate(LN(x), shift_mlp, scale_mlp) )
where (shift,scale,gate)*3 come from an MLP on the global conditioning vector c
(pooled language + step embedding). Gates are zero-initialised (AdaLN-Zero) so a
fresh block is the identity — the model starts as G_{t+1}=G_t and learns the
residual motion, which is essential for stable autoregressive rollout (brief D).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    # shift/scale are [B,d] (global cond, broadcast over the N token axis) OR [B,N,d]
    # (per-control cond, already aligned with x [B,N,d] -> no unsqueeze). Per-control modulation
    # is how SPATIAL grounding (agent.md §37) breaks the uniform-across-N AdaLN.
    if shift.dim() == x.dim():
        return x * (1 + scale) + shift
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x [B,N,D]
        B, N, D = x.shape
        h = self.n_heads
        qkv = self.qkv(x).reshape(B, N, 3, h, D // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # [B,h,N,d]
        o = F.scaled_dot_product_attention(q, k, v)       # flash when available
        o = o.transpose(1, 2).reshape(B, N, D)
        return self.proj(o)


class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads, ctx_dim):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx, ctx_mask=None):  # x [B,N,D], ctx [B,L,ctx_dim]
        B, N, D = x.shape
        L = ctx.shape[1]
        h = self.n_heads
        q = self.q(x).reshape(B, N, h, D // h).transpose(1, 2)        # [B,h,N,d]
        kv = self.kv(ctx).reshape(B, L, 2, h, D // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                            # [B,h,L,d]
        attn_mask = None
        if ctx_mask is not None:
            # ctx_mask [B,L] True=keep -> additive mask [B,1,1,L]
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, N, D)
        return self.proj(o)


class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, ctx_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(dim, n_heads, ctx_dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # The language cross-attn (below) is ALWAYS-ON / UN-GATED (to prevent posterior collapse),
        # which made it the ONLY unbounded residual branch -> over 28 layers the residual stream &
        # its gradient grew until LayerNorm's backward went non-finite (the systematic NaN). Bound
        # the injection to ~unit scale with a param-free LayerNorm: language stays fully on (never
        # re-collapses) but can no longer blow up. elementwise_affine=False => no new state-dict keys.
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        # AdaLN-Zero: produce 9 modulation tensors from c
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, lang, lang_mask=None):
        (sa_sh, sa_sc, sa_g, ca_sh, ca_sc, ca_g, mlp_sh, mlp_sc, mlp_g) = self.ada(c).chunk(9, dim=-1)
        # Bound the AdaLN modulation. The `ada` weights grow unboundedly during training (measured
        # ~3x by s6000), amplifying activations/gradients across 28 layers until LayerNorm's BACKWARD
        # overflows -> the systematic NaN. tanh caps per-block scale & gate to (-1,1); since tanh(0)=0
        # this preserves the AdaLN-Zero init (identity at start) so resuming a checkpoint is seamless.
        sa_sc, ca_sc, mlp_sc = torch.tanh(sa_sc), torch.tanh(ca_sc), torch.tanh(mlp_sc)
        sa_g, mlp_g = torch.tanh(sa_g), torch.tanh(mlp_g)
        # gate is [B,d] (global -> broadcast over N) or [B,N,d] (per-control -> align with x).
        gate = (lambda g: g if g.dim() == x.dim() else g.unsqueeze(1))
        x = x + gate(sa_g) * self.attn(modulate(self.norm1(x), sa_sh, sa_sc))
        # Cross-attention to language is ALWAYS-ON (standard residual, NOT AdaLN-Zero-gated).
        # The previous ca_g≈0 gating kept language switched off so it could never carry gradient
        # (the model was provably instruction-insensitive: contrastive loss pinned at the margin).
        # Un-gating forces language to influence the output, so the main + contrastive losses can
        # actually learn to USE it. (ca_g is computed but unused, kept for state-dict shape.)
        x = x + self.norm_ca(self.cross(modulate(self.norm2(x), ca_sh, ca_sc), lang, lang_mask))
        x = x + gate(mlp_g) * self.mlp(modulate(self.norm3(x), mlp_sh, mlp_sc))
        return x


class TimestepEmbed(nn.Module):
    """Sinusoidal embedding of an integer step index -> dim."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):  # t [B] long/float
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(self.max_period, device=t.device)) *
                          torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)
