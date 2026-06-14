"""GPSToken-JEPA world model — PLAN v1 (agent.md §93-95, PLAN_GPSTOKEN_JEPA_zh.md).

Sparse 2D-Gaussian tokens (lifted to 3D) -> OUR DiT capacity (Qwen frozen conditioning + DiTBlock stack,
kept per user: "保留我们的大容量骨干") -> heads:
  * geom head  : per-token 3D translation (LOAD-BEARING). E1 two variants, selectable:
      - "xyz"   : direct 3D displacement Δxyz.
      - "flowd" : 2D image flow (Δu,Δv) + log-depth change Δlogz, unprojected to 3D (PLAN §5: "本质是
                  2D 预测 + 一维深度"). The two are an A/B (PLAN §5 E1).
  * content head: future token feature (JEPA auxiliary — shapes dynamics-aware features for VLA, NOT motion).
  * relevance  : token feature · instruction-text  (grounding; trained with counterfactual in the trainer).
Rotation is NOT a head — it emerges from the per-token translation field (Δx=(R−I)(x−c)) and is read out
by Kabsch at eval (PLAN §3.1, §8: immune to the 6 failed rotation architectures).

Single-window (frame0 -> frameK) prediction (JEPA convention, de-risks rollout). The model exposes
encode_cond / encode_tokens / predict / geom_to_xyz / relevance; the trainer orchestrates data + losses.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dynamics.conditioning import QwenVLEncoder
from ..dynamics.transformer import DiTBlock, CrossAttention
from ..dynamics.pe import FourierPE3D


class GPSTokenWM(nn.Module):
    def __init__(self, qwen_path: str | None = None, d_model: int = 1536, n_heads: int = 16,
                 n_query: int = 16, fdim: int = 128, geom_mode: str = "xyz", num_freqs: int = 10):
        super().__init__()
        self.encoder = QwenVLEncoder(qwen_path) if qwen_path else QwenVLEncoder()
        H = self.encoder.hidden_size
        n_l = self.encoder.num_layers
        d = d_model
        self.d, self.fdim, self.geom_mode, self.n_query = d, fdim, geom_mode, n_query
        # ---- conditioning (faithfully copied from model_full.encode: per-layer proj + query aggregator) ----
        self.layer_proj = nn.ModuleList([nn.Linear(H, d) for _ in range(n_l)])
        self.cond_proj = nn.Linear(H, d)
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.layer_id_emb = nn.Parameter(torch.randn(n_l, d) * 0.02)
        self.aggregator = CrossAttention(d, n_heads, ctx_dim=d)
        self.agg_norm = nn.LayerNorm(d, eps=1e-6)
        # ---- token embed: FourierPE(3D pos) ⊕ frozen feat ⊕ sigma -> d ----
        self.pe = FourierPE3D(num_freqs=num_freqs)
        self.feat_in = nn.Linear(H, fdim)                      # frozen Qwen patch feat -> token feature ch.
        tok_in = self.pe.out_dim + fdim + 2
        self.tok_embed = nn.Sequential(nn.Linear(tok_in, d), nn.SiLU(), nn.Linear(d, d))
        # ---- DiT predictor (the kept 1.66B capacity) ----
        self.blocks = nn.ModuleList([DiTBlock(d, n_heads, ctx_dim=d) for _ in range(n_l)])
        self.final_norm = nn.LayerNorm(d, eps=1e-6)
        # ---- heads ----
        self.geom_head = nn.Linear(d, 3)
        nn.init.zeros_(self.geom_head.weight); nn.init.zeros_(self.geom_head.bias)   # start = no motion
        self.content_head = nn.Linear(d, fdim)
        self.rel_proj = nn.Sequential(nn.Linear(fdim, d), nn.SiLU(), nn.Linear(d, H))
        self.rel_temp = 0.07

    # ---------------- conditioning ----------------
    def encode_cond(self, vlm_inputs: dict):
        """-> ctx_per_block [1,n_l,Q,d], ctx_mask [1,Q], cond_global [1,d], text_feats [L_t,H]."""
        enc = self.encoder(vlm_inputs)
        hidden_all, valid_mask, text_mask = enc if len(enc) == 3 else (enc[0], enc[1], enc[1])
        n_l = hidden_all.shape[0]
        ctx_full = torch.stack([self.layer_proj[j](hidden_all[j].float()) for j in range(n_l)], 0)
        tw = text_mask.float()[:, None]
        pooled = (hidden_all[-1].float() * tw).sum(0) / tw.sum().clamp_min(1e-6)
        cond_global = self.cond_proj(pooled)[None]
        text_feats = hidden_all[-1][text_mask].detach().float()
        agg = []
        for j in range(n_l):
            q = (self.query + self.layer_id_emb[j])[None]
            a = self.aggregator(q, ctx_full[j][None], valid_mask[None])[0]
            agg.append(self.agg_norm(a))
        ctx_per_block = torch.stack(agg, 0)[None]
        ctx_mask = torch.ones(1, self.n_query, dtype=torch.bool, device=hidden_all.device)
        return ctx_per_block, ctx_mask, cond_global, text_feats

    # ---------------- predictor ----------------
    def predict(self, tok_xyz0, tok_feat_fdim, tok_sigma, center, radius,
                ctx_per_block, ctx_mask, cond_global):
        """tokens (frame0) -> per-token hidden x [1,M,d]. tok_feat_fdim [M,fdim] = self.feat_in(grid feat),
        computed ONCE by the caller and reused for SIGReg/relevance/JEPA-target."""
        norm_xyz = (tok_xyz0 - center) / radius
        pe = self.pe(norm_xyz[None])[0]                                       # [M, pe_out]
        # fp32 cat (pe/sigma are fp32, feat is bf16 under autocast) -> tok_embed Linear re-casts
        x = torch.cat([pe.float(), tok_feat_fdim.float(), tok_sigma.float()], dim=-1)[None]
        x = self.tok_embed(x)
        for j, blk in enumerate(self.blocks):
            x = blk(x, cond_global, ctx_per_block[:, j], ctx_mask)
        return self.final_norm(x)                                            # [1,M,d]

    def heads(self, x, tok_xyz0, K_intr, viewmat):
        """x [1,M,d] -> predicted future xyz [M,3], predicted future feature [M,fdim]."""
        h = x[0]                                                             # [M,d]
        g = self.geom_head(h).float()                                       # [M,3] fp32 for camera math
        xyz1 = self.geom_to_xyz(g, tok_xyz0.float(), K_intr.float(), viewmat.float())
        feat_pred = self.content_head(h)                                    # [M,fdim]
        return xyz1, feat_pred

    def geom_to_xyz(self, g, tok_xyz0, K_intr, viewmat):
        if self.geom_mode == "xyz":
            return tok_xyz0 + g                                             # direct 3D displacement
        # ---- flowd: (Δu, Δv, Δlogz) in image space -> unproject to world 3D ----
        ones = torch.ones(tok_xyz0.shape[0], 1, device=tok_xyz0.device, dtype=tok_xyz0.dtype)
        cam0 = (torch.cat([tok_xyz0, ones], -1) @ viewmat.T)[:, :3]
        z0 = cam0[:, 2:3].clamp_min(1e-4)
        fx, fy, cx, cy = K_intr[0, 0], K_intr[1, 1], K_intr[0, 2], K_intr[1, 2]
        u0 = cam0[:, 0:1] / z0 * fx + cx
        v0 = cam0[:, 1:2] / z0 * fy + cy
        u1, v1 = u0 + g[:, 0:1], v0 + g[:, 1:2]
        z1 = z0 * torch.exp(g[:, 2:3].clamp(-2.0, 2.0))
        xc = (u1 - cx) / fx * z1
        yc = (v1 - cy) / fy * z1
        cam1 = torch.cat([xc, yc, z1], -1)
        inv = torch.linalg.inv(viewmat.float()).to(cam1.dtype)
        return (torch.cat([cam1, ones], -1) @ inv.T)[:, :3]

    def relevance(self, tok_feat, text_emb):
        """tok_feat [M,fdim], text_emb [H] -> relevance logits [M]."""
        q = F.normalize(self.rel_proj(tok_feat), dim=-1)
        t = F.normalize(text_emb, dim=-1)[None]
        return (q * t).sum(-1) / self.rel_temp

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
