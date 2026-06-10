"""GaussianDynamics — language-conditioned per-Gaussian delta-field transformer (v2).

Scaled to ~1B+ and conditioned on Qwen3-VL features PER LAYER: block j cross-attends
to the (already-projected) token features of Qwen3-VL transformer layer j. The
per-layer 2048->d projections and the VLM itself live in `InstructGSWorldModel`
(igsw/model_full.py); this module receives ready per-block context tensors so it
can be scaled and sharded independently.

Identity-at-init preserved (zero-init delta head + AdaLN-Zero blocks) so the model
starts as G_{t+1}=G_t and learns residual motion — essential for stable rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..gaussians.types import GaussianSet, inverse_sigmoid
from .tokenizer import GaussianTokenizer
from .transformer import DiTBlock, TimestepEmbed
from .manifold import axis_angle_to_quat, quat_to_rotmat


@dataclass
class DynamicsConfig:
    d_model: int = 1536          # ~1.7B at n_layers=28
    n_heads: int = 16
    n_layers: int = 28           # 1:1 with Qwen3-VL's 28 transformer layers
    num_freqs: int = 10
    feature_dim: int = 0
    lang_dim: int = 2048         # Qwen3-VL-2B hidden size (context dim per block)
    mlp_ratio: float = 4.0
    max_disp: float = 0.1        # per-step translation bound (gauge units)
    max_rot: float = 0.3         # per-step rotation bound (rad)
    use_checkpoint: bool = True   # grad-checkpoint blocks (needed at >1B)
    checkpoint_every: int = 1     # 1=checkpoint all; 2=every other (faster, more mem); 0=none


@dataclass
class GaussianState:
    """Batched raw Gaussian tensors ([B,N,*]); the model's working representation."""
    means: torch.Tensor          # [B,N,3]
    quats: torch.Tensor          # [B,N,4]
    scales: torch.Tensor         # [B,N,3] >0
    opacities: torch.Tensor      # [B,N] in (0,1)
    colors: torch.Tensor         # [B,N,3]
    features: torch.Tensor | None = None  # [B,N,D]

    @classmethod
    def from_gaussianset(cls, gs: GaussianSet) -> "GaussianState":
        f = gs.features[None] if gs.features is not None else None
        return cls(gs.means[None], gs.quats[None], gs.scales[None],
                   gs.opacities[None], gs.colors[None], f)

    def index_batch(self, b: int) -> GaussianSet:
        f = self.features[b] if self.features is not None else None
        return GaussianSet(self.means[b], self.quats[b], self.scales[b],
                           self.opacities[b], self.colors[b], f)


class GaussianDynamics(nn.Module):
    def __init__(self, cfg: DynamicsConfig, entity_head: bool = False):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tokenizer = GaussianTokenizer(d, num_freqs=cfg.num_freqs, feature_dim=cfg.feature_dim)
        self.tstep = TimestepEmbed(d)
        self.blocks = nn.ModuleList(
            [DiTBlock(d, cfg.n_heads, ctx_dim=d, mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        out_dim = 3 + 3 + 3 + 1 + 3 + cfg.feature_dim  # v, omega, dlog_s, dlogit_o, dcolor, dfeat
        self.head = nn.Linear(d, out_dim)
        nn.init.zeros_(self.head.weight)   # identity start
        nn.init.zeros_(self.head.bias)
        self._out_dim = out_dim
        # §54 ENTITY-SLOT SE(3) head (built ONLY when on, so v8-lang has no unused params under DDP).
        # Pools the per-control DiT feature (d) + pooled relevance feature (ent_rd) per seg entity ->
        # one rigid (v_e, omega_e) per entity per step. Zero-init last layer => v_e=omega_e=0 =>
        # rigid term = 0 => the per-control head is the WHOLE motion = exact legacy behavior at init.
        self.ent_rd = 256
        self.ent_mlp = None
        if entity_head:
            self.ent_mlp = nn.Sequential(nn.Linear(d + self.ent_rd, d), nn.SiLU(), nn.Linear(d, 6))
            nn.init.zeros_(self.ent_mlp[-1].weight); nn.init.zeros_(self.ent_mlp[-1].bias)
        self.last_resid = None

    def forward(self, state, ctx_per_block, ctx_mask, cond_global, step,
                cond_local=None, film_local=None, v_logit_local=None, gate_local=None,
                seg_local=None, rel_feat_local=None):
        return self.predict_deltas(state, ctx_per_block, ctx_mask, cond_global, step,
                                   cond_local=cond_local, film_local=film_local,
                                   v_logit_local=v_logit_local, gate_local=gate_local,
                                   seg_local=seg_local, rel_feat_local=rel_feat_local)

    def predict_deltas(self, state: GaussianState, ctx_per_block, ctx_mask, cond_global, step,
                       cond_local=None, film_local=None, v_logit_local=None, gate_local=None,
                       seg_local=None, rel_feat_local=None):
        """state: GaussianState[B,N,*]; ctx_per_block: [B,n_layers,L,d] (projected VLM
        features per block); ctx_mask: [B,L]; cond_global: [B,d]; step: [B].

        Per-control SPATIAL grounding (agent.md §37, opt-in):
          cond_local   [B,N,d]   per-control visual token, ADDED to each token (residual stream);
          film_local   [B,N,2d]  per-control (shift,scale), makes the AdaLN conditioning per-control
                                 (c becomes [B,N,d]) so modulation is no longer uniform across N.
          v_logit_local[B,N,3]   per-control velocity logit from the grounding feature, ADDED to the
                                 main head's velocity logits BEFORE the tanh bound. This is the
                                 load-bearing path: the resumed head projects per-control directions
                                 into its null space (diagnosed), so grounding needs this direct,
                                 un-projectable vote on which control moves.
          gate_local   [B,N,1]   per-control DYNAMICS GATE in [0,1] (Exp-1, --dyn_gate): the BOUNDED
                                 velocity/rotation are MULTIPLIED by it AFTER the tanh, so a control
                                 the gate calls "static" (gate→0) cannot move regardless of what the
                                 regression head emits (DynaSplat/DeGauss static-group near-identity).
                                 gate≈1 at init (warm-start) => identity, legacy behavior preserved.
        All default None (legacy global-only behavior)."""
        log_s = torch.log(state.scales.clamp_min(1e-8))
        logit_o = inverse_sigmoid(state.opacities.clamp(1e-6, 1 - 1e-6))
        x = self.tokenizer.tokenize_tensors(
            state.means, state.quats, log_s, logit_o, state.colors, state.features
        )                                                # [B,N,d]
        if cond_local is not None:
            x = x + cond_local                           # full-strength per-control visual signal
        c = cond_global + self.tstep(step)               # [B,d]
        if film_local is not None:
            # per-control FiLM -> c becomes [B,N,d]; DiTBlock handles the extra token dim.
            shift, scale = film_local.chunk(2, dim=-1)   # [B,N,d] each
            c = c[:, None, :] * (1.0 + scale) + shift    # [B,N,d]
        ce = self.cfg.checkpoint_every
        for j, blk in enumerate(self.blocks):
            ctx = ctx_per_block[:, j]                     # [B,L,d]
            do_ckpt = (self.cfg.use_checkpoint and self.training and ce > 0 and (j % ce == 0))
            if do_ckpt:
                x = torch.utils.checkpoint.checkpoint(blk, x, c, ctx, ctx_mask, use_reentrant=False)
            else:
                x = blk(x, c, ctx, ctx_mask)
        delta = self.head(self.final_norm(x))            # [B,N,out_dim]
        fdim = self.cfg.feature_dim
        v, omega, dlog_s, dlogit_o, dcolor = torch.split(delta[..., :13], [3, 3, 3, 1, 3], dim=-1)
        dfeat = delta[..., 13:] if fdim > 0 else None
        if v_logit_local is not None:
            v = v + v_logit_local                        # grounding's direct per-control velocity vote
        if self.cfg.max_disp > 0:
            v = self.cfg.max_disp * torch.tanh(v)        # bounded per-control RESIDUAL (when entity head on)
        if self.cfg.max_rot > 0:
            omega = self.cfg.max_rot * torch.tanh(omega)
        # §54 ENTITY-SLOT SE(3): pool the DiT features by seg entity -> one rigid (v_e, omega_e) per
        # entity, broadcast as a rigid transform about the entity's CURRENT centroid; the per-control
        # head above becomes a small residual. Zero-init => v_e=omega_e=0 => rigid term = 0 => v stays
        # the residual = exact legacy motion at init. Rigidity is then a STRUCTURAL guarantee, not a soft
        # loss (an entity's controls share one SE(3) by construction -> no intra-object spread possible).
        self.last_resid = v.new_zeros(())
        if seg_local is not None and self.ent_mlp is not None:
            means0 = state.means[0]                                   # [N,3] current control positions
            uniq, inv = torch.unique(seg_local.view(-1), return_inverse=True)
            E = uniq.numel()
            ones = x.new_ones(x.shape[1], 1)
            cnt = torch.zeros(E, 1, device=x.device, dtype=x.dtype).index_add_(0, inv, ones)        # [E,1]
            cen = (torch.zeros(E, 3, device=x.device, dtype=means0.dtype)
                   .index_add_(0, inv, means0) / cnt.clamp_min(1).to(means0.dtype))                 # [E,3] centroid
            xfe = (torch.zeros(E, x.shape[-1], device=x.device, dtype=x.dtype)
                   .index_add_(0, inv, x[0]) / cnt.clamp_min(1))                                     # [E,d] pooled DiT
            if rel_feat_local is not None:
                afe = (torch.zeros(E, rel_feat_local.shape[-1], device=x.device, dtype=x.dtype)
                       .index_add_(0, inv, rel_feat_local.to(x.dtype)) / cnt.clamp_min(1))           # [E,rd]
            else:
                afe = x.new_zeros(E, self.ent_rd)
            ent = self.ent_mlp(torch.cat([xfe, afe], dim=-1))        # [E,6]
            ve = self.cfg.max_disp * torch.tanh(ent[:, :3])          # [E,3] entity translation
            we = self.cfg.max_rot * torch.tanh(ent[:, 3:6])          # [E,3] entity rotation (axis-angle)
            Ri = quat_to_rotmat(axis_angle_to_quat(we))[inv]         # [N,3,3]
            ci = cen[inv]                                            # [N,3]
            rigid = torch.einsum("nij,nj->ni", Ri, means0 - ci) + ci + ve[inv] - means0   # [N,3]
            self.last_resid = v[0].norm(dim=-1).mean()               # residual magnitude (w_resid target)
            v = rigid[None].to(v.dtype) + v                          # entity rigid + small per-control residual
            omega = we[inv][None].to(omega.dtype) + omega
        if gate_local is not None:
            # Exp-1 STATIC GATE: scale the (already tanh-BOUNDED) velocity/rotation by the per-control
            # dynamics probability p_dyn∈[0,1]. Applied AFTER the bound so the tanh discipline (§31) is
            # untouched and this is exactly v_gated = p_dyn·v_bounded — a static control (gate→0) emits
            # ~0 motion, which the rollout accumulates to ~0. gate≈1 at warm-start => no-op at init.
            v = v * gate_local                           # [B,N,1] broadcasts over the 3 xyz channels
            omega = omega * gate_local
        return v, omega, dlog_s, dlogit_o, dcolor, dfeat

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
