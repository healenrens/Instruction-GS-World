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
    def __init__(self, cfg: DynamicsConfig, entity_head: bool = False, rigid_agg: bool = False,
                 entity_rot: bool = False, motion_bases: int = 0, bases_mode: str = "pure",
                 detach_state_rot: bool = False):
        super().__init__()
        self.cfg = cfg
        # §66 v10-rigid: project per-control motion votes onto per-entity SE(3) (weighted Kabsch).
        # Parameter-free => no DDP/ckpt impact; identity on a rigid field => exact warm-start.
        self.rigid_agg = bool(rigid_agg)
        # §87 rotation redesign — two competing heads (mutually exclusive):
        #   entity_rot  (A+B+C): ROTATION ONLY goes entity-level. Attention readout (not mean-pool —
        #     the v8-ent direction killer) over the entity's control features -> 6D delta -> R_e about
        #     the entity centroid. Translation stays per-control (proven path, untouched).
        #   motion_bases (Shape-of-Motion, arXiv:2407.13764): B shared SE(3) bases per step (6D+t,
        #     learned-query attention readout over ALL control features) + per-control softmax
        #     coefficients; motion = blend in PARAMETER space (paper Sec. 3.1) then Gram-Schmidt.
        #     Deviation from paper: softmax coefficients (convex blend keeps GS well-conditioned)
        #     instead of L2-normalized — we are feed-forward, the paper is per-scene optimization.
        # detach_state_rot: stop-grad quats in the token input — cuts the 12-step recurrent gradient
        # chain (omega_t -> quat_{t+1} -> tokens -> DiT), the diagnosed explosion mechanism (§86).
        assert not (entity_rot and motion_bases > 0), "entity_rot and motion_bases are exclusive"
        self.entity_rot = bool(entity_rot)
        self.motion_bases = int(motion_bases)
        self.bases_mode = str(bases_mode)
        self.detach_state_rot = bool(detach_state_rot)
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
        # §87-A entity-rot head: scalar attention logit + value proj + 6D MLP (zero-init last layer
        # => R_e = I at init => exact warm-start; rotation contributes nothing until learned).
        if self.entity_rot:
            self.erot_attn = nn.Linear(d, 1)
            self.erot_val = nn.Linear(d, d)
            self.erot_mlp = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 6))
            nn.init.zeros_(self.erot_mlp[-1].weight); nn.init.zeros_(self.erot_mlp[-1].bias)
        # §87-B motion-bases head: B learned queries -> per-basis (6D rot delta, translation);
        # per-control coefficient head -> softmax blend. Zero-init basis MLP => identity motion at init.
        if self.motion_bases > 0:
            Bb = self.motion_bases
            self.basis_q = nn.Parameter(torch.randn(Bb, d) * 0.02)
            self.basis_val = nn.Linear(d, d)
            self.basis_mlp = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 9))
            nn.init.zeros_(self.basis_mlp[-1].weight); nn.init.zeros_(self.basis_mlp[-1].bias)
            self.coef_head = nn.Linear(d, Bb)
        self.erot_log = None                       # set to [] by the trainer before a rollout to record
        self.coef_last = None                      # last-step coefficients (aux seg-CE + logging)

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
        # §87 detach_state_rot: cut the recurrent rotation-gradient chain omega_t -> quat_{t+1} ->
        # token_{t+1} -> DiT (12-step unrolled recurrence = the diagnosed explosion, §86). Forward
        # identical; rotation gradients flow per-step only. Translation chain (means) untouched.
        toks_quats = state.quats.detach() if self.detach_state_rot else state.quats
        x = self.tokenizer.tokenize_tensors(
            state.means, toks_quats, log_s, logit_o, state.colors, state.features
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
        # §87-B MOTION BASES (Shape-of-Motion): blend B per-step SE(3) bases in PARAMETER space with
        # per-control softmax coefficients, Gram-Schmidt to SO(3). 'pure' REPLACES the per-control
        # motion (the low-rank hypothesis: rotation+rigidity become identifiable because every basis
        # is shared by many controls); 'residual' adds on top of the per-control field.
        if self.motion_bases > 0:
            from .rot6d import rot6d_delta_to_matrix, matrix_to_axis_angle
            xf = x[0]                                                 # [N,d] (B=1 rollout)
            att = torch.softmax((self.basis_q @ xf.T) / (xf.shape[-1] ** 0.5), dim=-1)  # [Bb,N]
            bas = att @ self.basis_val(xf)                            # [Bb,d] per-basis readout
            p = self.basis_mlp(bas)                                   # [Bb,9] zero-init
            d6_b = p[:, :6]
            t_b = self.cfg.max_disp * torch.tanh(p[:, 6:9])           # bounded basis translation
            coef = torch.softmax(self.coef_head(xf), dim=-1)          # [N,Bb] convex blend
            d6_i = coef @ d6_b                                        # blend in PARAMETER space (SoM)
            t_i = coef @ t_b
            R_i = rot6d_delta_to_matrix(d6_i)                         # [N,3,3], I at init
            means0b = state.means[0]
            cb = means0b.mean(0)
            v_bases = (torch.einsum("nij,nj->ni", R_i.to(means0b.dtype), means0b - cb)
                       + cb + t_i.to(means0b.dtype) - means0b)        # [N,3]
            om_bases = matrix_to_axis_angle(R_i)                      # [N,3]
            if self.bases_mode == "pure":
                v = v_bases[None].to(v.dtype)
                omega = om_bases[None].to(omega.dtype)
            else:                                                     # residual: add to per-control field
                v = v + v_bases[None].to(v.dtype)
                omega = omega + om_bases[None].to(omega.dtype)
            self.coef_last = coef
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
        # §87-A ENTITY-ROT: rotation (ONLY) goes entity-level — attention readout over the entity's
        # control features (NOT mean-pool: that erased direction in v8-ent) -> 6D delta -> R_e about
        # the entity centroid. Translation stays the per-control field above (the proven path).
        # Supervised in POSITION space by entity_rot_position_loss (trainer); R_e recorded per step.
        if self.entity_rot and seg_local is not None:
            from .rot6d import rot6d_delta_to_matrix, matrix_to_axis_angle
            means0r = state.means[0]
            uniq_r, inv_r = torch.unique(seg_local.view(-1), return_inverse=True)
            Er = uniq_r.numel()
            xr = x[0]                                                 # [N,d]
            # segment softmax attention: alpha_i = softmax over the i's entity members
            logit_a = self.erot_attn(xr).squeeze(-1).float()          # [N]
            lmax = torch.full((Er,), -1e30, device=xr.device).index_reduce_(0, inv_r, logit_a, "amax")
            ex = torch.exp(logit_a - lmax[inv_r])
            den = torch.zeros(Er, device=xr.device).index_add_(0, inv_r, ex).clamp_min(1e-12)
            alpha = (ex / den[inv_r]).to(xr.dtype)                    # [N], sums to 1 per entity
            val = self.erot_val(xr)                                   # [N,d]
            ro = torch.zeros(Er, val.shape[-1], device=xr.device, dtype=val.dtype)
            ro.index_add_(0, inv_r, alpha[:, None] * val)             # [Er,d] attention readout
            d6_e = self.erot_mlp(ro)                                  # [Er,6] zero-init
            R_e = rot6d_delta_to_matrix(d6_e)                         # [Er,3,3], I at init
            onesr = torch.ones(xr.shape[0], 1, device=xr.device)
            cntr = torch.zeros(Er, 1, device=xr.device).index_add_(0, inv_r, onesr)
            cenr = (torch.zeros(Er, 3, device=xr.device, dtype=means0r.dtype)
                    .index_add_(0, inv_r, means0r) / cntr.clamp_min(1).to(means0r.dtype))
            Ri_r = R_e[inv_r].to(means0r.dtype)
            rot_pos = (torch.einsum("nij,nj->ni", Ri_r, means0r - cenr[inv_r])
                       + cenr[inv_r] - means0r)                       # [N,3] rotation position term
            bg = (seg_local.view(-1) == 0)
            rot_pos = torch.where(bg[:, None], torch.zeros_like(rot_pos), rot_pos)
            om_e = matrix_to_axis_angle(R_e)[inv_r]
            om_e = torch.where(bg[:, None], torch.zeros_like(om_e), om_e)
            v = rot_pos[None].to(v.dtype) + v
            omega = om_e[None].to(omega.dtype) + omega
            if self.erot_log is not None:
                self.erot_log.append((R_e, uniq_r))                   # per-step record for the loss
        if gate_local is not None:
            # Exp-1 STATIC GATE: scale the (already tanh-BOUNDED) velocity/rotation by the per-control
            # dynamics probability p_dyn∈[0,1]. Applied AFTER the bound so the tanh discipline (§31) is
            # untouched and this is exactly v_gated = p_dyn·v_bounded — a static control (gate→0) emits
            # ~0 motion, which the rollout accumulates to ~0. gate≈1 at warm-start => no-op at init.
            v = v * gate_local                           # [B,N,1] broadcasts over the 3 xyz channels
            omega = omega * gate_local
        # §66 v10-rigid: per-entity weighted-Kabsch aggregation of the (gated) motion votes.
        # AFTER the gate so static entities vote ~0 -> fit ~= identity -> they stay static.
        if self.rigid_agg and seg_local is not None:
            from .rigid_agg import entity_rigid_aggregate
            wv = gate_local[0, :, 0] if gate_local is not None else None
            v, omega = entity_rigid_aggregate(state.means[0], v, omega, seg_local, w=wv)
        return v, omega, dlog_s, dlogit_o, dcolor, dfeat

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
