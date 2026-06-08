"""InstructGSWorldModel — the full ≥1B language-conditioned Gaussian world model.

Conditioning (point 1): Qwen3-VL is FROZEN (preserves its physical-world/language
knowledge). Learnable "spatial-aggregation" query tokens + a trainable per-layer
aggregator read EVERY frozen Qwen layer's token features and distill them into a
compact set of special tokens; dynamics block j cross-attends to the special
tokens distilled from Qwen layer j. (Set n_query=0 to instead cross-attend to all
raw text tokens.)

Pipeline (one forward = one training step's dynamics compute):
  (instruction+frame0) --Qwen3-VL frozen--> all-28-layer hidden [28,L,2048]
       --per-layer Linear--> [28,L,d] --query aggregator--> special tokens [28,Q,d]
  G0 --SC-GS control + LBS--> ; for k in 1..K: block j attends to layer-j special
       tokens --> per-control deltas --> advance control + deform dense (LBS)
  --> stacked dense states over K (rendered + supervised by the trainer)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gaussians.types import GaussianSet
from .dynamics.model import GaussianDynamics, DynamicsConfig, GaussianState
from .dynamics.scgs import SCGSRollout
from .dynamics.conditioning import QwenVLEncoder
from .dynamics.transformer import CrossAttention


class InstructGSWorldModel(nn.Module):
    def __init__(
        self,
        dyn_cfg: DynamicsConfig | None = None,
        qwen_path: str = "/mnt/pfs/public/xuhaoming/model_zoo/Cosmos-Reason2-2B",
        n_control: int = 2048,
        lbs_k: int = 4,
        n_query: int = 16,          # learnable spatial-aggregation special tokens (0 = use all tokens)
        agg_heads: int = 8,
        action_dim: int = 0,        # >0 enables action-conditioning (per-step EEF action -> cond)
        cond_mode: str = "aggregator",  # 'aggregator' (default A/B baseline) | 'metaquery'
        spatial_ground: bool = False,   # per-control SPATIAL visual grounding (agent.md §37)
    ):
        super().__init__()
        self.encoder = QwenVLEncoder(qwen_path)   # frozen feature extractor (no LoRA)
        dyn_cfg = dyn_cfg or DynamicsConfig()
        assert dyn_cfg.n_layers == self.encoder.num_layers, (
            f"n_layers ({dyn_cfg.n_layers}) must equal VLM layers ({self.encoder.num_layers})")
        d = dyn_cfg.d_model
        n_l = self.encoder.num_layers
        self.layer_proj = nn.ModuleList([nn.Linear(self.encoder.hidden_size, d) for _ in range(n_l)])
        self.cond_proj = nn.Linear(self.encoder.hidden_size, d)
        self.n_query = n_query
        assert cond_mode in ("aggregator", "metaquery"), f"bad cond_mode {cond_mode!r}"
        self.cond_mode = cond_mode
        if cond_mode == "metaquery":
            # MetaQuery: N learnable query tokens appended INTO the frozen Qwen forward;
            # Qwen's own 28 layers process them -> per-layer query hidden [28,N,2048] is
            # projected through the SHARED self.layer_proj (no aggregator/query/layer_id_emb).
            # This is the ONLY new trainable param for conditioning (~0.1M).
            assert n_query > 0, "metaquery needs n_query > 0 (number of appended query tokens)"
            self.meta_query = nn.Parameter(torch.randn(n_query, self.encoder.hidden_size) * 0.02)  # [N,2048]
        elif n_query > 0:
            self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
            self.layer_id_emb = nn.Parameter(torch.randn(n_l, d) * 0.02)
            self.aggregator = CrossAttention(d, agg_heads, ctx_dim=d)   # shared across layers
            self.agg_norm = nn.LayerNorm(d, eps=1e-6)
        self.dynamics = GaussianDynamics(dyn_cfg)
        self.n_control = n_control
        self.lbs_k = lbs_k
        # ---- per-control SPATIAL visual grounding (agent.md §37): each control samples the
        # frozen-Qwen IMAGE-patch feature at its frame-0 (u,v) -> a per-control visual feature
        # that is injected STRONGLY into the dynamics (so the output can vary per control and
        # break the global-AdaLN uniformity that made motion uniform). Two injection sites:
        #   (1) vis_tok: project H->d and ADD to each control token (full-strength residual);
        #   (2) vis_film: project H->2d -> per-control (shift,scale) FiLM ADDED to the global
        #       AdaLN conditioning c, making AdaLN per-control instead of uniform.
        # Both start at ZERO -> identity warm-start (a resumed aggregator ckpt is unchanged).
        self.spatial_ground = bool(spatial_ground)
        if self.spatial_ground:
            H = self.encoder.hidden_size
            self.vis_norm = nn.LayerNorm(H, eps=1e-6)
            self.vis_tok = nn.Sequential(nn.Linear(H, d), nn.SiLU(), nn.Linear(d, d))
            self.vis_film = nn.Linear(H, 2 * d)
            # vis_tok / vis_film inject the per-control visual feature into the DiT residual stream
            # and AdaLN. NOTE (diagnosed): the RESUMED dynamics head was trained to emit a UNIFORM
            # global translation, so it projects per-control directions into its null space — even
            # with pre-head per-control std ~1.7 the output std was ~2e-4. So these two paths alone
            # cannot break uniformity against a pre-trained head.
            # => vis_vhead: a DEDICATED per-control motion head straight from the grounding feature.
            # Its [M,3] output is ADDED to the main head's velocity logits (then the same tanh bound),
            # giving grounding an UN-projectable vote on which control moves. This is the load-bearing
            # path. A small per-control MLP -> the magnitude/direction stay LEARNED (a real predictor,
            # not a hard-wired copy). Zero-init last layer -> exact identity warm-start; it gets
            # gradient from step 1 because it is downstream of the (already-trained) loss, not the head.
            self.vis_vhead = nn.Sequential(nn.Linear(H, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(),
                                           nn.Linear(d, 3))
            nn.init.zeros_(self.vis_vhead[-1].weight); nn.init.zeros_(self.vis_vhead[-1].bias)
            nn.init.zeros_(self.vis_film.weight); nn.init.zeros_(self.vis_film.bias)
            # vis_tok / vis_film also zero-init -> exact identity warm-start; vis_vhead (zero-init,
            # but a DIRECT residual on the velocity logits, not gated by the trained head/AdaLN) is
            # the path that actually learns to localize. All start at identity; gradients bootstrap
            # them. (Non-zero vis_tok was tried and destabilized into a large uniform translation.)
            nn.init.zeros_(self.vis_tok[-1].weight); nn.init.zeros_(self.vis_tok[-1].bias)
        # ---- InfoNCE language-forcing heads (research_F): align PREDICTED motion with the
        # instruction so the dynamics is forced to USE language (non-saturating, vs the old hinge).
        proj = 256
        self.proj_dim = proj
        self.motion_enc = nn.Sequential(nn.Linear(6, 256), nn.GELU(), nn.Linear(256, 256))   # per-control
        self.motion_head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, proj))
        self.lang_proj = nn.Sequential(nn.Linear(self.encoder.hidden_size, 512), nn.GELU(), nn.Linear(512, proj))
        self.action_dim = action_dim
        if action_dim > 0:
            self.action_embed = nn.Sequential(
                nn.Linear(action_dim, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.action_embed[-1].weight)   # start OFF -> safe warm-start, grows in training
            nn.init.zeros_(self.action_embed[-1].bias)

    def encode(self, vlm_inputs: dict):
        if self.cond_mode == "metaquery":
            return self._encode_metaquery(vlm_inputs)
        # hidden_all [n_l,L,H]; valid_mask = all tokens; text_mask = instruction tokens only
        enc = self.encoder(vlm_inputs)
        hidden_all, valid_mask, text_mask = enc if len(enc) == 3 else (enc[0], enc[1], enc[1])
        n_l = hidden_all.shape[0]
        ctx_full = torch.stack([self.layer_proj[j](hidden_all[j]) for j in range(n_l)], 0)  # [n_l,L,d]
        # GLOBAL cond pooled from the INSTRUCTION tokens only (avoids frame-0-image domination)
        tw = text_mask.float()[:, None]
        pooled = (hidden_all[-1] * tw).sum(0) / tw.sum().clamp_min(1e-6)
        cond_global = self.cond_proj(pooled)[None]             # [1,d]
        if self.n_query > 0:
            # learnable query tokens aggregate each layer's image+text features (trainable head);
            # the dynamics then cross-attends to these -> learned language-conditioned visual grounding
            agg = []
            for j in range(n_l):
                q = (self.query + self.layer_id_emb[j])[None]                    # [1,Q,d]
                a = self.aggregator(q, ctx_full[j][None], valid_mask[None])[0]   # [Q,d]
                agg.append(self.agg_norm(a))
            ctx_per_block = torch.stack(agg, 0)[None]          # [1,n_l,Q,d]
            ctx_mask = torch.ones(1, self.n_query, dtype=torch.bool, device=hidden_all.device)
        else:
            ctx_per_block = ctx_full[None]                     # [1,n_l,L,d]
            ctx_mask = valid_mask[None]
        return ctx_per_block, ctx_mask, cond_global, pooled

    def _encode_metaquery(self, vlm_inputs: dict):
        """MetaQuery encode (research_G): append self.meta_query into the frozen Qwen
        forward, read per-layer query hidden [n_l,N,2048], project each through the SHARED
        self.layer_proj -> ctx_per_block [1,n_l,N,d]. Returns the same 4-tuple as encode().
        Qwen runs WITH grad here (frozen params get none); grad reaches only meta_query.
        Requires an image (vision injection + image-grid M-RoPE) -> run with --vlm_image 1."""
        if "pixel_values" not in vlm_inputs or "image_grid_thw" not in vlm_inputs:
            raise ValueError("cond_mode='metaquery' requires image inputs (pixel_values + "
                             "image_grid_thw); launch training with --vlm_image 1.")
        qp = next(self.encoder.parameters())
        query_embeds = self.meta_query.to(device=qp.device, dtype=qp.dtype)     # [N,2048]
        query_hidden = self.encoder.forward_metaquery(vlm_inputs, query_embeds)  # [n_l,N,2048]
        n_l = query_hidden.shape[0]
        ctx_per_block = torch.stack(
            [self.layer_proj[j](query_hidden[j]) for j in range(n_l)], dim=0
        ).unsqueeze(0)                                          # [1,n_l,N,d]
        ctx_mask = torch.ones(1, self.n_query, dtype=torch.bool, device=query_hidden.device)
        # GLOBAL cond + InfoNCE pooled vector: mean of the LAST-layer query hidden states
        # over the N query positions (instruction intent distilled by Qwen into the queries).
        pooled = query_hidden[-1].mean(0)                       # [2048]
        cond_global = self.cond_proj(pooled)[None]             # [1,d]
        return ctx_per_block, ctx_mask, cond_global, pooled

    def _control_visual(self, vlm_inputs: dict, control_uv: torch.Tensor, uv_hw, d_device, d_dtype):
        """Sample a per-control visual feature from the frozen-Qwen IMAGE grid at each control's
        frame-0 (u,v). control_uv [M,2] is (x=col, y=row) in the LIFTED image pixel space of size
        uv_hw=(H,W); grid_sample uses NORMALIZED [-1,1] coords so the (different) Qwen grid
        resolution is irrelevant. Returns (vis_tok [M,d], vis_film [M,2d]) or (None, None)."""
        grid, ghw = self.encoder.image_grid_features(vlm_inputs)
        if grid is None:
            return None, None
        gh, gw = ghw
        H, W = float(uv_hw[0]), float(uv_hw[1])
        # [gh,gw,Hc] -> [1,Hc,gh,gw] for grid_sample
        g = grid.to(device=d_device, dtype=torch.float32).permute(2, 0, 1)[None]   # [1,Hc,gh,gw]
        u = control_uv[:, 0].float() / max(W, 1.0) * 2.0 - 1.0      # x -> [-1,1]
        v = control_uv[:, 1].float() / max(H, 1.0) * 2.0 - 1.0      # y -> [-1,1]
        samp = torch.stack([u, v], dim=-1)[None, None]              # [1,1,M,2] (x,y order)
        import torch.nn.functional as F
        feat = F.grid_sample(g, samp, mode="bilinear", padding_mode="border",
                             align_corners=False)[0, :, 0].t()      # [M,Hc]
        feat = self.vis_norm(feat.to(d_dtype))                      # normalize Qwen-scale features
        return self.vis_tok(feat), self.vis_film(feat), self.vis_vhead(feat)   # [M,d], [M,2d], [M,3]

    def forward(self, vlm_inputs: dict, dense_g0: GaussianSet, K: int, start_step: int = 0,
                ctrl_idx=None, vlm_inputs_wrong: dict | None = None, actions=None,
                control_uv=None, control_uv_hw=None):
        ctx, m, cond_global, pooled_text = self.encode(vlm_inputs)
        # per-step action embedding (control input that causes the future)
        act_emb = self.action_embed(actions) if (self.action_dim > 0 and actions is not None) else None

        # ---- per-control SPATIAL visual grounding (agent.md §37) ----
        vis_tok = vis_film = vis_vlogit = None
        if self.spatial_ground and control_uv is not None:
            vt = self._control_visual(
                vlm_inputs, control_uv, control_uv_hw,
                cond_global.device, cond_global.dtype)
            if vt[0] is not None:
                vis_tok = vt[0][None]                              # [1,M,d]
                vis_film = vt[1][None]                             # [1,M,2d]
                vis_vlogit = vt[2][None]                           # [1,M,3]

        def delta_fn(state: GaussianState, step_idx):
            cond = cond_global
            if act_emb is not None:
                k = int(step_idx[0].item()) - start_step
                if 0 <= k < act_emb.shape[0]:
                    cond = cond_global + act_emb[k][None]
            return self.dynamics(state, ctx, m, cond, step_idx,
                                 cond_local=vis_tok, film_local=vis_film, v_logit_local=vis_vlogit)

        roll = SCGSRollout(dense_g0, n_control=self.n_control, k=self.lbs_k, ctrl_idx=ctrl_idx)
        dense_states, deltas, ctrl_traj = roll.rollout(delta_fn, K, start_step=start_step)
        out = {
            "means": torch.stack([s.means for s in dense_states], 0),
            "quats": torch.stack([s.quats for s in dense_states], 0),
            "scales": torch.stack([s.scales for s in dense_states], 0),
            "opacities": torch.stack([s.opacities for s in dense_states], 0),
            "colors": torch.stack([s.colors for s in dense_states], 0),
            "v": torch.stack([d[0][0] for d in deltas], 0),
            "om": torch.stack([d[1][0] for d in deltas], 0),
            "dls": torch.stack([d[2][0] for d in deltas], 0),
            "ctrl": torch.stack(ctrl_traj, 0),
        }
        # ---- InfoNCE embeddings: a PREDICTED-motion embedding + the instruction embedding.
        # Aligning these (vs other clips' instructions, in the trainer) forces the dynamics to
        # make its motion instruction-specific (non-saturating MI lower bound; research_F #1).
        v = out["v"]                                           # [K,M,3]
        mfeat = torch.cat([v.mean(0), v.std(0)], dim=-1)      # [M,6] per-control motion stats
        mfeat = self.motion_enc(mfeat).mean(0)               # DeepSets pool over controls -> [256]
        out["motion_emb"] = torch.nn.functional.normalize(self.motion_head(mfeat), dim=-1)   # [proj]
        out["lang_emb"] = torch.nn.functional.normalize(self.lang_proj(pooled_text.float()), dim=-1)
        # Contrastive language-dependence: predict step-0 control velocity under a WRONG
        # instruction (one extra encode + one dynamics step, no rollout/render). The trainer
        # penalizes if the wrong instruction predicts the GT step-0 motion as well as the
        # correct one -> forces the model to actually USE the text.
        if vlm_inputs_wrong is not None:
            ctx_w, m_w, cg_w, _ = self.encode(vlm_inputs_wrong)
            cs0 = GaussianState.from_gaussianset(roll.control0)
            step0 = torch.zeros(1, dtype=torch.long, device=dense_g0.means.device) + start_step
            v_w, _, _, _, _, _ = self.dynamics(cs0, ctx_w, m_w, cg_w, step0)
            out["v_wrong0"] = v_w[0]                      # [M,3]
        return out

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_report(self) -> str:
        dyn = sum(p.numel() for p in self.dynamics.parameters())
        proj = sum(p.numel() for p in self.layer_proj.parameters()) + sum(p.numel() for p in self.cond_proj.parameters())
        agg = sum(p.numel() for n, p in self.named_parameters() if ("aggregator" in n or "query" in n or "layer_id" in n))
        vlm = sum(p.numel() for p in self.encoder.parameters())
        return (f"dynamics={dyn/1e6:.0f}M proj={proj/1e6:.0f}M agg={agg/1e6:.1f}M "
                f"| trainable={self.num_trainable()/1e9:.3f}B | Qwen3-VL FROZEN ({vlm/1e9:.2f}B)")
