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
        dyn_gate: bool = False,         # Exp-1: per-control MOVER/STATIC gate on motion (default OFF, A/B)
        sem_dim: int = 0,               # Exp-1 #3 (optional): per-control object-semantic embedding dim (0 = off)
        gate_uses_sem: bool = True,     # §44h: feed the OCCLUSION-ROBUST 3D identity e_sem INTO the gate
                                        # (default on when sem_dim>0; A/B with 0 = gate sees only the 2D patch)
        gate_entity_pool: bool = False, # §49: pool the gate LOGIT over each seg entity -> ONE move/stay
                                        # decision per OBJECT (epi440 failure: gate closed on 59% of one
                                        # object's controls -> half the object froze = "part explodes/laggs").
                                        # move/stay IS an object-level property in rigid-entity data.
        entity_lbs: bool = False,       # §49: dense LBS binds only within the same seg entity (needs
                                        # seg_per_g passed to forward; kills cross-boundary dilution).
        rel_head: bool = False,         # §54: per-control LANGUAGE-RELEVANCE head — the language-causal
                                        # path. Each control's patch feature cross-attends the per-token
                                        # INSTRUCTION features; the relevance logit is ADDED to the gate
                                        # logit (zero-init => warm-start identical; un-nullable by an MLP).
        entity_head: bool = False,      # §54: entity-slot SE(3) head — per-entity rigid (v_e, ω_e) from
                                        # scatter-pooled DiT features (+ pooled relevance feature); the
                                        # per-control head degrades to a small residual. Rigidity by
                                        # construction (LBS reproduces the entity transform EXACTLY).
        rigid_agg: bool = False,        # §66 v10-rigid: project the per-control motion VOTES onto one
                                        # weighted-Kabsch SE(3) per entity (parameter-free; identity on a
                                        # rigid field => exact warm-start). Unlike entity_head it aggregates
                                        # in OUTPUT space (votes), so direction survives. Needs entity_lbs.
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
        self.dynamics = GaussianDynamics(dyn_cfg, entity_head=bool(entity_head), rigid_agg=bool(rigid_agg))
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
        # ---- Exp-1: per-control DYNAMICS GATE (mover/static) + optional OBJECT-SEMANTIC head.
        # Both read the SAME per-control frozen-Qwen patch feature the vis_* paths use (sampled at the
        # control's frame-0 uv in _control_visual). dyn_head -> a scalar logit p_dyn per control; the
        # dynamics MULTIPLIES the bounded velocity by sigmoid(p_dyn) (model.py gate_local), so the head
        # structurally forbids motion on controls it calls static and routes it to movers. p_dyn is
        # supervised by the FREE sim mover label (GT disp>thresh) -> the explicit move/stay signal the
        # collapsing L1 never gave (notes Exp-1 §2.1/§2.2). Needs spatial_ground (the patch feature).
        self.dyn_gate = bool(dyn_gate)
        self.sem_dim = int(sem_dim)
        self.gate_entity_pool = bool(gate_entity_pool)
        self.entity_lbs = bool(entity_lbs)
        # ---- §54 relevance head: WHICH entity does the instruction name? q = the control's 2D
        # patch feature (visual identity, instruction-blind by Qwen's causal mask), k/v = per-token
        # instruction features. Output a_i [M,256] + r_logit [M,1]; r_logit ADDS to the gate logit
        # (then §49 entity pooling pools the SUM). Zero-init last layer => exact identity warm-start.
        self.rel_head_on = bool(rel_head)
        if self.rel_head_on:
            assert cond_mode == "aggregator", "rel_head needs per-token text feats (aggregator mode)"
            assert self.dyn_gate, "rel_head feeds the dyn-gate; enable --dyn_gate"
            H = self.encoder.hidden_size
            rd = 256
            self.rel_dim = rd
            self.rel_norm = nn.LayerNorm(H, eps=1e-6)          # bound Qwen-scale text K/V (cf. vis_norm)
            self.rel_q = nn.Linear(H, rd)
            self.rel_attn = CrossAttention(rd, 4, ctx_dim=H)
            self.rel_mlp = nn.Sequential(nn.Linear(rd, rd), nn.SiLU())
            self.rel_logit = nn.Linear(rd, 1)
            nn.init.zeros_(self.rel_logit.weight); nn.init.zeros_(self.rel_logit.bias)
        # ---- §54 entity-slot SE(3) head lives in the dynamics module (needs the DiT features);
        # here we only record the flag so forward() routes seg + pooled relevance feats down.
        self.entity_head_on = bool(entity_head)
        self.rigid_agg = bool(rigid_agg)                         # §66 v10-rigid
        if self.rigid_agg and not self.entity_lbs:
            import warnings
            warnings.warn("rigid_agg=True without entity_lbs=True: control-level rigidity will NOT "
                          "propagate to a clean DENSE rigid motion (dense points may bind cross-entity "
                          "controls). Set entity_lbs=1.")
        # §44h: feed the OCCLUSION-ROBUST 3D identity e_sem into the gate. Only meaningful when the gate
        # AND the sem head are both on; otherwise it is a no-op (the gate sees only the 2D Qwen patch).
        self.gate_uses_sem = bool(gate_uses_sem) and self.dyn_gate and self.sem_dim > 0
        if self.dyn_gate:
            assert self.spatial_ground, "--dyn_gate needs --spatial_ground 1 (the per-control Qwen patch feature)"
            H = self.encoder.hidden_size
            if self.sem_dim > 0:
                # OPTIONAL (Exp-1 #3): per-control object-identity embedding, supervised by seg_per_g
                # (Gaussian-Grouping CE-to-LEARNABLE-prototype + 3D-NN consistency) so each control
                # encodes "what entity am I". Default OFF (sem_dim=0). BUILT BEFORE the gate so its
                # output e_sem can be concatenated into the gate input (§44h, gate_uses_sem).
                #
                # NOTE (bug fixed): the last layer was zero-init -> e_sem≡0 for every control. The old
                # semantic_id_loss built its CE prototypes as the DETACHED batch-mean of e_sem, so with
                # e_sem=0 the prototypes were 0, the logits were 0 (uniform softmax -> CE=log#entities),
                # AND ∂logits/∂z = protos.t() = 0 => the gradient to sem_head was EXACTLY zero. That is a
                # dead saddle: the loss sat at log(C)≈2.7 and AdamW could never move the head off zero.
                # FIX: (1) small NON-zero init on the last layer so e_sem!=0 and the gradient is alive
                # from step 1; (2) a LEARNABLE class-prototype bank (below) indexed by the RAW entity id
                # — a real, stable CE target (Gaussian-Grouping's identity classifier) that cannot
                # collapse, instead of a self-referential detached batch mean.
                self.sem_head = nn.Sequential(nn.Linear(H, d), nn.SiLU(), nn.Linear(d, self.sem_dim))
                nn.init.normal_(self.sem_head[-1].weight, std=0.02); nn.init.zeros_(self.sem_head[-1].bias)
                # Learnable cosine prototypes, one row PER RAW ENTITY ID (sparse ids index directly: row
                # `id`). 64 rows safely covers the ManiSkill entity ids (observed sparse {1..18}); unused
                # rows simply never receive gradient. Indexing by the raw id (not a per-batch dense remap)
                # keeps each entity's prototype STABLE across steps/clips so it can actually be learned.
                self.sem_classes = 64
                self.sem_proto = nn.Parameter(torch.randn(self.sem_classes, self.sem_dim) * 0.02)
            # §44h: the gate's input is the per-control 2D Qwen patch feature CONCATENATED with the
            # occlusion-robust 3D identity e_sem (gate_uses_sem). At a pixel where the arm OCCLUDES the
            # table the 2D patch is the ARM's feature (overlap-ambiguous) -> the table-control could leak;
            # e_sem (supervised by the 3D per-Gaussian seg id, overlap-INVARIANT) lets the gate know "this
            # 3D point is table (static)" even under the arm. gate_uses_sem=0 -> gate sees only H (A/B).
            gate_in = H + (self.sem_dim if self.gate_uses_sem else 0)
            self.dyn_head = nn.Sequential(nn.Linear(gate_in, d), nn.SiLU(), nn.Linear(d, 1))
            # WARM-START so sigmoid(p_dyn)≈1 initially (do NOT kill all motion at init): zero the last
            # layer's weight and set a large +bias -> the gate starts ~open (identity), then learns to
            # CLOSE on confident statics. (Guards the "gate collapses to 0" inverse failure, notes §Risk.)
            # NOTE: zeroing the LAST layer makes p_dyn≡bias at init REGARDLESS of the input width, so the
            # warm-start (sigmoid(4.0)=0.982, gate open) is byte-identical whether or not e_sem is
            # concatenated; the random first-layer weights on the e_sem columns are nulled at init and
            # only become active as gradients flow. => --gate_uses_sem 0/1 share the exact init behaviour.
            nn.init.zeros_(self.dyn_head[-1].weight); nn.init.constant_(self.dyn_head[-1].bias, 4.0)
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
        # §54: per-token INSTRUCTION features for the relevance head (the per-control language
        # binding). Qwen encodes [image, text] causally with the image FIRST, so the image-patch
        # features can NEVER contain instruction information — the text tokens are the only
        # carrier, and they must be exposed per-token (pooling destroys "WHICH object is named").
        text_feats = hidden_all[-1][text_mask].detach()        # [L_t,H] frozen-Qwen, no grad
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
        return ctx_per_block, ctx_mask, cond_global, pooled, text_feats

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
        # metaquery has no per-token text features (queries only) -> relevance head unsupported
        return ctx_per_block, ctx_mask, cond_global, pooled, None

    def _control_visual(self, vlm_inputs: dict, control_uv: torch.Tensor, uv_hw, d_device, d_dtype):
        """Sample a per-control visual feature from the frozen-Qwen IMAGE grid at each control's
        frame-0 (u,v). control_uv [M,2] is (x=col, y=row) in the LIFTED image pixel space of size
        uv_hw=(H,W); grid_sample uses NORMALIZED [-1,1] coords so the (different) Qwen grid
        resolution is irrelevant. Returns (vis_tok [M,d], vis_film [M,2d], vis_vhead [M,3],
        dyn_logit [M,1] | None, sem [M,S] | None)."""
        grid, ghw = self.encoder.image_grid_features(vlm_inputs)
        if grid is None:
            return None, None, None, None, None, None
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
        # Exp-1: the dyn-gate logit + (optional) object-semantic embedding read the SAME patch feature.
        # §44h: compute e_sem FIRST so the gate can consume the OCCLUSION-ROBUST 3D identity (not only the
        # overlap-ambiguous 2D patch). gate input = concat([2D patch feat, e_sem]) when gate_uses_sem.
        sem = self.sem_head(feat) if (self.dyn_gate and self.sem_dim > 0) else None  # [M,S] or None
        dyn_logit = None
        if self.dyn_gate:
            gate_in = torch.cat([feat, sem], dim=-1) if self.gate_uses_sem else feat  # [M,H(+S)]
            dyn_logit = self.dyn_head(gate_in)                                         # [M,1]
        # §54: also expose the post-norm patch feature itself (the relevance head's query; reused
        # verbatim for the counterfactual pass — patch features are instruction-independent).
        return self.vis_tok(feat), self.vis_film(feat), self.vis_vhead(feat), dyn_logit, sem, feat

    def _relevance_logit(self, patch_feat, text_feats):
        """§54 the language-causal path: per-control relevance r_logit [M,1] = does the instruction
        NAME this control's object? q = the control's (instruction-blind) 2D patch feature, k/v = the
        per-token instruction features. Same patch + different text MUST flip r => a vision-only
        solution is unsatisfiable (this is what the counterfactual loss exploits)."""
        q = self.rel_q(patch_feat)[None]                          # [1,M,rd]
        a = self.rel_attn(q, self.rel_norm(text_feats)[None])[0]  # [M,rd]
        a = self.rel_mlp(a)                                       # [M,rd] (also the entity head's lang feat)
        return self.rel_logit(a), a                              # ([M,1], [M,rd])

    def _pool_logit(self, logit, seg_c):
        """§49 entity pooling: average a per-control logit [M,1] within each seg entity -> [M,1]
        (one move/stay decision per object). seg_c [M] long. No-op if pooling off / seg absent."""
        if not self.gate_entity_pool or seg_c is None:
            return logit
        uniq, inv = torch.unique(seg_c, return_inverse=True)
        summ = torch.zeros(uniq.numel(), device=logit.device, dtype=logit.dtype)
        cnt = torch.zeros_like(summ)
        summ.scatter_add_(0, inv, logit[:, 0])
        cnt.scatter_add_(0, inv, torch.ones_like(logit[:, 0]))
        return ((summ / cnt.clamp_min(1))[inv])[:, None]

    def forward(self, vlm_inputs: dict, dense_g0: GaussianSet, K: int, start_step: int = 0,
                ctrl_idx=None, vlm_inputs_wrong: dict | None = None, actions=None,
                control_uv=None, control_uv_hw=None, seg_per_g=None):
        ctx, m, cond_global, pooled_text, text_feats = self.encode(vlm_inputs)
        # per-control seg ids (for entity pooling + the object-class relevance mask)
        seg_c = seg_per_g.to(cond_global.device)[ctrl_idx].long() if (seg_per_g is not None and ctrl_idx is not None) else None
        # per-step action embedding (control input that causes the future)
        act_emb = self.action_embed(actions) if (self.action_dim > 0 and actions is not None) else None

        # ---- per-control SPATIAL visual grounding (agent.md §37) + §54 language relevance ----
        vis_tok = vis_film = vis_vlogit = None
        dyn_logit = sem_emb = gate_local = None   # Exp-1 dynamics gate / object-semantic
        patch_feat = r_logit = p_dyn_pooled = rel_feat = None  # §54 relevance plumbing (reused: CF + entity head)
        objmask = None
        if self.spatial_ground and control_uv is not None:
            vt = self._control_visual(
                vlm_inputs, control_uv, control_uv_hw,
                cond_global.device, cond_global.dtype)
            if vt[0] is not None:
                vis_tok = vt[0][None]                              # [1,M,d]
                vis_film = vt[1][None]                             # [1,M,2d]
                vis_vlogit = vt[2][None]                           # [1,M,3]
                patch_feat = vt[5]                                 # [M,H] post-norm patch (q for relevance)
                if vt[3] is not None:                              # Exp-1: dyn-gate logit [M,1]
                    dyn_logit = vt[3]                              # [M,1] raw VISUAL move/stay logit
                    comb = dyn_logit
                    # §54: relevance ADDS to the gate, but ONLY for OBJECT-class controls (seg 1..7).
                    # The robot (arm 8 / gripper 10) executes under EVERY instruction, so its gate stays
                    # purely visual; only WHICH OBJECT is picked is instruction-dependent.
                    if self.rel_head_on and text_feats is not None and seg_c is not None:
                        objmask = ((seg_c >= 1) & (seg_c <= 7)).to(dyn_logit.dtype)[:, None]   # [M,1]
                        r_logit, rel_feat = self._relevance_logit(patch_feat, text_feats)     # [M,1],[M,rd]
                        comb = dyn_logit + r_logit * objmask
                    # §49 entity pooling of the (visual + language) gate logit -> one decision/object
                    p_dyn_pooled = self._pool_logit(comb, seg_c)   # [M,1]
                    gate_local = torch.sigmoid(p_dyn_pooled)[None] # [1,M,1] p_dyn∈[0,1] for the gate
                if vt[4] is not None:                              # Exp-1 #3 (optional): semantic emb
                    sem_emb = vt[4]                                # [M,S]

        # §54 entity-slot routing: pass the per-control seg ids + relevance feature so the dynamics can
        # pool a per-entity rigid SE(3) each step. resid_accum collects the per-step residual magnitude.
        # seg ids feed the entity head AND the §66 rigid-agg layer (both need per-entity grouping).
        ent_seg = seg_c if ((self.entity_head_on or self.rigid_agg) and seg_c is not None) else None
        ent_rel = rel_feat if (self.entity_head_on and rel_feat is not None) else None
        resid_accum = []

        def delta_fn(state: GaussianState, step_idx):
            cond = cond_global
            if act_emb is not None:
                k = int(step_idx[0].item()) - start_step
                if 0 <= k < act_emb.shape[0]:
                    cond = cond_global + act_emb[k][None]
            res = self.dynamics(state, ctx, m, cond, step_idx,
                                cond_local=vis_tok, film_local=vis_film, v_logit_local=vis_vlogit,
                                gate_local=gate_local, seg_local=ent_seg, rel_feat_local=ent_rel)
            if self.entity_head_on and self.dynamics.last_resid is not None:
                resid_accum.append(self.dynamics.last_resid)
            return res

        # §49: with entity_lbs, seg_per_g (per-dense-Gaussian entity id, from the clip) switches the
        # LBS binding to ENTITY-AWARE — a dense point deforms only with its own entity's controls.
        roll = SCGSRollout(dense_g0, n_control=self.n_control, k=self.lbs_k, ctrl_idx=ctrl_idx,
                           dense_seg=(seg_per_g if (self.entity_lbs and seg_per_g is not None) else None))
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
        if resid_accum:                                        # §54 residual reg target (push motion -> entity SE3)
            out["resid_norm"] = torch.stack(resid_accum).mean()
        # Exp-1: expose the per-control dynamics-gate logit (supervised by the free mover label in the
        # trainer) and the optional object-semantic embedding. At INFERENCE these are PREDICTED from the
        # Qwen patch feature alone (no GT needed) — the gate has already shaped out["v"]/["ctrl"] above.
        if p_dyn_pooled is not None:
            out["p_dyn"] = p_dyn_pooled[:, 0]                   # [M] pooled VISUAL+LANG gate logit (mover-BCE)
        if r_logit is not None:
            out["p_rel"] = r_logit[:, 0]                        # [M] raw relevance logit (relevance-BCE vs is_obj)
            out["objmask"] = objmask[:, 0]                      # [M] 1 = object-class control (relevance applies)
        if sem_emb is not None:
            out["e_sem"] = sem_emb                              # [M,S] object-identity embedding
            out["sem_proto"] = self.sem_proto                  # [n_classes,S] LEARNABLE id prototypes (CE target)
        # ---- InfoNCE embeddings: a PREDICTED-motion embedding + the instruction embedding.
        # Aligning these (vs other clips' instructions, in the trainer) forces the dynamics to
        # make its motion instruction-specific (non-saturating MI lower bound; research_F #1).
        v = out["v"]                                           # [K,M,3]
        mfeat = torch.cat([v.mean(0), v.std(0)], dim=-1)      # [M,6] per-control motion stats
        mfeat = self.motion_enc(mfeat).mean(0)               # DeepSets pool over controls -> [256]
        out["motion_emb"] = torch.nn.functional.normalize(self.motion_head(mfeat), dim=-1)   # [proj]
        out["lang_emb"] = torch.nn.functional.normalize(self.lang_proj(pooled_text.float()), dim=-1)
        # §54 COUNTERFACTUAL: re-score relevance under a WRONG instruction — SAME patch features, only
        # the text K/V change, so ONE extra frozen-Qwen forward for the text (no rollout/render). Under a
        # wrong instruction the TRUE-named object must NOT be selected: the trainer supervises p_rel_wrong
        # / p_dyn_wrong -> 0 on its controls. Same patch + different text forced to flip the gate is what
        # makes a vision-only solution unsatisfiable (breaks the gripper-proximity shortcut, §52a).
        if vlm_inputs_wrong is not None and self.rel_head_on and patch_feat is not None:
            hidden_w, _, tmask_w = self.encoder(vlm_inputs_wrong)
            text_feats_w = hidden_w[-1][tmask_w].detach()        # [L_t',H] wrong-instruction text feats
            r_logit_w, _ = self._relevance_logit(patch_feat, text_feats_w)     # [M,1] reuse the patch q
            out["p_rel_wrong"] = r_logit_w[:, 0]
            if dyn_logit is not None:
                mw = objmask if objmask is not None else torch.ones_like(r_logit_w)
                out["p_dyn_wrong"] = self._pool_logit(dyn_logit + r_logit_w * mw, seg_c)[:, 0]
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
