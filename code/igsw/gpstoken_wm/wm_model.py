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
from .sigreg import SIGReg
from .tokens import sample_grid_feat, project_to_uv
from .losses import geom_loss, jepa_loss, relevance_infonce, mover_magnitude


class GPSTokenWM(nn.Module):
    def __init__(self, qwen_path: str | None = None, d_model: int = 1536, n_heads: int = 16,
                 n_query: int = 16, fdim: int = 128, geom_mode: str = "xyz", num_freqs: int = 10,
                 feat_source: str = "qwen", dino_imgsize: int = 518):
        super().__init__()
        self.encoder = QwenVLEncoder(qwen_path) if qwen_path else QwenVLEncoder()
        H = self.encoder.hidden_size
        # direction A: token VISUAL feature source — "qwen" (VLM patches) or "dino" (frozen DINOv2 dense).
        # Qwen always does the LANGUAGE conditioning regardless; this only swaps the per-token visual feat.
        # dino_imgsize raises DINOv2 input res (518->770 = 37->55 patch grid) = FINER per-token features =
        # lower position-noise floor (PLAN §7 / §95续10: the floor is what surfaces rotation).
        self.feat_source = feat_source
        if feat_source == "dino":
            from .dino_features import DinoFeatures
            self.dino = DinoFeatures(img_size=dino_imgsize)
            feat_dim_in = self.dino.embed_dim
        else:
            self.dino = None
            feat_dim_in = H
        n_l = self.encoder.num_layers
        d = d_model
        self.d, self.fdim, self.geom_mode, self.n_query = d, fdim, geom_mode, n_query
        # ---- conditioning (faithfully copied from model_full.encode: per-layer proj + query aggregator) ----
        self.layer_proj = nn.ModuleList([nn.Linear(H, d) for _ in range(n_l)])
        self.cond_proj = nn.Linear(H, d)
        # (path-1 probe) scale-conditioning: inject the per-clip GLOBAL motion scale into cond so the
        # model can output full magnitude when the otherwise-aleatoric scale is SUPPLIED. zero-init last
        # layer => no-op at start (warm-start safe). Decision test: does supplying the scale fix magR
        # (=> path 1 'supply the scale via goal/action' suffices) or not (=> path 2 generative needed)?
        self.scale_head = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.scale_head[-1].weight); nn.init.zeros_(self.scale_head[-1].bias)
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.layer_id_emb = nn.Parameter(torch.randn(n_l, d) * 0.02)
        self.aggregator = CrossAttention(d, n_heads, ctx_dim=d)
        self.agg_norm = nn.LayerNorm(d, eps=1e-6)
        # ---- token embed: FourierPE(3D pos) ⊕ frozen feat ⊕ sigma -> d ----
        self.pe = FourierPE3D(num_freqs=num_freqs)
        self.feat_in = nn.Linear(feat_dim_in, fdim)            # frozen visual patch feat -> token feature ch.
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
        self.sigreg = SIGReg(num_proj=512)
        self.w_jepa, self.w_sigreg, self.w_ground = 0.5, 0.05, 1.0   # set by the trainer from args
        self.w_mag = 0.0                                             # mover-magnitude (DEAD END, kept off)
        self.w_motion = 0.0                                          # motion-weighted geom loss (direction-preserving mag fix)

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

    def forward(self, b: dict):
        """One training step (DDP-safe single entrypoint). b carries the prepared per-clip tensors +
        vlm_inputs. Returns (loss, logs) — loss has grad through all trainable params; logs detached."""
        import torch.nn.functional as F
        ctx, ctxm, cond, text_feats = self.encode_cond(b["vlm0"])
        if getattr(self, "cond_scale", False):                                # supply the (oracle) global scale
            gdm = b["xyz1_gt"].float() - b["tok_xyz0"].float(); mm = b["disp_tok"] > 0.01
            s = (gdm[mm].norm(dim=-1).mean() if mm.any() else gdm.norm(dim=-1).mean()).clamp_min(1e-3)
            cond = cond + self.scale_head(torch.log(s).reshape(1, 1))
        grid0, ghw0 = (self.dino.grid(b["rgb0_np"]) if self.dino is not None
                       else self.encoder.image_grid_features(b["vlm0"]))
        tok_feat = self.feat_in(sample_grid_feat(grid0, ghw0, b["cen"], b["H"], b["W"])).float()
        x = self.predict(b["tok_xyz0"], tok_feat, b["sig_n"], b["center"], b["radius"], ctx, ctxm, cond)
        xyz1_pred, feat_pred = self.heads(x, b["tok_xyz0"], b["K_intr"], b["viewmat"])
        with torch.no_grad():
            gridK, ghwK = (self.dino.grid(b["rgbK_np"]) if self.dino is not None
                           else self.encoder.image_grid_features(b["vlmK"]))
            fut_uv = project_to_uv(b["xyz1_gt"], b["K_intr"], b["viewmat"])
            tgt = self.feat_in(sample_grid_feat(gridK, ghwK, fut_uv, b["H"], b["W"])).float().detach()
        # direction-PRESERVING magnitude fix: up-weight high-motion tokens in the POSITION loss (vs the
        # dead-end relative mover_magnitude which inflated wrong directions). Still smooth-L1 on xyz => no
        # direction damage; just makes the model serve big movers (which smooth-L1's median-seeking under-serves).
        mw = None
        if self.w_motion > 0:
            d = b["disp_tok"]
            mv = d > 0.01
            mvmed = d[mv].median() if mv.any() else d.new_tensor(0.05)
            mw = (1.0 + self.w_motion * (d / mvmed.clamp_min(1e-3))).clamp(max=10.0)
        if getattr(self, "img_loss", False):
            # IMAGE-NORMALIZED 2D-flow target (user's idea §95续18): supervise per-token motion as the
            # NORMALIZED 2D IMAGE displacement (Δu/W, Δv/H) — how far it moves in the image as a fraction
            # of image size. Tied to the GPSToken 2D representation, and a consistently-scaled target
            # (image fractions, not wildly-varying meters) -> directly attacks magnitude under-prediction.
            # Small 3D anchor (0.1*geom) keeps depth sane.
            K_ = b["K_intr"].float(); vm_ = b["viewmat"].float()
            Wn = xyz1_pred.new_tensor([float(b["W"]), float(b["H"])])
            uv0 = project_to_uv(b["tok_xyz0"].float(), K_, vm_)
            uv1p = project_to_uv(xyz1_pred.float(), K_, vm_)
            uv1g = project_to_uv(b["xyz1_gt"].float(), K_, vm_)
            self._fp = (uv1p - uv0) / Wn; self._fg = (uv1g - uv0) / Wn
            per = F.smooth_l1_loss(self._fp, self._fg, beta=0.02, reduction="none").mean(-1)
            img_l = (per * mw).sum() / mw.sum().clamp_min(1e-6) if mw is not None else per.mean()
            l_geom = img_l + 0.1 * geom_loss(xyz1_pred.float(), b["xyz1_gt"].float(), weight=mw)
        elif getattr(self, "norm_target", False):
            # SCALE-NORMALIZED target (user's relative-distance idea, §95续16): the model predicts the
            # per-token displacement field DIVIDED by the clip's global motion scale = a unit-scale
            # RELATIVE field. The aleatoric absolute scale is FACTORED OUT of the regression target
            # (recovered separately at inference). Only the TARGET is normalized (normalizing pred too
            # would just be per-clip loss reweighting = the dead-end §95 path).
            pd = xyz1_pred.float() - b["tok_xyz0"].float(); gd = b["xyz1_gt"].float() - b["tok_xyz0"].float()
            m = b["disp_tok"] > 0.01
            if m.sum() >= 5:
                # Scale-EQUALIZE the target to a fixed reference magnitude sref (~the absolute regime),
                # NOT to unit: target = gd * (sref / s_gt). This factors out the per-clip aleatoric scale
                # (the user's relative-field idea) while keeping the target in the SAME gradient regime as
                # the stable absolute baseline (bounded rescale ~0.33-1.25x with the 8cm floor) -> avoids
                # the 10x-hot-gradient + per-clip-amplification instability that collapsed direction.
                sref = 0.1
                s_gt = gd[m].norm(dim=-1).mean().detach().clamp_min(0.08)
                per = F.smooth_l1_loss(pd, gd * (sref / s_gt), beta=0.02, reduction="none").mean(-1)
                l_geom = (per * mw).sum() / mw.sum().clamp_min(1e-6) if mw is not None else per.mean()
            else:
                l_geom = geom_loss(xyz1_pred.float(), b["xyz1_gt"].float(), weight=mw)
        else:
            l_geom = geom_loss(xyz1_pred.float(), b["xyz1_gt"].float(), weight=mw)
        l_mag = mover_magnitude(xyz1_pred.float(), b["tok_xyz0"].float(), b["xyz1_gt"].float(), b["disp_tok"])
        l_jepa = jepa_loss(F.layer_norm(feat_pred.float(), (self.fdim,)),
                           F.layer_norm(tgt, (self.fdim,)).detach())
        l_sig = self.sigreg(tok_feat)
        l_inst = tok_feat.new_zeros(())
        relsel = torch.zeros((), device=tok_feat.device)
        if b.get("is_obj_tok") is not None and self.w_ground > 0:
            rel = self.relevance(tok_feat, text_feats.mean(0))
            pos = b["is_obj_tok"].bool()
            l_inst = relevance_infonce(rel, pos)
            if pos.any():
                relsel = pos[rel.argmax()].float()
        loss = (l_geom + self.w_mag * l_mag + self.w_jepa * l_jepa
                + self.w_sigreg * l_sig + self.w_ground * l_inst)
        with torch.no_grad():
            err_pred = (xyz1_pred - b["xyz1_gt"]).norm(dim=-1).mean()
            err_static = (b["tok_xyz0"] - b["xyz1_gt"]).norm(dim=-1).mean()
            mv = b["disp_tok"] > 0.01
            dcos = (F.cosine_similarity((xyz1_pred - b["tok_xyz0"])[mv], (b["xyz1_gt"] - b["tok_xyz0"])[mv],
                    dim=-1).mean() if mv.any() else torch.zeros((), device=tok_feat.device))
            img_magr = torch.zeros((), device=tok_feat.device)
            if getattr(self, "img_loss", False) and mv.any():               # report IMAGE-space dcos + flow magR
                dcos = F.cosine_similarity(self._fp[mv], self._fg[mv], dim=-1).mean()
                img_magr = self._fp[mv].norm(dim=-1).median() / self._fg[mv].norm(dim=-1).median().clamp_min(1e-6)
        logs = {"loss": loss.detach(), "geom": l_geom.detach(),
                "mag": (img_magr if getattr(self, "img_loss", False) else l_mag).detach(), "jepa": l_jepa.detach(),
                "sig": l_sig.detach(), "inst": l_inst.detach() if torch.is_tensor(l_inst) else l_inst,
                "skill": (err_static - err_pred).detach(), "errp": err_pred.detach(),
                "dcos": dcos.detach(), "relsel": relsel.detach(), "fstd": tok_feat.float().std().detach()}
        return loss, logs

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
