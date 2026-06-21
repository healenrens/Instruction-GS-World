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
from .tokens import sample_grid_feat, project_to_uv, to_cam
from .losses import geom_loss, jepa_loss, relevance_infonce, mover_magnitude


def _footprint_sample(grid, ghw, uv, sigma_px, H, W):
    """JEPA non-uniform adaptation: pool a FROZEN pretrained feature grid over each token's sigma
    footprint (5-point plus-pattern at center + ±0.7σ) instead of point-sampling — the pretrained grid
    is UNIFORM-patch, our tokens are non-uniform, so we average the patch features the token covers."""
    offs = torch.tensor([[0., 0.], [.7, 0.], [-.7, 0.], [0., .7], [0., -.7]], device=uv.device, dtype=uv.dtype)
    acc = 0.
    for o in offs:
        acc = acc + sample_grid_feat(grid, ghw, uv + o[None] * sigma_px, H, W)
    return acc / offs.shape[0]


class GPSTokenWM(nn.Module):
    def __init__(self, qwen_path: str | None = None, d_model: int = 1536, n_heads: int = 16,
                 n_query: int = 16, fdim: int = 128, geom_mode: str = "xyz", num_freqs: int = 10,
                 feat_source: str = "qwen", dino_imgsize: int = 518, traj_pred: bool = False, Kf: int = 12):
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
        # CURVE/TRAJECTORY mode (--traj_pred): instead of ONE future xyz (frame Kf), the geom head outputs
        # PER-FRAME displacements (Kf x 3) = (Δu_t,Δv_t,Δlogz_t) for t=1..Kf, each unprojected to 3D by the
        # SAME flowd geometry. Captures the full curve + the non-uniform speed profile. The endpoint (last
        # waypoint, t=Kf) is directly comparable to the straight baseline. Zero-init => no-motion start.
        self.traj_pred, self.Kf = bool(traj_pred), int(Kf)
        out_dim = 3 * self.Kf if self.traj_pred else 3
        self.geom_head = nn.Linear(d, out_dim)
        nn.init.zeros_(self.geom_head.weight); nn.init.zeros_(self.geom_head.bias)   # start = no motion
        self.content_head = nn.Linear(d, fdim)
        # v2 fusion (--fuse): JEPA predicts into the FROZEN PRETRAINED latent (raw DINOv2, NOT our feat_in),
        # as a world-model read-out of "how the latent at this place changes" — derived from the same future
        # hidden x that produces motion (a consistent second view of the future, not a competing head).
        self.jepa_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, feat_dim_in))
        self.fuse = False
        # ---- B: camera-conditioning (--cam_cond). The predictor outputs world motion but its INPUT is
        # viewpoint-dependent (frame0 RGB + which tokens get placed) with NO camera signal -> it overfits
        # one viewpoint (agent.md §97/§98). Inject the camera pose at 2 levels so varying cameras become a
        # generalization ASSET, not poison. BOTH zero-init => exact no-op at start => WARM-STARTABLE from the
        # fixed-camera v2 (keep the 0.92), then learn to use the camera. cam_head: global pose -> cond;
        # cam_tok_head: per-token camera-frame position -> hidden (aligns RGB feat with view geometry).
        self.cam_head = nn.Sequential(nn.Linear(13, d), nn.SiLU(), nn.Linear(d, d))
        self.cam_tok_head = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        for m in (self.cam_head, self.cam_tok_head):
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        self.cam_cond = False
        self.jepa_couple = False   # ablation: if True, JEPA grad flows INTO the trunk (multi-task) instead of stop-grad read-out
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
                ctx_per_block, ctx_mask, cond_global, cam_tok=None):
        """tokens (frame0) -> per-token hidden x [1,M,d]. tok_feat_fdim [M,fdim] = self.feat_in(grid feat),
        computed ONCE by the caller and reused for SIGReg/relevance/JEPA-target.
        cam_tok [M,d] (--cam_cond): per-token camera-frame geometry, zero at warm-start."""
        norm_xyz = (tok_xyz0 - center) / radius
        pe = self.pe(norm_xyz[None])[0]                                       # [M, pe_out]
        # fp32 cat (pe/sigma are fp32, feat is bf16 under autocast) -> tok_embed Linear re-casts
        x = torch.cat([pe.float(), tok_feat_fdim.float(), tok_sigma.float()], dim=-1)[None]
        x = self.tok_embed(x)
        if cam_tok is not None:
            x = x + cam_tok[None]                                             # inject per-token camera geometry
        for j, blk in enumerate(self.blocks):
            x = blk(x, cond_global, ctx_per_block[:, j], ctx_mask)
        return self.final_norm(x)                                            # [1,M,d]

    def cam_cond_signals(self, tok_xyz0, center, radius, K_intr, viewmat):
        """--cam_cond: encode the camera pose into (cam_global [1,d] added to cond, cam_tok [M,d] added to
        per-token hidden). Zero-init heads => no-op at warm-start. World motion stays the target; the camera
        lets the predictor interpret the viewpoint-dependent input -> viewpoint-INVARIANT prediction."""
        R = viewmat[:3, :3]; t = viewmat[:3, 3]
        campos = -(R.t() @ t)                                                # camera center in world
        fx, fy, cx, cy = K_intr[0, 0], K_intr[1, 1], K_intr[0, 2], K_intr[1, 2]
        Wn = (cx * 2).clamp_min(1.0); Hn = (cy * 2).clamp_min(1.0)
        feats = torch.stack([campos[0], campos[1], campos[2],               # camera position (3)
                             R[2, 0], R[2, 1], R[2, 2], R[1, 0], R[1, 1], R[1, 2],  # look-dir + up (6)
                             fx / Wn, fy / Hn, cx / Wn, cy / Hn])            # fov/principal (4) = 13
        cam_global = self.cam_head(feats)[None]                              # [1,d]
        ones = torch.ones(tok_xyz0.shape[0], 1, device=tok_xyz0.device, dtype=tok_xyz0.dtype)
        cam0 = (torch.cat([tok_xyz0, ones], -1) @ viewmat.T)[:, :3]          # token position in CAMERA frame
        cam_tok = self.cam_tok_head(cam0 / radius)                          # [M,d], scene-scale normalized
        return cam_global, cam_tok

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

    def geom_to_traj(self, g, tok_xyz0, K_intr, viewmat):
        """CURVE mode. g [M, 3*Kf] = per-frame (Δu_t,Δv_t,Δlogz_t) RELATIVE to frame0 (absolute offset from
        the frame0 image position, NOT incremental) -> full 3D trajectory [Kf, M, 3] via the flowd unproject.
        Each frame is unprojected from frame0's (u0,v0,z0) + the predicted offset (same math as geom_to_xyz
        flowd). Returns the per-frame world xyz for t=1..Kf."""
        M = tok_xyz0.shape[0]
        ones = torch.ones(M, 1, device=tok_xyz0.device, dtype=tok_xyz0.dtype)
        cam0 = (torch.cat([tok_xyz0, ones], -1) @ viewmat.T)[:, :3]
        z0 = cam0[:, 2:3].clamp_min(1e-4)
        fx, fy, cx, cy = K_intr[0, 0], K_intr[1, 1], K_intr[0, 2], K_intr[1, 2]
        u0 = cam0[:, 0:1] / z0 * fx + cx
        v0 = cam0[:, 1:2] / z0 * fy + cy
        gk = g.view(M, self.Kf, 3)                                          # [M,Kf,3]
        inv = torch.linalg.inv(viewmat.float()).to(tok_xyz0.dtype)
        out = []
        for t in range(self.Kf):
            du, dv, dlz = gk[:, t, 0:1], gk[:, t, 1:2], gk[:, t, 2:3]
            u1, v1 = u0 + du, v0 + dv
            z1 = z0 * torch.exp(dlz.clamp(-2.0, 2.0))
            xc = (u1 - cx) / fx * z1
            yc = (v1 - cy) / fy * z1
            cam1 = torch.cat([xc, yc, z1], -1)
            out.append((torch.cat([cam1, ones], -1) @ inv.T)[:, :3])
        return torch.stack(out, 0)                                          # [Kf, M, 3]

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
        cam_tok = None
        if getattr(self, "cam_cond", False):                                 # B: inject camera pose
            cg, cam_tok = self.cam_cond_signals(b["tok_xyz0"].float(), b["center"], b["radius"],
                                                b["K_intr"].float(), b["viewmat"].float())
            cond = cond + cg
        x = self.predict(b["tok_xyz0"], tok_feat, b["sig_n"], b["center"], b["radius"], ctx, ctxm, cond, cam_tok=cam_tok)
        traj_pred_xyz = None
        if getattr(self, "fuse", False):
            h = x[0]
            gl = self.relevance(tok_feat, text_feats.mean(0))                     # gate logit [M]
            gate = torch.sigmoid(gl)                                              # [M] instruction-relevance GATE
            g = gate[:, None] * self.geom_head(h).float()                         # grounding GATES motion (not parallel head)
            if self.traj_pred:                                                    # CURVE: per-frame trajectory
                traj_pred_xyz = self.geom_to_traj(g, b["tok_xyz0"].float(), b["K_intr"].float(), b["viewmat"].float())
                xyz1_pred = traj_pred_xyz[-1]                                     # endpoint = last waypoint (t=Kf)
            else:
                xyz1_pred = self.geom_to_xyz(g, b["tok_xyz0"].float(), b["K_intr"].float(), b["viewmat"].float())
            jepa_pred = self.jepa_head(h if getattr(self, "jepa_couple", False) else h.detach()).float()  # READ-OUT (stop-grad, "不抢主干"); --jepa_couple lets JEPA grad shape the trunk (ablation)
            feat_pred = None; self._gate = gate; self._gate_logit = gl
        else:
            if self.traj_pred:
                g = self.geom_head(x[0]).float()
                traj_pred_xyz = self.geom_to_traj(g, b["tok_xyz0"].float(), b["K_intr"].float(), b["viewmat"].float())
                xyz1_pred = traj_pred_xyz[-1]
                feat_pred = self.content_head(x[0])
            else:
                xyz1_pred, feat_pred = self.heads(x, b["tok_xyz0"], b["K_intr"], b["viewmat"])
        with torch.no_grad():
            gridK, ghwK = (self.dino.grid(b["rgbK_np"]) if self.dino is not None
                           else self.encoder.image_grid_features(b["vlmK"]))
            fut_uv = project_to_uv(b["xyz1_gt"], b["K_intr"], b["viewmat"])
            if getattr(self, "fuse", False):
                # JEPA target = RAW frozen PRETRAINED feature (NOT our feat_in), footprint-pooled at the
                # future location (= where the token moves) -> "latent at the moved place", a pretrained
                # latent we predict to retain future-feature prediction without competing with motion.
                sig_px = b["sig_n"] * float(max(b["H"], b["W"]))
                tgt = _footprint_sample(gridK, ghwK, fut_uv, sig_px, b["H"], b["W"]).float().detach()
            else:
                tgt = self.feat_in(sample_grid_feat(gridK, ghwK, fut_uv, b["H"], b["W"])).float().detach()
        # direction-PRESERVING magnitude fix: up-weight high-motion tokens in the POSITION loss (vs the
        # dead-end relative mover_magnitude which inflated wrong directions). Still smooth-L1 on xyz => no
        # direction damage; just makes the model serve big movers (which smooth-L1's median-seeking under-serves).
        mw = None
        if self.w_motion > 0:
            d = b["disp_tok"]
            mv = d > 0.01
            mvmed = d[mv].median() if mv.any() else d.new_tensor(0.05)
            mw = (1.0 + self.w_motion * (d / mvmed.clamp_min(1e-3))).clamp(max=getattr(self, "mw_cap", 10.0))
        if getattr(self, "img_loss", False):
            # IMAGE-NORMALIZED 2D-flow target (user's idea §95续18): supervise per-token motion as the
            # NORMALIZED 2D IMAGE displacement (Δu/W, Δv/H) — how far it moves in the image as a fraction
            # of image size. Tied to the GPSToken 2D representation, and a consistently-scaled target
            # (image fractions, not wildly-varying meters) -> directly attacks magnitude under-prediction.
            # Small 3D anchor (0.1*geom) keeps depth sane.
            K_ = b["K_intr"].float(); vm_ = b["viewmat"].float()
            Wn = xyz1_pred.new_tensor([float(b["W"]), float(b["H"])])
            uv0 = project_to_uv(b["tok_xyz0"].float(), K_, vm_)
            z0 = to_cam(b["tok_xyz0"].float(), vm_)[:, 2].clamp_min(1e-3)
            if self.traj_pred:
                # CURVE: average the SAME normalized-image-flow + depth loss over ALL frames t=1..Kf, GT=traj[t].
                # _fp/_fg keep the ENDPOINT (t=Kf) flow for the dcos/magR logs (comparable to the straight base).
                tg = b["traj_gt"].float()                                         # [Kf, M, 3]
                img_acc = depth_acc = 0.0
                for t in range(self.Kf):
                    uvtp = project_to_uv(traj_pred_xyz[t], K_, vm_)
                    uvtg = project_to_uv(tg[t], K_, vm_)
                    fpt = (uvtp - uv0) / Wn; fgt = (uvtg - uv0) / Wn
                    pert = F.smooth_l1_loss(fpt, fgt, beta=0.02, reduction="none").mean(-1)
                    img_acc = img_acc + ((pert * mw).sum() / mw.sum().clamp_min(1e-6) if mw is not None else pert.mean())
                    ldpt = torch.log(to_cam(traj_pred_xyz[t], vm_)[:, 2].clamp_min(1e-3) / z0)
                    ldgt = torch.log(to_cam(tg[t], vm_)[:, 2].clamp_min(1e-3) / z0)
                    perzt = F.smooth_l1_loss(ldpt, ldgt, beta=0.05, reduction="none")
                    depth_acc = depth_acc + ((perzt * mw).sum() / mw.sum().clamp_min(1e-6) if mw is not None else perzt.mean())
                    if t == self.Kf - 1:
                        self._fp, self._fg = fpt, fgt                             # endpoint flow for logs
                l_geom = img_acc / self.Kf + getattr(self, "w_depth", 0.5) * (depth_acc / self.Kf)
            else:
                uv1p = project_to_uv(xyz1_pred.float(), K_, vm_)
                uv1g = project_to_uv(b["xyz1_gt"].float(), K_, vm_)
                self._fp = (uv1p - uv0) / Wn; self._fg = (uv1g - uv0) / Wn
                per = F.smooth_l1_loss(self._fp, self._fg, beta=0.02, reduction="none").mean(-1)
                img_l = (per * mw).sum() / mw.sum().clamp_min(1e-6) if mw is not None else per.mean()
                # + normalized DEPTH change (Δlog z): full 3D = image-flow + depth, both consistent-scale
                # targets (vs the failed flowd which used raw-pixel flow + loss on 3D position).
                ldp = torch.log(to_cam(xyz1_pred.float(), vm_)[:, 2].clamp_min(1e-3) / z0)
                ldg = torch.log(to_cam(b["xyz1_gt"].float(), vm_)[:, 2].clamp_min(1e-3) / z0)
                perz = F.smooth_l1_loss(ldp, ldg, beta=0.05, reduction="none")
                depth_l = (perz * mw).sum() / mw.sum().clamp_min(1e-6) if mw is not None else perz.mean()
                l_geom = img_l + getattr(self, "w_depth", 0.5) * depth_l
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
        if getattr(self, "fuse", False):
            jp = F.normalize(jepa_pred, dim=-1); jt = F.normalize(tgt, dim=-1)
            l_jepa = (1.0 - (jp * jt).sum(-1)).mean()                        # cosine into RAW pretrained latent
            l_sig = self.sigreg(jepa_pred)
            mvr = (b["disp_tok"] > 0.01).float()                            # grounding GATE supervised to movers
            l_inst = F.binary_cross_entropy_with_logits(self._gate_logit.float(), mvr)
            relsel = ((self._gate > 0.5).float() == mvr).float().mean()
        else:
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
        if getattr(self, "_ddp_touch", False):  # zero-weight touch so EVERY trainable param participates ->
            loss = loss + 0.0 * sum(p.float().sum()  # DDP runs with find_unused_parameters=False (the correct path)
                                    for p in self.parameters() if p.requires_grad)
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
