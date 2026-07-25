# Module E — Language-conditioned Visual Grounding for 3DGS Dynamics

> Rewritten after thorough, **citation-verified** research (4 parallel research agents →
> `research_E1..E4_*.md`; every paper checked on arXiv/proceedings, June 2026; no
> fabrication; each claim tagged or traceable). Supersedes the prior draft.
> Companion system-design notes: `research_E_DESIGN_draft.md`. Geometry/dynamics
> substrate: `agent.md` + `research_A..D`.

---

## 0. Goal (unchanged) and the corrected diagnosis

**Goal:** predict, *conditioned on language*, how a scene's 3D Gaussians transform, and
roll it out for ≥10 s. Language must actually *control* the predicted 3D motion.

**What we measured (decisive, from our own logs):**
- Feeding Qwen3-VL's **global pooled image+text hidden state** as the dynamics condition:
  swapping the instruction moved that feature by only **0.026** vs **0.334** for
  text-only (**≈12.9×** gap). ⇒ the pooled image+text vector is **frame-0-image-dominated**;
  the model learned "see frame 0 → predict the single most-likely future," ignoring the words.
- Text-only conditioning removed image-domination (language-divergence ↑ ~2.3×) but
  **threw away visual grounding**: an abstract sentence with no pointer into *this* scene.

**Root cause (confirmed by the grounding literature, §3.3):** a single **global** hidden
vector — pooled or text-only — *cannot carry per-object localization*. PnP-OVSS states this
explicitly: pooled embeddings lose patch-word correspondence that cross-attention preserves.

**Correct target:** use Qwen3-VL as a **language-conditioned visual-grounding compiler** —
given (frame 0 + instruction) it tells the dynamics **which Gaussians are the
object/source/target/hand vs. background**, plus the **motion intent** and **relations**.
Language stays the condition, but is routed through grounding, not a global vector.

---

## 1. Core principle

`G0 3DGS + Qwen text-only instruction → dynamics`  ❌ (proven degenerate)
`G0 3DGS + VLM-derived language-conditioned visual grounding → dynamics`  ✅

Qwen becomes a **grounding compiler** (reads frame 0 + instruction → role masks/tokens/intent);
the 3DGS dynamics does the continuous 3D motion. This both restores Qwen's vision **and**
avoids the global-vector failure.

---

## 2. Three information streams (mapped to OUR code)

**A — geometry (exists).** `frame0/video → Pi3 (igsw/lifting) → dense 3DGS + control set`.
Each Gaussian already carries its **frame-0 anchor pixel (u,v)** via
`points_to_gaussians(return_uv=True)`; control points are a known pixel subset via
`SCGSRollout(ctrl_idx)`. ⇒ any 2D map at frame 0 → per-Gaussian value by sampling at (u,v). ✓

**B — language-visual grounding (NEW = Module E).** `instruction + frame0 → frozen
Qwen3-VL (+ small trained interface) →` `GroundedVLMCondition`:
- `role_masks_2d` (object/source/target/hand/background, soft)
- `role_relevance_3d` (per-Gaussian; sample role_masks_2d at each anchor uv)
- `region_tokens [R,d]` (per-role visual-language features)
- `motion_tokens [Q,d]` (task/motion intent: pick/place/pour/push/open/close + relations)

**C — dynamics (exists, extend).** `igsw/dynamics` DiTBlock → add **Region cross-attn**,
**Motion cross-attn**, and a **mask-conditioned delta gate**, on top of Gaussian self-attn.

---

## 3. Verified literature (5 families)

### 3.1 MetaQuery extraction from a frozen MLLM — the central precedent  [VERIFIED]
- **MetaQueries** — "Transfer between Modalities with MetaQueries," Pan et al., **arXiv:2504.06256**
  (Apr 2025). Learnable query tokens appended to a **frozen** MLLM input under its **native
  causal mask**; their **final hidden states** are read; **only queries + a connector** are
  trained. The general mechanism we adopt.
- **AFUN** — affordance foundation model, **arXiv:2606.02551** (2026 preprint; verified to
  exist, treat numbers as preprint-stage). Frozen **Qwen3-VL** (8B headline, **2B explicitly
  ablated → our 2B is precedented**) + **64 MetaQuery tokens (32 semantic + 32 motion)**;
  **only 32.21M trainable** (queries + 2-layer proj MLP + motion decoder); frozen
  {Qwen3-VL, SAM3, Sonata 3D enc}. Semantic queries → SAM3 → functional mask; motion queries →
  decoder → 3D post-contact motion. **We replace AFUN's motion decoder with our SC-GS dynamics.**
  This is almost exactly our target paradigm.

### 3.2 LMM segmentation / region tokens  [VERIFIED]
LISA (2308.00692, `<SEG>`→MLP→SAM; λ_bce 2.0/λ_dice 0.5), GLaMM (2311.03356), **PixelLM**
(2312.02228, *no SAM* — own seg codebook+decoder), **VideoGLaMM** (2411.04923, →**SAM2**,
temporal), u-LLaVA (2311.05348), LISA++ (2312.17240), **Groma** (2404.13013, localized region
tokens via a region proposer, no coord regression), **Ferret** (2310.07704, hybrid
discrete-coords + continuous region features). Lesson: emit **mask/region tokens**, not a
global vector. (If we ever want Qwen to *emit* masks, these are the recipes; for v7-metaquery
we *read* queries instead → no vocab/lm_head expansion needed.)

### 3.3 Training-free grounding from frozen-VLM attention  [VERIFIED]
- **"Your LVLM Only Needs A Few Attention Heads For Visual Grounding"** — **arXiv:2503.06287**,
  CVPR 2025. Only ~3 "localization heads" carry text→image grounding; discover by
  attention-mass + spatial-entropy ranking on a small calibration set; at inference sum those
  heads' last-text-token→image attention, smooth, binarize, (optional SAM). 87.2% REC@RefCOCO,
  fully training-free.
- **PnP-OVSS** (2311.17095, CVPR 2024): cross-attn + GradCAM + Salience-DropOut; states pooled
  embeddings lose localization (our exact failure). Encoder-decoder only → not directly for
  decoder-only Qwen.
- **DAAM** (2210.04885): diffusion attention maps; the upsample+mean-aggregate pattern transfers.

### 3.4 External grounding pipelines (teacher/fallback)  [VERIFIED]
- **GroundingDINO** (Apache-2.0; `IDEA-Research/grounding-dino-tiny`, ~0.7 GB; `transformers`
  zero-shot detection; phrase list → boxes).
- **SAM2** (Apache-2.0; `facebook/sam2.1-hiera-large`; `SAM2VideoPredictor` propagates a
  frame-0 box/mask across the clip). **GroundingDINO→SAM2 = recommended teacher** (~1.6 GB,
  ~8 GB VRAM, pure pip).
- Florence-2 (MIT, polygons), OWLv2 (Apache, boxes) = fallbacks. **SAM3** real
  (2511.16719) but **gated + custom license** → skip for now. Qwen-native boxes (JSON,
  0–1000 normalized) work but have a **single-instance bias** for duplicate objects.

### 3.5 2D→3DGS grounding + VLA + affordance  [VERIFIED]
- **Gen-LangSplat** (2510.22930) — **the cross-scene answer**: one autoencoder pretrained on
  ScanNet, **512→16-D, frozen for ALL scenes, no per-scene fitting** (vs LangSplat 2312.16084
  per-scene AE, which is *not* cross-scene reusable). ⇒ Module E stores a **shared frozen
  feature→16-D latent on canonical (frame-0) Gaussians**.
- **OpenGaussian** (2406.02058): per-Gaussian instance codes via **intra-mask smoothing +
  inter-mask contrastive** losses + mask-level CLIP association (no depth test). Our role masks
  drop in for SAM masks. **Feature-3DGS** (2312.03203): differentiable N-D feature rasterizer
  to distill any teacher onto Gaussians. Gaussian-Grouping (2312.00732), N2F2 (2403.10997).
- **Dynamic 3D Gaussians** (2308.09713): exact **local-rigidity / rotation-similarity /
  isometry** losses with `w_ij=exp(−λ_w‖μ_{j,0}−μ_{i,0}‖²)`, **λ_w=2000 (≈2.2 cm), k=20**, and
  an **explicit background-static** term — this *is* our `L_bg_static`/`L_role_rigid`, driven by
  per-Gaussian role relevance instead of a binary mask.
- **VLA (language→action→motion):** RT-2 (2307.15818, 8-D action, 256 bins, co-fine-tune),
  OpenVLA (2406.09246), **π0** (2410.24164, **closest fit**: frozen PaliGemma + *separate
  flow-matching action expert*, continuous chunk H=50, **block-wise attention — actions read
  the VLM, the VLM never attends forward**), RoboFlamingo (2311.01378, frozen VLM + small
  policy head). These validate **freeze Qwen, train a small dynamics/motion head**.
- **Affordance:** VRB (2304.08488, contact GMM + post-contact waypoints, **homography to the
  human-free first frame** — same frame-0 anchoring as us), Where2Act (2101.02692),
  Robo-ABC (2401.07487, cross-category contact transfer).

---

## 4. Module-E design for our system

### 4.1 `GroundedVLMCondition` (Stream B output)
```python
class GroundedVLMCondition:
    role_masks_2d:  Dict[str, Tensor]   # object/source/target/hand/background  [H,W]
    role_relevance_3d: Tensor           # [N_gauss, R]  (sampled at anchor uv)
    region_tokens:  Tensor              # [R, d]  per-role visual-language feature
    motion_tokens:  Tensor              # [Q, d]  task/motion intent
    relation_graph: Optional[...]       # source->object->target
```

### 4.2 Per-Gaussian token (extend `GaussianTokenizer(feature_dim=…)`)
`[FourierPE(μ), q, log s, logit σ, rgb]` **+** `[ sem_latent(16), role_rel(object/source/
target/hand), role_id_emb, pooled_region_feat ]`. `role_rel_i = role_masks_2d[role][u_i,v_i]`.

### 4.3 Dynamics extensions (`igsw/dynamics/transformer.py` DiTBlock)
- keep Gaussian **self-attn** (geometry/physics);
- **Region cross-attn**: Gaussians → `region_tokens` (relevance-biased: add β·role_rel to logits);
- **Motion cross-attn**: Gaussians → `motion_tokens`;
- **mask-conditioned delta gate**: `Δ_i = sigmoid(MLP([g_i, role_rel_i, motion_pool]))·rawΔ_i`,
  **negative-init bias ⇒ default = static** (only task-relevant Gaussians move by default).

### 4.4 Loss set (verified weights; on top of current L_traj pos+vel+rot, L_render aux)
- `L_bg_static = Σ_i (1−relevance_i)·(‖Δμ_i‖² + 0.1‖Δq_i‖²)` — weight **0.5 (warm-up) → 0.1**.
- `L_obj_focus` = relevance-weighted trajectory loss (concentrate direct 3D supervision on
  object/source/target/hand Gaussians) — **0.5**.
- `L_role_rigid` = Dynamic-3DG rigidity *within each role group* (**λ_w=2000, k=20**) — **0.1**;
  `L_iso` (long-term isometry) — **0.05**.

### 4.5 Why this works WITHOUT counterfactual instruction data (the key argument)
The logged failure was that the loss could be minimized by a **scene-only prior** (predict the
one likely future from frame 0), so the instruction carried ~no gradient. Role-grounding fixes
this **structurally**: (1) the instruction becomes the **selector of which Gaussians may move**
and **the motion semantics applied to that subset**, so a scene-only prediction can no longer
minimize the (now relevance-weighted) loss; (2) background is *removed* from the prediction
problem (`L_bg_static`); (3) grounding is inherited from **frozen web-pretrained** models, so
cross-scene generalization is not learned from scarce robot data. Different verbs → different
`motion_tokens` → different deltas on the *same identified subset* = a **within-trajectory,
counterfactual-like** signal even without paired counterfactual clips.

---

## 5. Three implementation versions (verified recipes)

**v7-min (fastest, no Qwen training):** GroundingDINO-tiny (frame0 + phrase list → boxes) →
**SAM2 VideoPredictor** (boxes→masks, propagate across clip) → per-Gaussian relevance at anchor
uv → add as token feature + `L_bg_static`. Run as an **async worker / cache** (frame0 only).
Proxy + `HF_HUB_ENABLE_HF_TRANSFER=0`. → quickly tests: do explicit role masks cut background
drift + raise language-divergence?

**v7-attn (training-free, continuous):** frozen Qwen3-VL, `attn_implementation="eager"`,
`output_attentions=True`; per role phrase slice phrase-token→image-token attention, reshape via
`image_grid_thw`, mean over ~3 calibrated localization heads (2503.06287), upsample (32 px/token
= patch16×merge2), smooth → soft relevance; sample per-Gaussian at `uv//32`. **Verify
`len(image_pos)` (DeepStack adds tokens) before reshape.** → soft, differentiable-friendly bias.

**v7-metaquery (target):** append **32 SEM + 32 MOTION** learnable queries to frozen Qwen3-VL;
read final hidden states → `Z_sem` (→ small head → role masks, optionally distilled from v7-min
teacher) and `Z_motion` (→ Motion cross-attn). **Qwen3-VL-2B gotchas (verified, load-bearing):**
image features injected via `masked_scatter` on `image_token_id=151655` (not concat); **M-RoPE
`position_ids`=(3,B,L)** → extend all 3 axes for queries (`get_rope_index`); keep causal mask;
**no vocab/lm_head expansion** (queries are read, not generated); deepstack [5,11,17]; bf16.
Train only queries + projections + dynamics additions (AFUN-precedented ~tens of M).

---

## 6. Trainable-interface policy (avoid small-data LoRA damage)
- **Layer 1 (default):** Qwen3-VL **fully frozen**; train only MetaQuery tokens + projections +
  dynamics additions (AFUN/π0/RoboFlamingo precedent — small, stable).
- **Layer 2 (only if grounding weak):** LoRA Qwen's top/cross-modal/projector layers **only**,
  with a **distillation** constraint preserving original grounding; never push large 3D-traj
  gradient through the whole VLM.
- **Layer 3 (strong control):** action-conditioning (prototyped, stream7) / counterfactual data;
  final path `language + grounded objects → motion intent → Gaussian dynamics` (π0/RT-2 style).

## 7. Semantic Gaussians: shared frozen projection (not per-scene)
Use a **Gen-LangSplat-style frozen 512→16-D projection** on canonical frame-0 Gaussians (NOT
LangSplat's per-scene AE). Distill role relevance / region features with OpenGaussian losses
(intra-mask smooth + inter-mask contrastive) if we want crisp per-Gaussian role codes.

## 8. Ablation ladder + metrics (NOT just PSNR)
Baseline (text-only) → A (text + role-mask features) → B (+ `L_bg_static`) → C (v7-attn /
MetaQuery, no pooled) → D (full MetaQuery sem+motion + Region/Motion cross-attn + gate).
**Metrics:** (1) **language divergence** (same G0, diff instruction → rollout Δ), (2)
**background motion energy** (low-relevance displacement ↓), (3) **object motion accuracy**
(object-relevant trajectory loss ↓), (4) **role consistency** (apple moves; plate/basket/bg
don't get dragged). Expect A/B to fix background drift fast; C/D to fix language divergence.

## 9. OOM / scale (user point 1)
- **DeepSpeed ZeRO-2** in `train_stream.py` (shard fp32 Adam states across 4 GPUs → frees
  ~14 GB; engine.backward/step replacing manual opt, keeping the rollout-inside-forward).
  stream7 is already ~70/80 GB; Module E adds the grounding interface → ZeRO-2 buys the room.
- Run grounding teacher (GroundingDINO/SAM2) as an **async worker / disk cache** (frame0 only)
  so train GPUs stay on dynamics. Existing levers: `checkpoint_every`, `max_gaussians`, `M`,
  `render_steps`.

## 10. Plan + decision
1. **v7-min** (GroundingDINO+SAM2 async role masks + per-Gaussian relevance + `L_bg_static` +
   `L_obj_focus`) — fastest signal on background-drift + language-divergence. Reuses every hook
   (`return_uv`, `ctrl_idx`, `feature_dim`). Fold in DeepSpeed ZeRO-2 when memory needs it.
2. **v7-metaquery** — the principled target (frozen Qwen + SEM/MOTION queries → masks + motion
   cross-attn). Distill masks from the v7-min teacher to bootstrap.
3. Keep **action-conditioning (stream7)** as the Layer-3 control signal (π0-style).
Engineering hooks already present: `points_to_gaussians(return_uv=True)`, `SCGSRollout(ctrl_idx)`,
`GaussianTokenizer(feature_dim)`, always-on cross-attn in `DiTBlock`, `streaming.py` frame0+instr.

## 11. Citation index (verified arXiv)
MetaQueries 2504.06256 · AFUN 2606.02551 · LISA 2308.00692 · GLaMM 2311.03356 · PixelLM
2312.02228 · VideoGLaMM 2411.04923 · LISA++ 2312.17240 · u-LLaVA 2311.05348 · Groma 2404.13013 ·
Ferret 2310.07704 · Few-Heads-Grounding 2503.06287 · PnP-OVSS 2311.17095 · DAAM 2210.04885 ·
SAM2 (Ravi et al. 2024) · SAM3 2511.16719 · LangSplat 2312.16084 · Gen-LangSplat 2510.22930 ·
OpenGaussian 2406.02058 · Feature-3DGS 2312.03203 · Gaussian-Grouping 2312.00732 · N2F2
2403.10997 · Dynamic-3D-Gaussians 2308.09713 · RT-2 2307.15818 · OpenVLA 2406.09246 · π0
2410.24164 · RoboFlamingo 2311.01378 · VRB 2304.08488 · Where2Act 2101.02692 · Robo-ABC 2401.07487.
(Full per-paper detail + VERIFIED/UNVERIFIED tags + URLs in `research_E1..E4_*.md`.)
