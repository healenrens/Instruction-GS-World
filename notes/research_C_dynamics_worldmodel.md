# Research Brief C — Language-conditioned 3D Gaussian Dynamics / Gaussian World Models

**Scope:** Exact, reimplementation-grade specification of how prior work predicts the temporal
evolution of a set of 3D Gaussians, with emphasis on (a) input representation, (b) the exact
dynamics/deformation parameterization, (c) language/action conditioning mechanism, (d) all loss
terms + weights, (e) architecture sizes, (f) code/license. Final section is a synthesis +
concrete recommendation for **Instruct-GS-World**: per-Gaussian SE(3)+scale+opacity+feature
deltas conditioned on a Qwen3-VL hidden state, rolled out autoregressively for 10 s+.

Citations are to paper section/equation and to specific GitHub files. Where a number could not be
confirmed from primary source it is marked **[UNVERIFIED]**.

---

## 0. Notation used throughout

A 3D Gaussian primitive `g_i` is parameterized by:
- `μ ∈ ℝ³` — center (mean)
- `r ∈ ℝ⁴` — rotation quaternion (→ rotation matrix `R`)
- `s ∈ ℝ³` — scale (log-scale internally; `exp` activation)
- `σ ∈ ℝ` (a.k.a. `o`) — opacity (sigmoid activation)
- `c` — color, either RGB or spherical-harmonic (SH) coefficients
- `f ∈ ℝ^d` — optional distilled semantic feature

Covariance `Σ = R S Sᵀ Rᵀ` with `S = diag(s)`. Rendering is the standard 3DGS alpha-blend:
`C(p) = Σ_i c_i α_i Π_{j<i}(1−α_j)`, where `α_i = σ_i · exp(−½ (p−μ'_i)ᵀ Σ'⁻¹ (p−μ'_i))` and `μ', Σ'`
are the 2D-projected mean/covariance.

---

## 1. ManiGaussian (ECCV 2024) — GuanxingLu/ManiGaussian

Sources: arXiv 2403.08321v2 (https://arxiv.org/html/2403.08321v2); GitHub
https://github.com/GuanxingLu/ManiGaussian (files cited inline). License: **MIT**.

### (a) Input representation
- Test time: **single front-view RGB-D** camera at **128×128**. Training: up to **20 cameras**
  for multi-view photometric supervision of the GS render.
- Observation tuple `o(t) = (C(t), D(t), P(t))` where `C` = RGB, `D` = depth, `P ∈ ℝ⁴` = gripper
  proprioception (paper Sec. 3, "Problem Formulation").
- RGB-D is back-projected and **voxelized to 100³** with a 10-channel feature
  (`∈ ℝ^{100³×10}`: RGB, coords, indices, occupancy). A **shallow 3D U-Net** ("representation
  model" `f_φ`) encodes voxels into **128-dim** voxel features `∈ ℝ^{100³×128}`.

### (b) Gaussian regression head and dynamics parameterization
- A per-point **Gaussian regressor** `g_φ` outputs `θ(t) = (μ, c, r, s, σ, f)` with
  **16,384 Gaussian points**. Per-attribute output dims (confirmed in code
  `agents/manigaussian_bc/models_embed.py`, `split_dimensions_with_offset = [3,1,3,4,3,3]` +
  optional SH):
  - `μ`: 3 (position **offset** added to an anchor)
  - `σ`: 1, `sigmoid`
  - `s`: 3, `exp`
  - `r`: 4 quaternion, `F.normalize`
  - `features_dc` (color): 3 (SH band 0 → 12 with higher bands)
  - `f` (semantic): 3
- **Dynamics / deformation field** (paper Eq. 3):
  `(μ_i(t+1), r_i(t+1)) = (μ_i(t) + Δμ_i(t), r_i(t) + Δr_i(t))`
  with `Δθ(t) = p_φ(θ(t), a(t))`. Only **position and rotation** are deformed (scale, opacity,
  color, feature held constant frame-to-frame). The deformation predictor `p_φ` is a
  **fully-connected ResNet (`ResnetFC`)**: in code `gs_deformation_field = ResnetFC(d_in=70,
  d_latent=…, d_lang=…, d_out=3+4)` → outputs `Δμ ∈ ℝ³`, `Δr ∈ ℝ⁴`.

### (c) Language / action conditioning (exact mechanism)
- **Action injection = concatenation.** From `models_embed.py`:
  `if self.use_action: dyna_input = torch.cat((dyna_input, data['action'].repeat(N,1)), dim=-1)`.
  Action is an **8-dim** vector (next-best-pose) concatenated to each Gaussian's feature before the
  deformation MLP.
- **Latent/feature conditioning inside `ResnetFC` = additive per-block injection** (file
  `agents/manigaussian_bc/resnetfc.py`): each residual block does `x = x + lin_z[blk](z)` (and with
  SPADE option, `x = scale_z[blk](z) * x + lin_z[blk](z)` — i.e. FiLM-style). `d_lang` is a declared
  parameter; in the released BC code the **language stream is carried in `z`** (fused upstream), not
  a separate cross-attention.
- **Policy/language for action prediction** uses a **PerceiverIO** multi-modal transformer over the
  rendered/feature scene + tokenized instruction to predict the discretized next-best action
  (translation/rotation/gripper/collision). Language enters the *policy*, while the *world model*
  conditions on the resulting action `a(t)`.

### (d) Loss terms with weights (paper Eqs. 5–9)
- **Geometry / current-frame photometric (Eq. 5):** `L_Geo = ‖C(t) − Ĉ(t)‖₂²`, **λ_Geo = 0.01**.
- **Semantic feature distillation (Eq. 6):** `L_Sem = 1 − cos(F(t), F̂(t))` (cosine distance to a
  **Stable Diffusion** visual-encoder feature field), **λ_Sem = 1e-4**.
- **Action / behavior cloning (Eq. 7):** `L_Act = CE(p_trans, p_rot, p_open, p_collision)`.
- **Future-dynamics prediction (Eq. 8):**
  `L_Dyna = ‖Ĉ(t+1)(a(t), o(t)) − C(t+1)‖₂²`, **λ_Dyna = 0.001** — render the *deformed* future
  Gaussians and match the real next RGB.
- **Total (Eq. 9):** `L = L_Act + λ_Geo L_Geo + λ_Sem L_Sem + λ_Dyna L_Dyna`.

### (e) Architecture/training sizes
- 3D U-Net rep model (shallow), 128-dim voxel features; 16,384 Gaussians; ResnetFC deformation head.
- Training: **100k iters**, batch 2, **LAMB**, lr **5e-4**, **3k-iter warmup** during which the
  deformation predictor is frozen. RLBench: 10 tasks / 166 variations; +13.1% avg success vs SOTA.

### (f) Code / license: MIT (train/agent code). Note the Inria 3DGS rasterizer dependency carries
its own non-commercial research license.

---

## 2. ManiGaussian++ (IROS 2025) — April-Yz/ManiGaussian_Bimanual

Sources: arXiv 2506.19842v1 (https://arxiv.org/html/2506.19842v1); GitHub
https://github.com/April-Yz/ManiGaussian_Bimanual.

### (a) Input
- **RGB-D from up to 6 cameras**, **256×256** training resolution. Same voxelize→U-Net rep pipeline.

### (b) Task-oriented GS + hierarchical dynamics
- Adds **instance logits** `l_i ∈ ℝ³` to each Gaussian: `θ_i(t) = (μ, c, r, s, σ, l)`.
  Instance map rendered `L(p) = Σ_i α_i l_i Π_{j<i}(1−α_j)` (Eq. 3). GT masks from **GroundedSAM**
  prompted by task keywords (distinguishes acting arm / stabilizing arm / target object).
- **Hierarchical leader–follower world model** (Eq. 5):
  - rep `v(t) = f_φ(o(t))`; regressor `θ(t) = g_φ(v(t))`
  - **Leader** (stabilizing arm): `θ_r(t+1) = q_{s,φ}(θ(t), a_s(t), v(t))`
  - **Follower** (acting arm): `θ_l(t+1) = q_{a,φ}(θ_r(t+1), a_s(t), a_a(t), v(t))`
- **Deformation decomposition (Eq. 4):**
  `(μ(t+1), r(t+1)) = (μ(t) + Δμ_s(t) + Δμ_a(t),  r(t) + Δr_s(t) + Δr_a(t))`
  — the two arms' SE(3) effects are **additively composed**; the follower is conditioned on the
  leader's already-deformed state (sequential / hierarchical).

### (c) Conditioning
- Per-arm **action vectors `a_s`, `a_a` concatenated** into the respective deformation MLPs;
  follower additionally consumes leader output `θ_r(t+1)` (prefix-state conditioning).

### (d) Losses (Eq. 10): `L = L_BC + λ_Recon L_Recon + λ_Task L_Task + λ_Pred L_Pred`
- `L_Recon` (Eq. 6): multi-view photometric `‖C(t) − Ĉ(t)‖₂²`
- `L_Task` (Eq. 7): cross-entropy on rendered embodiment/instance masks `−Σ_p Σ_l B̂^l(p) log B^l(p)`
- `L_Pred` (Eq. 8): future prediction `‖Ĉ(t+1) − C(t+1)‖₂²`
- `L_BC` (Eq. 9): bimanual behavior cloning CE over left+right actions
- Specific λ values **not given numerically in the paper** **[UNVERIFIED]** (inherit ManiGaussian's
  order of magnitude: Recon ~1e-2, Pred ~1e-3). +20.2% over SOTA on 10 sim tasks; 60% real.

### (f) Code released; license per repo (MIT-style, check repo) **[UNVERIFIED exact license]**.

---

## 3. GWM — "Towards Scalable Gaussian World Models for Robotic Manipulation" (ICCV 2025)

Sources: arXiv 2508.17600v1 (https://arxiv.org/html/2508.17600v1); project
https://gaussian-world-model.github.io; GitHub https://github.com/Gaussian-World-Model/gaussianwm.
**This is the closest architectural template to our goal** (latent diffusion over Gaussians).

### (a) Input / Gaussian acquisition
- Current scene Gaussians are produced **feed-forward** by **Splatt3R** (built on **MASt3R**:
  stereo point-maps → pixel-aligned Gaussians). Representation
  `G = {x_p, σ_p, Σ_p, C_p}_{p∈P}` (centers, opacity, covariance, SH).
- Uses **2 context frames**.

### (b) Future-Gaussian prediction = **latent diffusion**, not regression
- **3D Gaussian VAE:** Downsample to **N = 512** Gaussians via **Farthest Point Sampling**, then an
  **L-layer cross-attention encoder** maps to latent `X ∈ ℝ^{N×D}`. A self-attention transformer
  **decoder** reconstructs `Ĝ`.
  - VAE loss (Eq. 4): `L_VAE = Chamfer(Ĝ, G) + ‖C(Ĝ) − C(G)‖₁` (Chamfer on centers + L1 render).
- **Latent Diffusion Transformer (DiT)** predicts the *next-frame* latent `x_{t+1}`:
  - **EDM preconditioning / DDPM**. Forward (Eq. 5):
    `p(x_{t+1}^τ | x_{t+1}^0) = N(x_{t+1}^τ; x_{t+1}^0, σ²(τ) I)`. SDE forms Eqs. 6–7.
  - Denoising objective (Eq. 10):
    `L(θ) = E[ ‖ F_θ(c_in^τ x_{t+1}^τ, y_t^τ) − (1/c_out^τ)(x_{t+1}^0 − c_skip^τ x_{t+1}^τ) ‖₂² ]`
    (standard EDM target). Noise schedule: `log σ ∼ N(−0.4, 1.2²)`, `σ_data = 0.5`.
  - DiT blocks use **AdaLN** for timestep/noise conditioning and **RMSNorm**.

### (c) Conditioning
- **Robot action conditions via cross-attention** (action → K,V in DiT cross-attn layers).
- Timestep/noise via **AdaLN modulation**. **No language conditioning** in GWM (action+vision only)
  — we would add language ourselves.
- RL rollout horizon **RL = 12** steps; used both as a representation-learning auxiliary and as a
  neural simulator for model-based RL. +16.25% over baselines; +30% on a real diffusion policy.

### (e) Sizes: DiT layer/head counts in Appendix B.2, not in main text **[UNVERIFIED exact dims]**;
N=512 latent tokens, latent dim D unspecified in HTML.

### (f) Code released; license per repo **[UNVERIFIED]**.

---

## 4. Dynamic 3D Gaussians (Luiten et al., 3DV 2024) — JonathonLuiten/Dynamic3DGaussians

Sources: arXiv 2308.09713; site https://dynamic3dgaussians.github.io; GitHub
https://github.com/JonathonLuiten/Dynamic3DGaussians (`train.py`). License: **MIT** (core code;
Inria rasterizer non-commercial).

### (a) Input / motion parameterization
- Multi-view video, **per-timestep free optimization**: each Gaussian's `μ_t, r_t` are *free
  variables per frame* (NOT an MLP field). **Persistent across time:** color, opacity, scale, and
  a background/foreground segmentation — these are frozen after frame 0. Motion = directly optimized
  trajectory `{μ_{i,t}, q_{i,t}}`. Tracking is *emergent*: a Gaussian = a physical point, so its
  trajectory is the 6-DOF track (no flow/correspondence input).

### (b) Physics regularizers (exact equations; k = 20 kNN, neighbors fixed from frame 0)
- **Local rigidity (short-term, between t−1 and t):**
  `L^rigid_{i,j} = w_{i,j} ‖ (μ_{j,t−1} − μ_{i,t−1}) − R_{i,t−1} R_{i,t}⁻¹ (μ_{j,t} − μ_{i,t}) ‖₂`
  with **weight `w_{i,j} = exp(−λ_w ‖μ_{j,0} − μ_{i,0}‖₂²)`, λ_w = 2000** (σ ≈ 2.2 cm).
- **Local-rotation similarity:**
  `L^rot = (1/k|S|) Σ_i Σ_{j∈knn(i)} w_{i,j} ‖ q̂_{j,t} q̂_{j,t−1}⁻¹ − q̂_{i,t} q̂_{i,t−1}⁻¹ ‖₂`
  (neighbors must rotate the same as the center).
- **Long-term local-isometry:**
  `L^iso = (1/k|S|) Σ_i Σ_{j∈knn(i)} w_{i,j} | ‖μ_{j,0}−μ_{i,0}‖₂ − ‖μ_{j,t}−μ_{i,t}‖₂ |`
  (neighbor distances preserved vs. frame 0).
- **Code loss weights (`train.py` `loss_weights`):**
  `{im:1.0, seg:3.0, rigid:4.0, rot:4.0, iso:2.0, floor:2.0, bg:20.0, soft_col_cons:0.01}`.
  - `im = 0.8·L1 + 0.2·(1−SSIM)`; `seg` similarly on segmentation.
  - `floor = clamp(y, min=0).mean()` (keep above ground), `bg` = L1 anchoring background
    points+rot to init, `soft_col_cons` = L1 color-temporal consistency.
- Optimizer: per-frame initialization carries forward previous frame's params (forward Euler-ish),
  then gradient-descent that frame.

### Relevance: gives the **canonical physics priors** (rigidity/rotation/isometry) we should reuse as
*rollout regularizers* to stop autoregressive drift, even though we use an MLP/transformer field.

---

## 5. Deformable-3DGS (Yang et al., CVPR 2024) and 4D-GS / 4DGaussians (Wu et al., CVPR 2024)

### 5A. Deformable 3D Gaussians — arXiv 2309.13101 (ingra14m/Deformable-3D-Gaussians, MIT)
- **Canonical Gaussians + a single deformation MLP** `F_θ`.
- Architecture: **D = 8 FC layers, width W = 256, ReLU**, NeRF-style **skip connection at layer 4**;
  then a small head outputs the deltas.
- Input positional encoding (Eq. 5): `γ(p) = (sin(2^k π p), cos(2^k π p))_{k=0}^{L−1}`.
  Frequencies: **position L_x = 10**; **time L_t = 6** (synthetic) / **L_t = 10** (real).
- Field (Eq. 4): `(δx, δr, δs) = F_θ(γ(sg(x)), γ(t))`, with **stop-gradient `sg` on position** so the
  canonical centers don't chase the deformation. Deformed Gaussian: `G(x+δx, r+δr, s+δs, σ)` — note
  **opacity & color are NOT deformed**.
- **Annealing Smooth Training (AST)** (Eq. 6): add Gaussian noise to the time input,
  `X(i) = N(0,1)·β·Δt·(1 − i/τ)`, **β = 0.1**, **τ = 20k**, `Δt` = mean frame interval — linearly
  annealed; smooths early training, removed later.
- Loss (Eq. 7): `L = (1−λ) L1 + λ L_{D-SSIM}`, **λ = 0.2**.

### 5B. 4D Gaussian Splatting (4DGaussians) — arXiv 2310.08528 (hustvl/4DGaussians, MIT)
- **Canonical Gaussians + HexPlane spatiotemporal voxel encoder + tiny MLP decoder.**
- **HexPlane** (Eq. 10):
  `f_voxel = ⋃_l Π_{(i,j)} interp(R_l(i,j))`, planes
  `(i,j) ∈ {(x,y),(x,z),(y,z),(x,t),(y,t),(z,t)}` — 6 multi-res 2D feature grids.
  - Base plane resolution **64×64**, multi-res upsampling **L = {2,4,8}**, **h = 32** channels/plane;
    plane feature `R_l(i,j) ∈ ℝ^{h × lN_i × lN_j}`. Query = **bilinear interp** of 4 nearby cells,
    **product (∏) across the 6 plane-pairs**, **union (⋃) over resolutions**.
- **Multi-head decoder** (Eq. 11): `(Δχ, Δr, Δs) = g(f_voxel)` via separate tiny MLP heads
  (`φ_x, φ_r, φ_s`), **hidden dim 64**. Deformed (Eq. 12): `(χ',r',s') = (χ+Δχ, r+Δr, s+Δs)`;
  final `S' = {χ', s', r', σ, C}` (opacity/color again static).
- Loss (Eq. 13): `L = ‖Ĉ − C‖ (L1 color) + L_tv (grid total-variation)`. LRs: grid 1.6e-3→1.6e-5,
  MLP 1.6e-4→1.6e-6; 20k iters with 3k static warmup. 82 FPS @ 800×800 on RTX 3090.

**Takeaway for us:** both prove the `(x,y,z,t) → (Δμ,Δr,Δs)` field works; HexPlane gives a fast,
position-indexed spatiotemporal encoder we can adapt for 3D positional encoding of Gaussians.

---

## 6. GaussianFlow (ICLR 2025) — Zerg-Overmind/GaussianFlow

Sources: arXiv 2403.12365v2 (https://arxiv.org/html/2403.12365v2); project
https://zerg-overmind.github.io/GaussianFlow.github.io.

- **Goal:** supervise 3D Gaussian *dynamics* (translation+rotation+scale) directly with **2D optical
  flow**, analytically and differentiably.
- Per-Gaussian 2D motion via covariance "whitening" of the projected Gaussian. With 2D mean `μ_{i,t}`
  and 2D covariance `Σ_{i,t} = B_{i,t} B_{i,t}ᵀ`:
  - Map a pixel into canonical Gaussian coords (Eq. 3): `x̂_{t1} = B_{i,t1}⁻¹ (x_{t1} − μ_{i,t1})`.
  - Push to next frame (Eq. 4): `x_{i,t2} = B_{i,t2} x̂_{t1} + μ_{i,t2}` (preserves Mahalanobis dist).
  - Per-Gaussian flow (Eq. 5): `flow^G_{i,t1t2} = x_{i,t2} − x_{t1}`.
- **Pixel-level Gaussian flow = α-weighted sum** over the K Gaussians covering the pixel (Eqs. 6–8):
  `flow^G_{t1t2} = Σ_{i=1}^K w_i (x_{i,t2} − x_{t1})`, with
  `w_i = T_i α_i / Σ_i T_i α_i` (normalized alpha-composite weights; `T_i` = transmittance).
- **Flow loss (Eq. 10):** `L_flow = ‖ flow^o_{t1t2}(x_{t1}) − flow^G_{t1t2} ‖`, weight
  **λ_flow = 1.0** (4D generation) / **0.5** (NVS). CUDA-implemented, end-to-end differentiable.

**Takeaway:** if we ever have 2D flow/video supervision, this is the exact analytic bridge from 2D
flow to per-Gaussian SE(3)+scale deltas — directly supervises our delta field without 3D GT.

---

## 7. L4GM and language→3D-future for robotics (3D-VLA, WorldVLA)

### 7A. L4GM (NeurIPS 2024) — arXiv 2406.10324 (NVIDIA, license: NVIDIA non-commercial)
- **Feed-forward 4D LRM.** Base = **LGM** asymmetric U-Net mapping multiview → **pixel-aligned
  Gaussians**: 6 down blocks ch `[64,128,256,512,1024,1024]`, mid `[1024]`, 5 up blocks →
  **128×128×4 = 65,536 Gaussians/frame**.
- **Temporal self-attention** inserted **after cross-view attention** in each U-Net block: reshape
  `(BV)(THW)C` for attention then back, so per-frame Gaussian sets stay temporally consistent.
- Input: **4 multiview images** (ImageDream from frame 0) + **Plücker ray** camera embeddings,
  256×256, replicated across T.
- **Interpolation model** upsamples low→high fps: takes two multiview sets 3 frames apart, emits
  **4 Gaussian sets** (2 inserted timesteps) at 24 FPS.
- Loss: `L = L_RGB + L_Mask` per t, all views; `L_RGB = MSE + λ·LPIPS`, `L_Mask = MSE(alpha)`.
- **Purely video-conditioned, no language.** Relevant only as the "per-frame Gaussian set + learned
  interpolation" pattern (vs. our deformation-field approach).

### 7B. 3D-VLA (ICML 2024) — arXiv 2403.09631 (UMass, code released)
- A **3D-LLM backbone** + **interaction tokens**; **embodied diffusion models** generate **goal
  RGB image and goal point cloud**, aligned into the LLM via a projector. Predicts a *goal 3D state*
  (point cloud), not a per-step Gaussian deformation. 2M 3D-language-action pairs. **Language is the
  prefix to the LLM**, the diffusion goal-generators are conditioned on LLM output embeddings.

### 7C. WorldVLA (2025) — arXiv 2506.21539 (alibaba-damo-academy; RynnVLA-002 successor)
- **Autoregressive unified VLA + world model in one LLM.** Three tokenizers (image, text, action),
  **shared vocabulary**. World-model head predicts **future image tokens** from action+image; action
  head predicts next actions. **Action-attention masking** so action chunks don't accumulate error.
  2D-token future (not 3D Gaussians) — informative for the *autoregressive + masking* recipe.

---

## 8. Cross-method comparison (parameterization of dynamics)

| Method | Future repr. | Field form | Deformed attrs | Conditioning | Loss anchor |
|---|---|---|---|---|---|
| ManiGaussian | next-frame Gaussians | FC-ResNet `p_φ(θ,a)` per-Gaussian | Δμ, Δr only | **action concat** + latent FiLM | render L2 future (λ=1e-3) |
| ManiGaussian++ | next-frame | leader-follower FC, additive 2-arm | Δμ, Δr (Δ_s+Δ_a) | per-arm action concat + prefix state | render L2 future |
| GWM | next-frame latent | **latent DiT diffusion** | full latent (all attrs) | **action cross-attn** + AdaLN | EDM denoise + Chamfer/render |
| Dyn3DG | per-frame free vars | none (free opt) | μ,r free; rest frozen | none (recon only) | rigid/rot/iso physics |
| Deformable-3DGS | canonical+t | MLP `F_θ(γx,γt)` D8 W256 | Δx,Δr,Δs | time PE | L1+DSSIM |
| 4DGaussians | canonical+t | HexPlane + tiny MLP | Δχ,Δr,Δs | (x,y,z,t) grid | L1 + grid TV |
| GaussianFlow | per-frame | any (supervision only) | t,r,s via 2D flow | — | α-weighted flow loss |
| L4GM | per-frame sets | feed-fwd U-Net + temporal attn | full (re-predicted) | multiview+Plücker | MSE+LPIPS+mask |

Key splits:
- **Regression (ManiGaussian, Deformable, 4DGS)** vs **diffusion (GWM)** vs **free-opt (Dyn3DG)**.
- **Per-Gaussian delta field** (ManiGaussian, Deformable, 4DGS) vs **re-predict whole set**
  (L4GM, GWM-VAE).
- **What deforms:** almost everyone deforms only **μ, r** (and 4DGS/Deformable add **s**). Nobody in
  the GS-dynamics line deforms **opacity/feature** per step except via full re-prediction (GWM).
- **Conditioning:** concat (ManiGaussian) is simplest; **cross-attention (GWM)** scales best to a
  rich language/VLM embedding; FiLM/AdaLN is the cheap middle ground.

---

## 9. SYNTHESIS & RECOMMENDATION for Instruct-GS-World

**Target:** predict per-Gaussian **SE(3) + scale + opacity + feature deltas** conditioned on a
**Qwen3-VL (2B) hidden-state** language embedding, **autoregressive rollout ≥ 10 s**, **variable-size
Gaussian set**.

### 9.1 Adopt a transformer **delta field**, not free-opt and not full re-prediction
- Free-opt (Dyn3DG) cannot generalize / be conditioned. Full re-prediction (L4GM/GWM-VAE) loses
  identity correspondence (bad for 10 s autoregression and for feature persistence). The
  **per-Gaussian delta field** (ManiGaussian/Deformable/4DGS lineage) preserves Gaussian identity
  across the rollout, which is exactly what stable long-horizon autoregression needs.

### 9.2 Exact recommended parameterization (per Gaussian i, step t→t+1)
Predict a **6-DoF residual in the local frame** plus log-scale / logit deltas:
```
[ξ_i, δlogs_i, δσ̃_i, δf_i] = D_θ( token_i , z_lang , a_t )        # transformer field
ξ_i = (v_i ∈ ℝ³, ω_i ∈ ℝ³)            # se(3) twist
μ_{t+1} = μ_t + R_t · v_i              # translate in body frame (or world: μ_t + v_i)
R_{t+1} = R_t · Exp(ω_i)              # right-multiply rotation via matrix exp of ω (SO(3))
s_{t+1} = exp( log s_t + δlogs_i )    # multiplicative scale → stays positive
σ_{t+1} = sigmoid( logit(σ_t) + δσ̃_i )# opacity delta in logit space
f_{t+1} = f_t + δf_i                  # additive feature delta (optionally tanh-bounded)
```
Rationale vs prior art:
- **`Exp(ω)` (Lie-algebra) instead of `r+Δr` (ManiGaussian Eq.3 / 4DGS Eq.12).** Additive-quaternion
  deltas are non-geometric and drift over a 10 s rollout; right-multiplied `Exp(ω)` is a proper SO(3)
  composition, stays on the manifold, and is the correct "SE(3) delta" the spec asks for.
- **Multiplicative scale / logit-opacity** keep parameters in valid ranges across many steps (additive
  Δs as in 4DGS can go negative after long rollout).
- Deform **all** attributes (the spec requires opacity+feature), unlike ManiGaussian which freezes
  s,σ,c,f. Keep `δ` small via output `tanh` scaling + a delta-magnitude penalty.

### 9.3 Variable-size Gaussian set in a transformer (tokenization / attention / PE)
- **One token per Gaussian.** Token = `Linear([μ, log s, r(6D), σ̃, f, c_lowSH])` → model dim d_m.
  Use the **continuous 6D rotation representation** (Zhou et al.) for the input encoding (not raw
  quaternion) for smoothness.
- **3D positional encoding by Gaussian center** (this is the key choice): Fourier features of `μ`,
  `PE(μ) = [sin(2^k π B μ), cos(2^k π B μ)]` (like Deformable-3DGS Eq.5 / NeRF), **added** to the
  token. This injects spatial locality without imposing an order → permutation-equivariant.
- **Attention over the variable set:** standard self-attention is O(N²). For N up to ~16k (cf.
  ManiGaussian) prefer (a) **k-NN / windowed local attention** in 3D (neighbors define the
  "rigidity" graph anyway), or (b) **serialized neighborhoods** (Point-Transformer-V3 Hilbert/Z-order
  curve) for linear-ish cost, or (c) **FPS down-sample to N=512 latent tokens + cross-attention**
  exactly like **GWM's VAE** if N is huge. Padding+attention-mask handles variable N within a batch.
- **No global learned positional index** — position must come *only* from `PE(μ)`, so the field
  generalizes to arbitrary Gaussian counts and reorderings.

### 9.4 Conditioning on a Qwen3-VL (2B) hidden state — **use cross-attention (GWM-style), not concat**
- Take the **last-layer (or penultimate) hidden states** of Qwen3-VL → sequence
  `H ∈ ℝ^{L_tok × 2048}` (2B hidden ≈ 2048; confirm exact width per checkpoint). **Project**
  `z_lang = LinearProj(H) ∈ ℝ^{L_tok × d_m}` and use it as **K,V in cross-attention** blocks
  interleaved with the Gaussian self-attention (this is GWM's mechanism — keys/values from condition
  into DiT). This scales to rich, multi-token VLM embeddings far better than ManiGaussian's
  single-vector concat.
- **Action / robot state** (if present): inject via **AdaLN** modulation (GWM) or as extra
  cross-attn tokens. Diffusion timestep (if you go diffusion) → AdaLN.
- Mean-pool `z_lang` to a global vector additionally for **FiLM** on the delta-head — cheap global
  gain that complements token-level cross-attn.

### 9.5 Regression vs diffusion head
- **Start with deterministic regression** (ManiGaussian/Deformable simplicity, easy to debug,
  cheap rollout). Loss = render-based future + delta regularizers (below).
- **If multimodality matters** (a language instruction admits several plausible futures), upgrade to
  **GWM's latent-DiT diffusion** over the delta tokens with **EDM preconditioning** (Eq. 10:
  `L = ‖F_θ(c_in x^τ, cond) − (1/c_out)(x⁰ − c_skip x^τ)‖₂²`, `log σ∼N(−0.4,1.2²)`, `σ_data=0.5`).
  Diffuse the *deltas*, not absolute Gaussians, to keep identity.

### 9.6 Losses (recommended set + starting weights)
1. **Future photometric render** (primary, like ManiGaussian Eq.8 / GWM render):
   `L_render = 0.8·L1 + 0.2·(1−SSIM)` between render of predicted `G_{t+1}` and GT image, **w=1.0**.
2. **Geometry/depth** if available: depth-render L1, **w=0.5**.
3. **Physics rollout regularizers (Dyn3DG, exact)** to prevent autoregressive drift — apply on
   *predicted* motion with k=20 kNN (neighbors from current frame), weight
   `w_{ij}=exp(−2000·‖μ_{j,0}−μ_{i,0}‖²)`:
   - rigidity **w=4.0**, rotation **w=4.0**, isometry **w=2.0** (Dyn3DG code values).
4. **Delta-magnitude / smoothness prior:** `‖v‖²+‖ω‖²+‖δlogs‖²+‖δσ̃‖²+‖δf‖²`, **w=1e-2** (stops
   runaway deltas over 10 s).
5. **Feature consistency** (if distilling Qwen-VL/SD features): cosine, **w=1e-4** (ManiGaussian Eq.6).
6. **(Optional) GaussianFlow** if 2D flow GT exists: `L_flow=‖flow^o−flow^G‖`, **w=0.5**.
7. **(If diffusion) EDM denoise** replaces #1 as the training objective; keep #3,#4 as auxiliary on
   the x0-prediction.

### 9.7 Autoregressive 10 s+ stability checklist (the hard part)
- **Train with multi-step rollout (TBPTT)**, not just 1-step — supervise rendered frames at t+1..t+K
  (K≥4) so errors that compound are penalized (ManiGaussian only does 1-step; that is insufficient
  for 10 s).
- **Lie-algebra SE(3) + multiplicative scale + logit opacity** (9.2) for manifold-correct accumulation.
- **Physics regularizers (9.6 #3)** are the main anti-drift tool; isometry vs **frame-0** anchors
  long-term shape.
- **Renormalize quaternions/clip scales each step**; optionally **re-densify/prune** opacity→0
  Gaussians during rollout (carry a "death" via opacity logit going very negative).
- Consider **scheduled sampling**: mix GT and predicted previous states early in training.

### 9.8 One-paragraph recommendation
Adopt a **per-Gaussian delta field implemented as a transformer** (token-per-Gaussian, 3D Fourier PE
of center, local/serialized attention, FPS+cross-attn fallback for huge sets). Parameterize the
dynamics as **`μ←μ+R·v`, `R←R·Exp(ω)`, `s←s·exp(δ)`, `σ←sigmoid(logit σ+δ)`, `f←f+δ`** (SE(3) on the
manifold + multiplicative/logit deltas). Condition on the **Qwen3-VL hidden-state sequence via
cross-attention (GWM-style K/V)** plus a FiLM/AdaLN global term for action/timestep. Train with
**multi-step rollout**, a **render loss (0.8 L1 + 0.2 SSIM)**, **Dyn3DG rigidity/rotation/isometry
priors (4/4/2, λ_w=2000, k=20)**, and a **delta-magnitude penalty (1e-2)**; start deterministic and
upgrade the delta head to **EDM latent diffusion** only if the language-conditioned future is
genuinely multimodal.

---

## 10. Source index
- ManiGaussian: arXiv 2403.08321v2; GitHub GuanxingLu/ManiGaussian (`agents/manigaussian_bc/
  models_embed.py`, `resnetfc.py`). MIT.
- ManiGaussian++: arXiv 2506.19842v1; GitHub April-Yz/ManiGaussian_Bimanual.
- GWM: arXiv 2508.17600v1 (ICCV'25); GitHub Gaussian-World-Model/gaussianwm.
- Dynamic 3D Gaussians: arXiv 2308.09713 (3DV'24); GitHub JonathonLuiten/Dynamic3DGaussians
  (`train.py` `loss_weights`). MIT.
- Deformable-3DGS: arXiv 2309.13101 (CVPR'24); GitHub ingra14m/Deformable-3D-Gaussians. MIT.
- 4DGaussians (Wu et al.): arXiv 2310.08528 (CVPR'24); GitHub hustvl/4DGaussians. MIT.
- GaussianFlow: arXiv 2403.12365v2 (ICLR'25); GitHub Zerg-Overmind/GaussianFlow.
- L4GM: arXiv 2406.10324 (NeurIPS'24); NVIDIA, non-commercial license.
- 3D-VLA: arXiv 2403.09631 (ICML'24); GitHub UMass-Embodied-AGI/3D-VLA.
- WorldVLA: arXiv 2506.21539; GitHub alibaba-damo-academy/RynnVLA-002.
- Point-Transformer-V3 (serialization/3D PE design reference) for variable-set attention.
