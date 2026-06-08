# Research Brief E4 — 2D language-grounding → 3D-Gaussian conditioning, role-gated motion priors, and the VLA "language→action→motion" precedent

**Purpose:** Verified, citation-checked reference for **Module E** (Language-conditioned Visual Grounding for 3DGS Dynamics). Every paper below was verified on arXiv / project page in June 2026. Zero fabrication; where a claim could not be verified from primary text it is flagged.
**Date:** 2026-06-05
**Scope:** (1) semantic 3D Gaussians — distillation/lifting mechanisms + what is reusable for a *generalizable* (cross-scene) model; (2) role-/affordance-gated dynamics + background-static priors in dynamic-3D / scene-flow; (3) VLA language→action→motion (RT-2, OpenVLA, π0, RoboFlamingo); (4) affordance / contact-motion prediction (VRB, Where2Act, Robo-ABC, AFUN). Then a **concrete, implementable synthesis** for Module E.

**Our setting (one line):** a dense 3DGS set is lifted from frame-0; **each Gaussian stores its frame-0 anchor pixel (u,v)** (`points_to_gaussians(return_uv=True)`). So any 2D mask / relevance map at frame-0 can be sampled per-Gaussian → **per-Gaussian role relevance**. Dynamics predicts per-Gaussian SE(3)+scale deltas, rolled out, supervised by direct 3D trajectories (CoTracker + Pi3) + render loss.

> Cross-refs: this brief is the action/grounding companion to `research_B_semantic_gaussians.md` (semantic-feature distillation details) and `research_E_instruction_visual.md` (VLM grounding strategy / AFUN-style MetaQuery). It does **not** re-derive the LangSplat autoencoder layer sizes (see B); it focuses on the *lift-to-Gaussian* mechanism and the *generalizable* variant, the *motion-gating* losses, and the *VLA/affordance* precedent.

---

## 1. Semantic 3D Gaussians — how 2D features/masks become per-Gaussian semantics, and what is reusable cross-scene

The shared recipe: run a 2D foundation model (CLIP / SAM / LSeg / DINO) per training view → get dense features or masks → **distill them onto Gaussians** either (a) by adding a feature channel to each Gaussian and rendering it differentiably (alpha-compositing the feature, supervised against the 2D feature map), or (b) by contrastive/clustering training of a per-Gaussian instance code, then attaching CLIP at the *instance* level. The crucial design axis for us is **per-scene optimization vs. a shared, frozen, low-dim projection** (the only thing that generalizes).

### 1.1 LangSplat — CVPR 2024
- arXiv: https://arxiv.org/abs/2312.16084 · code: https://github.com/minghanqin/LangSplat · page: https://langsplat.github.io/
- **Mechanism:** SAM 3-level masks (subpart/part/whole) → OpenCLIP ViT-B/16 embedding per masked region → 512-D CLIP per pixel per level. A **scene-wise autoencoder** compresses 512-D → **3-D** latent; the 3-D latent is what is learned per Gaussian and rendered (tiled rasterizer). Querying decodes back to 512-D and matches text.
- **The generalization problem (verified):** the autoencoder is **trained separately per scene** ("trains a scene-wise language autoencoder and then learns language features on the scene-specific latent space"). The 3-D latent space is scene-specific → **the projection is NOT transferable across scenes.** Storing raw 512-D CLIP on Gaussians blows memory up "over 35×" — which is *why* they compress. ⇒ **Not directly reusable** for a generalizable model; only the *idea* (low-dim latent + rasterize feature) is.
- Verified speed claim: 199× faster than LERF at 1440×1080; 84.3% on their localization benchmark.

### 1.2 Feature-3DGS — CVPR 2024 Highlight
- arXiv: https://arxiv.org/abs/2312.03203 · code: https://github.com/ShijieZhou-UCLA/feature-3dgs · page: https://feature-3dgs.github.io/
- **Mechanism:** attach an **arbitrary-dimension feature vector** to each Gaussian and render it with a **Parallel N-dimensional Gaussian Rasterizer**; supervise the rendered feature map against a 2D teacher (SAM, CLIP-LSeg). Optional **convolutional speed-up module** (render a *low-dim* feature then a tiny conv upsamples channels) to dodge the resolution/channel mismatch between RGB images and high-D feature maps. Distillation matches or beats the 2D teacher and is much faster to train/render.
- **Reusable for us:** the *differentiable feature-rasterization + low-dim-then-upsample* trick is exactly how you'd render a **per-Gaussian role-relevance / semantic latent** for an auxiliary supervision signal. The teacher-agnostic design means we can distill *anything* (e.g. our Qwen-derived relevance maps).

### 1.3 N2F2 — ECCV 2024
- arXiv: https://arxiv.org/abs/2403.10997 · pdf: https://www.robots.ox.ac.uk/~vgg/publications/2024/Bhalgat24/bhalgat24.pdf
- **Mechanism:** **Nested** neural feature field — *different subsets of dimensions of one feature vector encode different granularities* (subpart→part→whole), trained with hierarchical/deferred-volumetric supervision against CLIP. At query time it composes relevance across nested levels (weighted) — no separate per-scale fields, parameter-efficient.
- **Reusable for us:** the **"nest multiple semantic scales in one fixed-width vector"** idea lets a single small per-Gaussian semantic latent serve coarse (object) and fine (part) roles — useful because our roles live at object granularity but rigidity priors want part granularity.

### 1.4 OpenGaussian — NeurIPS 2024 (point-level, the most relevant reusable mechanism)
- arXiv: https://arxiv.org/abs/2406.02058 · NeurIPS pdf: https://proceedings.neurips.cc/paper_files/paper/2024/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf
- **Why it matters:** explicitly targets **3D point-level** understanding (not just 2D pixel parsing), fixing weak feature expressiveness + inaccurate 2D↔3D association. Three reusable pieces, with **verified equations**:
  1. **Single-frame instance-feature training from SAM masks (no cross-frame tracking).** Render a per-Gaussian instance feature (6-D), then:
     - Intra-mask smoothing (pull features inside a mask to the mask mean): `L_s = Σ_i Σ_{h,w} B_{i,h,w} · ||M_{:,h,w} − M̄_i||²`
     - Inter-mask contrastive (push different masks' means apart): `L_c = 1/(m(m−1)) Σ_i Σ_{j≠i} 1 / ||M̄_i − M̄_j||²`
  2. **Two-stage codebook discretization.** Coarse stage concatenates feature **and 3D position** `[F∈R^{n×6}; X∈R^{n×3}]` → k1 clusters (k1=64 or 32) so spatially distant objects don't merge; fine stage uses features only → k2 (10 or 5). "Position … is used solely for the codebook construction and is not involved in optimization."
  3. **Instance-level (mask-level) CLIP association, no depth test:** for each 3D instance render it alone → IoU-match to 2D SAM masks with feature validation: `S_ij = IoU(π(M_i), B_j) · (1 − ||M_i − P_j||₁)`; attach the best mask's CLIP. Avoids the lossy per-pixel 2D↔3D averaging that hurts LangSplat-style methods.
- **Reusable for us:** intra/inter-mask losses + position-aware coarse clustering are precisely how to get **clean, instance-consistent per-Gaussian codes** — and our **role masks** can play the role of SAM masks directly (we already know each Gaussian's (u,v)). The mask-level CLIP association avoids depth-occlusion bugs.

### 1.5 Gaussian Grouping — ECCV 2024
- arXiv: https://arxiv.org/abs/2312.00732 · page: https://ymq2017.github.io/gaussian-grouping/
- **Mechanism:** each Gaussian gets a compact **Identity Encoding** (a learned low-D code), supervised through differentiable rendering by **SAM 2D masks** (tracked into multi-view-consistent IDs) **+ a 3D spatial-consistency regularization** (nearby Gaussians → similar identity). Enables train-free 3D removal / recomposition / object-location exchange.
- **Reusable for us:** the **identity-code + 3D-spatial-consistency** pattern is exactly a "soft instance/role label per Gaussian." Our role masks → identity supervision; spatial-consistency reg → role labels don't fragment across a moving object (helps rigidity).

### 1.6 Gen-LangSplat — the cross-scene / generalizable answer (verified)
- arXiv: https://arxiv.org/abs/2510.22930 (html: https://arxiv.org/html/2510.22930)
- **Mechanism:** replaces LangSplat's **per-scene** autoencoder with **one pre-trained generalized autoencoder** trained over many scenes ("pre-trained on CLIP embeddings extracted from SAM-derived masks over ScanNet scenes"), compressing **512-D → 16-D**; "both the encoder and decoder weights are **frozen for all downstream tasks**, allowing immediate deployment on novel scenes with **no additional fine-tuning**." ~2× more efficient than LangSplat (no per-scene AE training).
- **This is the template for our generalizable model:** train a **shared, frozen, low-dim semantic projection ONCE** (e.g. CLIP/SigLIP → 16-D), store the 16-D latent on **canonical (frame-0) Gaussians**, and never re-fit per scene. (Matches the recommendation already in `research_B`.)

**Takeaway for Module E (semantics):** do **not** use a per-scene autoencoder. Use a **frozen shared CLIP→low-D projection** (Gen-LangSplat) on canonical Gaussians; obtain **instance/role consistency** with OpenGaussian-style intra/inter-mask losses + Gaussian-Grouping spatial-consistency, using **our role masks** as the mask source (we already have per-Gaussian (u,v), so no rendering needed for the *mask sampling* — only for optional feature distillation à la Feature-3DGS).

---

## 2. Role-/region-gated dynamics & background-static priors (dynamic-3D / scene-flow)

The literature that gates motion by region splits into (a) **physics-regularized dynamic-Gaussian tracking** (local rigidity + an explicit *background-doesn't-move* loss) and (b) **scene-flow with foreground/background (rigidity) separation**. Both give us the exact loss forms for our background-static / role-rigidity priors.

### 2.1 Dynamic 3D Gaussians — the canonical source of the motion-regularization losses (VERIFIED EQUATIONS)
- arXiv: https://arxiv.org/abs/2308.09713 · page: https://dynamic3dgaussians.github.io/
- **Setup:** Gaussians persist (fixed color/opacity/size) and **move+rotate over time**; 6-DoF dense tracking emerges from analysis-by-synthesis, *with no input flow/correspondence*, thanks to physics-inspired regularizers. Exact, verified forms (k-NN with **k=20**, weights fixed at t=0):
  - **Weight (isotropic Gaussian of init distance):** `w_{i,j} = exp(−λ_w ||μ_{j,0} − μ_{i,0}||²)`, with **λ_w = 2000** ⇒ std ≈ **2.2 cm**.
  - **Short-term local-rigidity:** `L^rigid_{i,j} = w_{i,j} ||(μ_{j,t−1} − μ_{i,t−1}) − R_{i,t−1} R_{i,t}^{−1} (μ_{j,t} − μ_{i,t})||₂`, averaged over `i∈S, j∈knn_{i;k}`.
  - **Local-rotation similarity:** `L^rot = mean_{i,j} w_{i,j} ||q̂_{j,t} q̂_{j,t−1}^{−1} − q̂_{i,t} q̂_{i,t−1}^{−1}||₂` (normalized quaternions).
  - **Long-term isometry (anti-drift):** `L^iso = mean_{i,j} w_{i,j} | ||μ_{j,0}−μ_{i,0}||₂ − ||μ_{j,t}−μ_{i,t}||₂ |`.
  - **Background handling (this is our background-static prior!):** a **background segmentation loss** against a pseudo-GT background mask, plus they "directly apply a loss that **background points shouldn't move or rotate**", and the rigidity/rotation/isometry losses are **restricted to foreground points only.**
- **Direct reuse:** our `L_bg_static` is literally their "background shouldn't move" loss, but **driven by per-Gaussian role relevance** instead of a binary mask: `L_bg = Σ_i (1 − rel_i) · ||Δμ_i||²` (optionally also `||Δq_i||²`). Our `L_role_rigid` = their `L^rigid`/`L^iso` evaluated **within each role group** (k-NN restricted to same-role neighbors), which gives "container moves rigidly, background frozen, object deforms/moves freely."

### 2.2 Neural Scene Flow Prior + rigid scene-flow lineage (foreground/background, rigidity masks)
- Neural Scene Flow Prior (NeurIPS 2021): https://arxiv.org/abs/2111.01253 — an **MLP architecture as an implicit regularizer** for runtime-optimized dense scene flow (no training set); supports dense long-term correspondences by integrating per-point motion. Conceptually it's the "neural-prior" analogue of our learned deformation field.
- Weakly-Supervised Rigid 3D Scene Flow (CVPR 2021): https://arxiv.org/abs/2102.08945 — splits a scene into **rigidly-moving clusters**; enforces *per-cluster rigid SE(3)* and **background ego-motion** separately. This is the scene-flow precedent for **role-rigidity** (treat each role as a rigid-ish cluster) and **background = single rigid/static transform**.
- "Learning Rigidity in Dynamic Scenes" (ECCV 2018): https://link.springer.com/chapter/10.1007/978-3-030-01228-1_29 — predicts a **rigidity mask** (which pixels belong to static vs moving) to estimate camera motion + scene flow. Precedent for predicting a **motion/relevance mask** that gates where flow is allowed.

**Takeaway:** the exact background-static and rigidity loss machinery already exists and is verified (Dynamic 3D Gaussians). Our novelty is **driving the gate continuously from language-derived per-Gaussian role relevance** rather than a geometric/photometric segmentation.

---

## 3. VLA — language → action → motion (the "tokens that condition motion" precedent)

These show *how language conditions a continuous control output* and motivate routing language through a **motion-intent representation** rather than a global instruction vector. Action-tokenization specifics, verified:

### 3.1 RT-2 — VLM + actions-as-text-tokens, co-fine-tuned (VERIFIED)
- arXiv: https://arxiv.org/abs/2307.15818 · CoRL 2023: https://proceedings.mlr.press/v229/zitkovich23a.html
- **Action representation (verified):** an **8-D** action = `[terminate, Δpos_x, Δpos_y, Δpos_z, Δrot_x, Δrot_y, Δrot_z, gripper_extension]`; each continuous dim is **uniformly discretized into 256 bins**; the action becomes a space-joined string, e.g. **`"terminate Δpos_x … gripper"` → `"1 128 91 241 5 101 127"`**.
- **Two backbones differ (verified):** **RT-2-PaLI-X** — integers ≤1000 already have unique tokens, so bins map to integer tokens directly; **RT-2-PaLM-E** — overwrite the **256 least-frequently-used tokens** to hold the action vocabulary.
- **Co-fine-tuning (verified):** trained jointly on **robot trajectories + Internet-scale VL data (VQA, etc.)**; this is what transfers web knowledge → control (emergent generalization to novel objects/commands, chain-of-thought multi-stage reasoning). At inference, output is constrained to valid action tokens.
- **Relevance to us:** RT-2 is the proof that **language + a frozen-ish web-pretrained VLM can drive a structured motor output if you give the model an explicit action representation and co-train.** Our analogue: language → **motion-intent tokens** (pick/place/pour/push/open/close + source/target) that condition the Gaussian deformation, rather than predicting per-Gaussian futures straight from a pooled instruction embedding.

### 3.2 OpenVLA — open 7B VLA, explicit 256-bin tokenizer (VERIFIED)
- arXiv: https://arxiv.org/abs/2406.09246 · page: https://openvla.github.io/ · code: https://github.com/openvla/openvla
- **Architecture (verified):** fused **SigLIP + DINOv2** visual encoder → projector → **Llama-2-7B** backbone that predicts tokenized actions. **Each action dim → 1 of 256 bins**; Llama reserves only 100 special tokens, so they **overwrite the 256 least-used vocabulary tokens** with action tokens. Trained on **970k** Open-X-Embodiment trajectories.
- **Relevance:** confirms the **per-dimension 256-bin discretization + token-overwrite** recipe as the de-facto standard, and that a **dual visual encoder (DINOv2 for geometry + SigLIP for semantics)** is what current VLAs use — mirrors our "geometry stream (Pi3) + semantic stream (CLIP/Qwen)."

### 3.3 π0 (pi-zero) — VLM + flow-matching **action expert** (continuous actions) (VERIFIED EQUATIONS)
- arXiv: https://arxiv.org/abs/2410.24164 · blog: https://www.physicalintelligence.company/blog/pi0 · LeRobot: https://huggingface.co/docs/lerobot/pi0
- **Architecture (verified):** **PaliGemma** VLM backbone + a **separate ~300M "action expert"** for robot **state+action** tokens. Instead of discrete tokens it outputs **continuous actions via conditional flow matching.**
  - Loss: `L^τ(θ) = E ||v_θ(A_t^τ, o_t) − u(A_t^τ | A_t)||²` (v_θ = learned vector field; **τ∈[0,1]** noise level).
  - Predicts an **action chunk of horizon H=50**: `A_t = [a_t, …, a_{t+H−1}]`.
  - Inference: **10 forward-Euler steps**, `A_t^{τ+δ} = A_t^τ + δ·v_θ(A_t^τ, o_t)`, δ=0.1.
  - Observation `o_t = [I_t^1..n, ℓ_t, q_t]` (images, language, proprio).
  - **Block-wise attention:** Block1 (images+lang) bidirectional, Block2 (state) attends to itself, Block3 (actions) attends to the whole sequence; **no block attends forward** — preserves the VLM's pretraining distribution while letting actions read everything.
- **Relevance to us (high):** π0 is the **closest VLA template for a continuous, chunked output** — exactly like our per-Gaussian **continuous SE(3)+scale deltas over a horizon**. Two concrete imports: (i) **flow-matching/diffusion head** as an alternative to a regression head for the deformation field; (ii) **block-wise attention** where dynamics (action) tokens cross-attend to frozen VLM (image+language) tokens but the VLM block can't attend forward → we keep Qwen frozen and uncontaminated while the dynamics transformer reads its grounding tokens.

### 3.4 RoboFlamingo — VLM + explicit recurrent policy head (VERIFIED)
- arXiv: https://arxiv.org/abs/2311.01378 · page: https://roboflamingo.github.io/
- **Mechanism (verified):** built on **OpenFlamingo**; the VLM does single-step vision-language comprehension, an **explicit policy head models sequential history** and predicts **7-DoF EE pose + gripper**; only the policy head (+ light fine-tune) is trained by imitation. Decouples VL understanding from decision-making (enables open-loop, low-compute deployment).
- **Relevance:** the cleanest precedent for our exact factorization — **freeze the big VLM, train a small downstream head/transformer that consumes VLM features and emits the control/motion output.** This is our "Qwen frozen + trainable dynamics transformer" arrangement, validated.

**Takeaway:** across RT-2/OpenVLA/π0/RoboFlamingo the pattern is consistent and supports our plan: **language conditions motion through an explicit action/motion representation**, the big VLM can be **frozen or co-trained but not relied on to emit control directly**, and **continuous chunked outputs (π0) + a separate expert with block-wise cross-attention** is the best match for predicting Gaussian deltas.

---

## 4. Affordance / contact-motion prediction — "predict post-contact 3D/2D motion from image (+language)"

These output exactly the kind of **where + how-it-moves** signal we want as a **motion-intent prior**, and AFUN is the near-exact precedent for our frozen-Qwen MetaQuery design.

### 4.1 VRB (Vision-Robotics Bridge) — CVPR 2023 (VERIFIED OUTPUTS+LOSSES)
- arXiv: https://arxiv.org/abs/2304.08488 · page: https://robo-affordances.github.io/
- **Affordance = contact points + post-contact trajectory**, learned from **human egocentric video** (Ego4D / EPIC-Kitchens). Verified specifics:
  - **Contact:** network predicts **K=5** heatmaps via 2D spatial-softmax → means μ_k of a **GMM** fit to GT contact points (covariances kept **fixed**); contact loss `L_contact = ||μ_i − σ_2D(g^deconv_θ(g^conv_θ(I_t)))||₂`.
  - **Post-contact trajectory:** a **Transformer** predicts **T=5** future wrist waypoints; trained on **relative shifts** (direction of movement, not absolute): `L_traj = ||τ − T_θ(z_t)||₂`.
  - **Human-bias fix (verified):** detect contact frame with a hand-object detector; track wrist for the post-contact path; **project contact + trajectory back onto the first (human-free) frame via homography** `τ = H_t ∘ {h_t}` to remove the human and ego-motion.
- **Relevance to us (high):** VRB *is* "predict where contact happens + how things move next, anchored on a clean first frame." Our frame-0 anchoring is identical in spirit. We can adopt **(contact heatmap → which Gaussians get touched) + (post-contact 2D/3D trajectory → motion-intent token / a coarse target displacement for the manipulated role)**.

### 4.2 Where2Act — ICCV 2021 (VERIFIED)
- arXiv: https://arxiv.org/abs/2101.02692
- **Per-pixel actionable info** for articulated 3D objects: for each point predicts **(a) actionability score, (b) interaction proposals (gripper poses in SE(3)), (c) per-proposal success likelihood**, over 6 short primitives (push/pull/…); learned in SAPIEN. Output is a dense **"act here, this way, it'll move"** map.
- **Relevance:** precedent for a **per-point (→ per-Gaussian) actionability + SE(3) interaction** field; close to attaching a **per-Gaussian "is-this-the-actionable-region + suggested motion direction"** prior, supervised or distilled.

### 4.3 Robo-ABC — ECCV 2024 (VERIFIED)
- arXiv: https://arxiv.org/abs/2401.07487 · page: https://tea-lab.github.io/Robo-ABC/
- **Mechanism:** build an **affordance memory** (contact points) from human video; for a new object, **retrieve** a visually/semantically similar example and **map its contact points via semantic correspondence** (from pre-trained diffusion features) — **zero-shot, no training/part-seg**. +31.6% retrieval accuracy vs end-to-end SOTA; 85.7% real grasp success on cross-category objects.
- **Relevance:** shows **semantic correspondence transfers contact/affordance across categories** — i.e. cross-scene grounding is feasible with frozen features, supporting our **generalizable** (no per-scene fitting) stance. A fallback route to get contact priors when language grounding is uncertain.

### 4.4 AFUN — 2026 (the near-exact precedent for Module E) (VERIFIED)
- arXiv: https://arxiv.org/abs/2606.02551 (html: https://arxiv.org/html/2606.02551)
- **Mechanism (verified):** from a single **RGB-D + language task**, predict a **task-conditional functional mask** (*where*) and a **3D post-contact motion curve** (*how*). Backbone = **Qwen3-VL-8B (frozen)**, **SAM3 (frozen)** for masks, **Sonata (frozen)** 3D encoder. Uses **64 MetaQuery tokens (32 semantic + 32 motion)**: semantic queries → SAM3 to produce the functional mask; motion queries + 3D features → motion decoder → 3D post-contact curve. Only **~32.21M trainable params** (MetaQuery tokens + projection MLP + motion decoder). MetaQuery+projection are pre-initialized by aligning Qwen features to SAM3's text-conditioning space on Visual Genome.
- **Relevance to us (highest):** AFUN is essentially Module E minus the Gaussian dynamics: **frozen Qwen3-VL → learnable semantic tokens (→ role masks) + motion tokens (→ 3D motion) with a tiny trainable interface.** We swap AFUN's single-curve motion decoder for our **per-Gaussian rollout dynamics**, and we feed the **role mask → per-Gaussian role relevance** (we have (u,v) per Gaussian, no rendering needed). Validates the whole "frozen VLM + MetaQuery semantic/motion tokens + small heads" budget.

**Takeaway:** affordance work gives us (i) the **output schema** (contact/where + post-contact motion/how), (ii) a **frozen-VLM MetaQuery interface** that is parameter-cheap (AFUN), and (iii) evidence that **semantic correspondence generalizes contact across categories** (Robo-ABC) — all supporting a generalizable, language-grounded motion prior.

---

## 5. CONCRETE SYNTHESIS — Module E for our Gaussian dynamics model

### 5.0 The three information streams (recap, made concrete)
1. **Geometry stream:** frame-0 / multi-view → Pi3 → dense 3DGS + control Gaussians. Each (control) Gaussian carries `[μ (xyz), q (rot), s (scale), o (opacity), rgb]` **and its frame-0 anchor (u,v) + cam index** (already implemented via `points_to_gaussians(return_uv=True)`).
2. **Grounding stream (Module E):** `(instruction, frame-0 image) → frozen Qwen3-VL + MetaQuery → {role masks, role/region tokens, motion-intent tokens}`. Masks are sampled per-Gaussian through (u,v) → **per-Gaussian role relevance** `rel_i^role`.
3. **Dynamics stream:** Gaussian tokens (augmented with role relevance + region/motion conditioning) → dynamics transformer → per-Gaussian SE(3)+scale deltas → roll out.

### 5.1 (a) Injecting grounding into the dynamics transformer — four mechanisms

**(i) Extra per-Gaussian token features.** Concatenate to each Gaussian token:
```
g_i = [ μ_i, q_i, s_i, o_i, rgb_i,                       # geometry/appearance
        sem_latent_i,                                    # 16-D frozen CLIP→latent (Gen-LangSplat style)
        rel_i^object, rel_i^source, rel_i^target, rel_i^hand,   # per-Gaussian role relevances ∈[0,1]
        role_id_emb_i,                                   # argmax-role embedding (learned)
        region_feat_i ]                                  # mask-pooled VLM region feature for i's role
```
`rel_i^role = mask_role[u_i, v_i]` (soft, from SAM3/Grounded-SAM refine of Qwen output, or Qwen text→image attention). `region_feat_i` = attention-pool of Qwen image tokens **inside that role's mask** (so each Gaussian inherits its role's region embedding).

**(ii) Region cross-attention.** A small set of **region tokens** (one per role: object/source/target/hand/background) with features = mask-pooled VLM region embeddings. Dynamics blocks **cross-attend Gaussian tokens → region tokens** with a **relevance-biased attention**:
```
attn_bias_{i→role} = β · rel_i^role          # Gaussian i attends more to its own role's region token
```
This is the OpenGaussian/Gaussian-Grouping "instance code" idea turned into attention.

**(iii) Motion cross-attention.** **Motion-intent tokens** = Qwen **motion MetaQuery** outputs (AFUN-style; pick/place/pour/push/open/close + source→target relation). Dynamics blocks **cross-attend Gaussian tokens → motion tokens**, so the *manipulated* Gaussians (high `rel^object`) read the motion semantics. Mirrors π0's action expert attending to the (frozen) VLM tokens with **block-wise attention** (dynamics reads VLM; VLM never attends forward → Qwen stays uncontaminated).

**(iv) Mask-conditioned delta gate (the key inductive bias).** The raw predicted delta is **gated** so motion concentrates on task-relevant Gaussians:
```
raw_Δ_i      = DynamicsHead(g_i, region/motion cross-attn)      # Δμ_i, Δlog s_i, Δq_i (as so(3)/tangent)
gate_i       = σ( MLP([ g_i, rel_i^object, rel_i^source, rel_i^target, motion_summary ]) )   # ∈[0,1]
Δ_i          = gate_i · raw_Δ_i
```
Initialize the gate MLP bias **negative** (gate≈0 at start) so the model **defaults to static** and must *earn* motion on the correct Gaussians — this is the single most important trick for fighting background drift early in training. (Equivalently, multiply by `max(rel_i^*)` as a hard prior for v7-minimal.)

> v-minimal (fastest, no Qwen change): skip (ii)/(iii); just use (i) role relevances + (iv) `gate_i = max_role rel_i^role`. v-attn: relevances from frozen-Qwen text→image attention. v-metaquery: full (i)-(iv) with AFUN-style MetaQuery (recommended).

### 5.2 (b) The loss set (with sensible weights)

Supervision = direct 3D trajectories (CoTracker+Pi3) + render. Add three **role-gating** terms. Let `rel_i = max_role∈{object,source,target,hand} rel_i^role` (task relevance), `bg_i = 1 − rel_i`.

| term | formula | weight | purpose |
|---|---|---|---|
| **L_traj** (primary) | `Σ_i ‖μ̂_{i,t} − μ*_{i,t}‖₁` over rollout | **1.0** | match GT 3D Gaussian trajectories |
| **L_render** | photometric (L1+SSIM/LPIPS) of rolled-out render vs future frames | **0.1–1.0** | appearance/geometry sanity |
| **L_bg_static** | `Σ_i bg_i · (‖Δμ_i‖² + γ·‖Δq_i‖²)` , γ≈0.1 | **0.5** (warm-up→0.1) | freeze background (verbatim Dynamic-3DG "bg shouldn't move", driven by relevance) |
| **L_obj_focus** | `Σ_i rel_i · ‖μ̂_{i,t} − μ*_{i,t}‖₁` (re-weighted traj) | **0.5** | concentrate trajectory supervision on the manipulated subset |
| **L_role_rigid** | within each role group r: `Σ_{i∈r} Σ_{j∈knn_i∩r} w_{ij} ‖(μ_{j,t−1}−μ_{i,t−1}) − R_{i,t−1}R_{i,t}^{−1}(μ_{j,t}−μ_{i,t})‖₂` | **0.1** | container/object move rigidly (Dynamic-3DG L^rigid, restricted to same role) |
| **L_iso** (anti-drift, optional) | Dynamic-3DG long-term isometry within role group | **0.05** | rollout stability over ≥10 s |

with `w_{ij} = exp(−λ_w‖μ_{j,0}−μ_{i,0}‖²)`, **λ_w=2000**, **k=20** (Dynamic-3DG values). Notes: keep **L_traj/L_obj_focus on the same scale** (don't let focus dominate); **warm up L_bg_static high then decay** so the model first learns "mostly static," then learns object motion; if soft relevances are noisy, threshold to a hard mask for L_bg_static only.

### 5.3 (c) Why this helps language-conditioning **even without counterfactual data**

The failure mode in our log: with a pooled image+text hidden state, swapping the instruction moved the conditioning by only ~0.026 vs 0.334 text-only (≈12.9× weaker), so the model learned **"look at frame-0 → predict the one future,"** ignoring the instruction (degenerate scene-conditioned prediction). Counterfactual data (same scene, different instruction → different motion) would fix this directly but we don't have it. Role-grounding fixes it **structurally** instead:

1. **Supervision is spatially routed to the correct Gaussians.** `L_obj_focus` + the **delta gate** mean the trajectory gradient lands on the **apple/source/target** Gaussians, not the whole scene. The instruction now controls *which Gaussians are even allowed to move*, so the loss **cannot** be driven down by a scene-only prior that moves everything — it must use the role assignment, which is the only thing that differs when the language differs.
2. **Background is explicitly removed from the prediction problem.** `L_bg_static` + negative-biased gate turn "predict the future of N Gaussians" into "predict the future of the small relevant subset" → far less capacity wasted on (and far less drift from) static background; the language signal is no longer drowned by frame-0 reconstruction.
3. **The language→motion path is short and explicit.** Motion-intent tokens (AFUN/π0-style) + motion cross-attention give a direct route from "pick/place/pour" to the manipulated Gaussians' deltas, rather than hoping a global vector encodes it. Different verbs → different motion tokens → different deltas on the *same* identified object subset — a **within-trajectory** counterfactual-like signal (role identity varies with language) without needing paired counterfactual episodes.
4. **Generalization is inherited, not learned from scarce robot data.** Roles come from a **frozen** web-pretrained grounding stack (Qwen3-VL / CLIP / SAM), and the semantic projection is **shared & frozen** (Gen-LangSplat) → cross-scene grounding doesn't have to be learned from our limited manipulation data; the dynamics head only learns "given these roles + motion intent, how do the relevant Gaussians move" (Robo-ABC evidence that such grounding transfers across categories).

**Net:** even with non-counterfactual data, the instruction stops being an ignorable global vector and becomes the **selector of the moving subset + the motion semantics applied to it** — which is exactly the part of the problem that varies with language.

---

## 6. Verified source list
**Semantic 3DGS:** LangSplat https://arxiv.org/abs/2312.16084 · Feature-3DGS https://arxiv.org/abs/2312.03203 · N2F2 https://arxiv.org/abs/2403.10997 · OpenGaussian https://arxiv.org/abs/2406.02058 · Gaussian Grouping https://arxiv.org/abs/2312.00732 · Gen-LangSplat https://arxiv.org/abs/2510.22930
**Dynamics / scene-flow priors:** Dynamic 3D Gaussians https://arxiv.org/abs/2308.09713 · Neural Scene Flow Prior https://arxiv.org/abs/2111.01253 · Weakly-Sup. Rigid 3D Scene Flow https://arxiv.org/abs/2102.08945 · Learning Rigidity (ECCV'18) https://link.springer.com/chapter/10.1007/978-3-030-01228-1_29
**VLA:** RT-2 https://arxiv.org/abs/2307.15818 (CoRL https://proceedings.mlr.press/v229/zitkovich23a.html) · OpenVLA https://arxiv.org/abs/2406.09246 · π0 https://arxiv.org/abs/2410.24164 · RoboFlamingo https://arxiv.org/abs/2311.01378
**Affordance / contact-motion:** VRB https://arxiv.org/abs/2304.08488 · Where2Act https://arxiv.org/abs/2101.02692 · Robo-ABC https://arxiv.org/abs/2401.07487 · AFUN https://arxiv.org/abs/2606.02551

**Verification status:** all 18 papers verified on arXiv / proceedings / project page (June 2026). Exact equations quoted from primary HTML for: Dynamic 3D Gaussians (rigidity/rotation/isometry losses, λ_w=2000, k=20, background-static loss), π0 (flow-matching loss, H=50, 10 Euler steps, block-wise attention), OpenGaussian (intra/inter-mask losses, two-stage codebook, mask-level CLIP IoU association), VRB (K=5 GMM contact, T=5 trajectory waypoints, homography projection), RT-2 (256-bin 8-D action string, PaLI-X vs PaLM-E token handling), OpenVLA (256-bin per-dim, overwrite least-used tokens), Gen-LangSplat (512→16-D, frozen cross-scene autoencoder), AFUN (frozen Qwen3-VL-8B + 32+32 MetaQuery, 32.21M trainable). **No unverifiable/fabricated claims.** AFUN is a 2026 preprint (arXiv 2606.02551) — verified to exist; treat numbers as preprint-stage.
