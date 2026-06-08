# Research Brief B: Language-Aligned Semantic Features for 3D Gaussians

**Purpose:** Exhaustive technical reference for reimplementation. Zero simplifications.  
**Date:** 2026-06-05  
**Topic:** Attaching language-aligned semantic features to 3D Gaussians for natural-language localization/addressing of objects and regions.

---

## 1. LangSplat (CVPR 2024 Highlight)

**Paper:** "LangSplat: 3D Language Gaussian Splatting" — Qin et al., CVPR 2024, pp. 20051–20060  
**arXiv:** https://arxiv.org/abs/2312.16084  
**Code:** https://github.com/minghanqin/LangSplat

### 1a. SAM Multi-Scale Mask Extraction

LangSplat uses SAM's **ambiguity-aware mode** to produce three hierarchical segmentation maps per image:

- **Subpart level** M^s: smallest semantic units (e.g., "wheel of a car")
- **Part level** M^p: intermediate groups (e.g., "door of a car")
- **Whole level** M^w: complete objects (e.g., "car")

SAM inherently produces 3 masks per point prompt — these correspond to whole/part/subpart granularity within the object hierarchy. The process:

1. Feed a **regular grid of 32×32 point prompts** into SAM → obtains initial mask sets M_0^s, M_0^p, M_0^w (one per ambiguity level per point).
2. **Filter redundant masks** within each set using: predicted IoU score, stability score, and overlap rate between masks.
3. Each filtered set independently performs full-image segmentation at its semantic level, resulting in three final segmentation maps: M^s, M^p, M^w (full-image, non-overlapping per level).

*Implementation note:* The repository uses the `segment-anything-langsplat` fork (a patched SAM) which exposes all 3 mask levels.

### 1b. CLIP Feature Extraction Per Mask

For each training image I_t, pixel v, and semantic level l ∈ {s, p, w}:

```
L_t^l(v) = V(I_t ⊙ M^l(v))
```

Where:
- V(·) is the **OpenCLIP ViT-B/16** image encoder
- ⊙ is element-wise (pixel-wise) masking — the image is multiplied by the binary mask for the region containing pixel v
- Output: **512-dimensional** CLIP embedding (D = 512)

Each pixel v thus gets a 512-d embedding per scale, extracted from the masked crop centered on that pixel's containing mask region.

### 1c. Scene-Specific Autoencoder

**Architecture** (from `autoencoder/model.py`):

**Encoder E:** 512 → 256 → 128 → 64 → 32 → **3** (d = 3 by default)
- First layer: Linear(512, 256), no BN/activation
- Subsequent layers: BatchNorm1d → ReLU → Linear
- Final: L2 normalization: `x = x / x.norm(dim=-1, keepdim=True)`

**Decoder Ψ:** 3 → 16 → 32 → 64 → 128 → 256 → 256 → **512**
- First layer: Linear(3, 16), no activation
- Subsequent layers: ReLU → Linear
- Final: L2 normalization

**Training loss** (from `autoencoder/train.py`):

```
L_ae = L2(Ψ(E(f)), f) + 0.001 × L_cosine(Ψ(E(f)), f)

L2     = mean((output - gt)^2)
L_cos  = 1 - cosine_similarity(output, gt, dim=0).mean()
```

**Training hyperparameters:**
- Optimizer: Adam, lr = 0.0001 (default; paper command uses lr = 0.0007)
- Batch size: 64 (train), 256 (eval)
- Epochs: 100
- Checkpoint: best model saved based on eval loss

**Critical property:** The autoencoder is trained **per scene** using that scene's own CLIP features. The latent dimension d = 3 was chosen to enable RGB visualization of the language field (each latent dim → one color channel).

*Exact encoder/decoder dims from training command:*
```bash
python train.py --encoder_dims 256 128 64 32 3 \
                --decoder_dims 16 32 64 128 256 256 512 \
                --lr 0.0007
```

### 1d. Language Feature as Extra Gaussian Channels + Alpha-Blended Rendering

Each 3D Gaussian i is augmented with **3 extra attribute vectors**: {f_i^s, f_i^p, f_i^w}, where f_i^l ∈ R^d (d = 3 by default), representing the per-scale language latent features.

These are optimized jointly with the rest of the Gaussian parameters (position, covariance, opacity, color SH coefficients), but **only during a second training stage** after the geometry is initialized from a pretrained 3DGS checkpoint.

**Rendering** (identical to standard alpha-compositing in 3DGS, applied per feature channel):

```
F^l(v) = Σ_{i ∈ N} f_i^l · α_i · Π_{j=1}^{i-1}(1 - α_j),   l ∈ {s, p, w}
```

Where:
- α_i = o_i · G_i^{2D}(v): product of Gaussian opacity o_i and the 2D projected Gaussian value at pixel v
- G_i^{2D}(v) = exp(-0.5 (v - μ_i^{2D})^T Σ_i^{2D-1} (v - μ_i^{2D}))
- N: ordered set of Gaussians covering pixel v (front-to-back)
- Result: rendered low-dimensional feature map F^l(v) ∈ R^d per pixel

**Tile-based rasterization** (same CUDA kernel as original 3DGS, applied to d extra channels).

**Important:** No densification is performed during language feature training (only in the geometry pretraining phase). The language training uses only L1 loss (no SSIM).

### 1e. Training Loss for Language Gaussians

```
L_lang = Σ_{l ∈ {s,p,w}} Σ_{t=1}^{T} d_lang(F_t^l(v), H_t^l(v))
```

Where H_t^l(v) = E(L_t^l(v)) is the encoded (3-d latent) ground-truth CLIP embedding for pixel v in image t at scale l.

**Implementation** (`train.py`): Pure masked L1 loss:
```python
L = l1_loss(language_feature * mask, gt_language_feature * mask)
```
Weight = 1.0 (no lambda weighting). Mask is the valid-feature binary mask per pixel.

### 1f. Open-Vocabulary Query at Test Time

**Step 1:** Encode the text query using OpenCLIP ViT-B/16 text encoder → φ_qry ∈ R^512  
**Step 2:** Render F^l for all 3 scales → decode each pixel: φ_img = Ψ(F^l(v)) → 512-d CLIP space  
**Step 3:** Compute relevancy score per pixel using **softmax-based relevancy** (from LERF formulation):

```
relevancy(v) = min_i [ exp(φ_img · φ_qry) / (exp(φ_img · φ_qry) + exp(φ_img · φ_canon^i)) ]
```

Where φ_canon^i are embeddings of **4 canonical phrases**: {"object", "things", "stuff", "texture"}.

The min over canonical phrases acts as a denominator regularizer, preventing the relevancy score from being trivially high in uninformative regions.

**Step 4:** Produce three relevancy maps (one per scale s, p, w). Apply **mean convolution filter (size 20)** to smooth each map.

**Step 5:** Select the scale with the **highest maximum smoothed relevancy score** → binary mask via threshold 0.4.

---

## 2. Feature-3DGS (CVPR 2024 Highlight)

**Paper:** "Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields" — Zhou et al., CVPR 2024  
**arXiv:** https://arxiv.org/abs/2312.03203  
**Code:** https://github.com/ShijieZhou-UCLA/feature-3dgs

### 2a. Core Innovation: Parallel N-Dimensional Gaussian Rasterizer

The standard 3DGS CUDA rasterizer renders 3 RGB channels. Feature-3DGS extends it to render **N arbitrary feature channels simultaneously** via a parallel rasterizer.

- Screen is divided into **16×16 tiles** (same as original 3DGS)
- Each thread processes one pixel
- RGB and feature maps share **identical spatial resolution** (no downsampling artifacts)
- Both rendered via the same volumetric front-to-back alpha-blending:

```
F_s(v) = Σ_i f_i · α_i · Π_{j<i}(1 - α_j)
```

Where f_i ∈ R^N is the per-Gaussian semantic feature vector, optimized during training.

**Per-Gaussian storage:** Each Gaussian stores Θ_i = {x_i, q_i, s_i, α_i, c_i, f_i}, where f_i ∈ R^N. In experiments, N = 128 (compressed) with teacher dim M up to 512.

### 2b. Convolutional Speed-Up Module

When N < M (teacher feature dim), a **1×1 kernel convolutional decoder** upsamples the rendered low-dim feature map:

```
F̂_M = Conv_{1×1}(F_N),   N < M
```

- Single-layer convolution: N_channels_in → M_channels_out, kernel 1×1
- No complex feature extraction — pure linear channel expansion
- Initialized randomly, trained jointly with Gaussians

*Configuration:* `NUM_SEMANTIC_CHANNELS = feature_out_dim / NUMBER` (NUMBER=4 by default), so LSeg 512→128, SAM 256→64.

### 2c. Foundation Features and Exact Dimensions

| Foundation Model | Teacher Dim M | Rendered Gaussian Dim N (with speedup) | Architecture |
|-----------------|--------------|----------------------------------------|--------------|
| **LSeg** | 512 | 128 | CLIP ViT-L/16 image encoder + ViT-L/16 text encoder |
| **SAM** | 256 | 64 | MAE pre-trained ViT-H/16, output at 64×64 spatial |
| **CLIP** | 512 | — | ViT-B/32 (text encoding only, for editing guidance) |

LSeg provides dense pixel-level language-aligned features at 360×480 resolution.  
SAM provides dense visual features (not language-aligned by itself).

### 2d. Distillation Loss

```
L = L_rgb + γ · L_f

L_rgb = (1 - λ) · L1(I, Î) + λ · L_{D-SSIM}(I, Î)    [λ = 0.2]

L_f   = || F_t(I) - F_s(Î) ||_1    [L1 between teacher and rendered student features]

γ = 1.0   [equal weighting of RGB and feature losses]
```

Where F_t(I) is the teacher 2D foundation model feature extracted from the real image, and F_s(Î) is the feature rendered by the Gaussian rasterizer.

**Optimizer:** Adam  
- Feature learning rate: 1e-3 for Gaussian features  
- Decoder learning rate: 1e-4 for the CNN upsampler  
- Training: standard 3DGS densification from iter 500 to 15,000; total 7K–30K iterations

### 2e. Applications Enabled

- Novel-view semantic segmentation (mIoU +23% over NeRF methods)
- Language-guided editing (via CLIP text encoding)
- Segment-anything (via SAM feature distillation)
- Rendering speed: 14.55 FPS; distillation speed 2.7× faster than NeRF-based methods

---

## 3. Gaussian Grouping (ECCV 2024)

**Paper:** "Gaussian Grouping: Segment and Edit Anything in 3D Scenes" — Ye et al., ECCV 2024  
**arXiv:** https://arxiv.org/abs/2312.00732  
**Code:** https://github.com/lkeab/gaussian-grouping

### 3a. Identity Encoding Per Gaussian

Each 3D Gaussian is augmented with a **16-dimensional Identity Encoding vector** e_i ∈ R^{16}.

- View-independent (SH degree = 0) → same encoding from all views, enabling object-level consistency
- Initialized randomly, optimized during training
- Dimension 16 chosen by ablation: 32 dims is 1.3× slower with no quality gain

### 3b. SAM-Based 2D Mask Association Across Views

**Challenge:** SAM generates masks independently per view → different mask IDs across views → inconsistent supervision.

**Solution — DEVA Temporal Propagation:**
1. Apply SAM in "everything mode" to generate masks for each view independently
2. Treat multi-view images as a **video sequence** with gradually changing views
3. Use **DEVA** (zero-shot universal temporal propagation model) to propagate and associate mask IDs across the view sequence
4. Output: K total unique mask identities with consistent IDs across all training views

This is **60× faster** than cost-based linear assignment (which would require recomputing matching at each training iteration).

### 3c. Identity Encoding Rendering

Rendered via standard alpha-compositing:

```
E_id(v) = Σ_{i ∈ N} e_i · α'_i · Π_{j=1}^{i-1}(1 - α'_j)
```

Where α'_i is derived from the 2D Gaussian projection Σ^{2D} = J W Σ^{3D} W^T J^T.

A **linear classification layer** f: R^{16} → R^K (K = total masks ≈ 256 classes) projects rendered identity features to per-class logits, followed by softmax.

### 3d. 3D Grouping Loss (Full Formula)

```
L_render = L_rec + λ_{2d} · L_{2d} + λ_{3d} · L_{3d}

λ_{2d} = 1.0
λ_{3d} = 2.0
```

**L_rec:** Standard 3DGS reconstruction loss (L1 + D-SSIM on RGB).

**L_{2d} (2D Identity Cross-Entropy):**
```
L_{2d} = CrossEntropy(softmax(f(E_id(v))), label(v))
```
Supervises rendered identity features to match DEVA-associated SAM mask IDs.

**L_{3d} (3D Spatial Consistency via KL Divergence):**
```
L_{3d} = (1 / m·k) Σ_{j=1}^{m} Σ_{i=1}^{k} F(e_j) log( F(e_j) / F(e'_i) )
```

Where:
- m = 1000 sampled anchor Gaussians (randomly sampled each iteration)
- k = 5 nearest neighbors in 3D Euclidean space for each anchor
- F(·) applies the linear classifier + softmax → distribution over K classes
- KL divergence forces spatially proximate Gaussians (in 3D) to have similar identity distributions

This regularizes the field so nearby Gaussians agree on group membership even without direct 2D supervision.

### 3e. Training Procedure

- Optimizer: Adam
- lr (identity encoding): 2.5×10^{-3}
- lr (linear layer): 5×10^{-4}
- Iterations: 30,000
- Hardware: Single A100 GPU
- Point cloud downsampling for L_{3d}: 300K points

### 3f. Scene Editing Applications

The grouping enables: 3D object removal, inpainting, colorization, style transfer, scene recomposition — all by selecting Gaussians belonging to the target group.

---

## 4. Brief Coverage: LERF, N2F2, OpenGaussian

### 4a. LERF (ICCV 2023)

**Paper:** "LERF: Language Embedded Radiance Fields" — Kerr et al., ICCV 2023  
**arXiv:** https://arxiv.org/abs/2303.09553

**Key Idea:** Embed CLIP features directly into NeRF by volume rendering CLIP embeddings along training rays. Supervises multi-scale CLIP features using an image pyramid (different crop scales → different CLIP embeddings at each scale), creating a dense multi-scale language field.

**Architecture:** NeRF-based (Instant-NGP backbone). CLIP features queried at points in 3D, composited by alpha-blending along rays. DINO features used as regularizer on object boundaries.

**Relevancy scoring:** Same min-over-canonical-phrases formula later adopted by LangSplat.

**What to borrow:**
- Multi-scale CLIP supervision concept (adapted by LangSplat to explicit Gaussians)
- Relevancy score formula (softmax over query vs. canonical phrases)
- DINO regularization for boundary sharpening

**Limitation:** 199× slower than LangSplat due to NeRF rendering. Not suitable for real-time or dynamic use.

---

### 4b. N2F2 — Nested Neural Feature Fields (ECCV 2024)

**Paper:** "N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields" — ECCV 2024  
**arXiv:** https://arxiv.org/abs/2403.10997

**Key Idea:** Stores a **single high-dimensional feature vector** per Gaussian (or grid point), where **different dimensions encode different scales of semantic granularity** (nested encoding). Scale-to-dimension mapping:

```
M(s) = floor(D · (1 - s))
```

Where s ∈ [0,1] is a normalized scale (0 = coarsest, 1 = finest), D is total feature dim.

**Architecture:** TriPlane + MLP (512×512 planes, 64-dim features, 3-layer MLP with 256 hidden). Features queried at Gaussian centers and rendered via differentiable splatting.

**Supervision:** OpenCLIP ViT-B/16 embeddings from SAM-segmented regions at multiple scales → supervised against first M(s) feature dimensions with learnable projection W_{1:M(s)}.

**Composite embedding for inference** (avoids explicit scale selection):
```
γ_k^{3D} = Softmax_i(max_j(W_{1:i} Θ_{k,1:i})^T φ_j^{canon})
```
Single-pass querying, 1.7× faster than LangSplat.

**What to borrow:**
- Nested dimension encoding — different feature dims capture different granularities
- Composite embedding for efficient multi-scale querying
- Avoids the 3-separate-Gaussian-groups approach of LangSplat

---

### 4c. OpenGaussian (arXiv 2024)

**Paper:** "OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding"  
**arXiv:** https://arxiv.org/abs/2406.02058

**Key Idea:** Focus on **3D point-level** (per-Gaussian) understanding rather than pixel-level. Attach a **6-dimensional instance feature vector** f ∈ R^6 to each Gaussian.

**Training losses (no cross-view tracker needed):**
1. **Intra-mask smoothing:** minimize distance between rendered features within a SAM mask and their mean
2. **Inter-mask contrastive:** maximize distance between mean features of different SAM masks

**Codebook for language grounding:** Two-level coarse-to-fine discretization:
- Coarse: concatenate 6-d instance feature + 3-d position → k_1 = 32 or 64 codewords (position-aware)
- Fine: within each coarse cluster, use k_2 = 5 or 10 codewords on features only

CLIP (512-d) applied post-hoc to associate codewords with text queries (training-free CLIP association).

**What to borrow:**
- Intra/inter-mask contrastive loss for feature distinctiveness without cross-view tracking
- Lightweight 6-d feature vectors that are still discriminative enough for grouping
- Two-level codebook for scalable open-vocabulary association

---

## 5. Critical Synthesis: Recommendations for a Generalizable Dynamics Model

### 5a. The Per-Scene Autoencoder Problem

**LangSplat's AE is a blocking issue for a generalizable model:**
- The AE is trained on each scene's own CLIP features → the 3-d latent space has no shared meaning across scenes
- A dynamics model operating across scenes (or predicting future states) cannot reuse AE weights
- Language queries at test time require decoding back to CLIP space via the scene-specific decoder → impossible without it

**Gen-LangSplat (arXiv 2510.22930)** directly addresses this: a single MLP-based AE pre-trained on millions of ScanNet mask embeddings achieves 93% cosine similarity to original CLIP features with a **16-dimensional latent**. Fixed weights across all scenes.

This is the correct approach for a generalizable model.

---

### 5b. Which Foundation Model to Use

**Recommendation: SigLIP2-SO-400M (or ViT-L/16) as the frozen feature extractor.**

Rationale:
1. **Qwen3-VL uses SigLIP2 as its vision encoder** (SO-400M for 7B/72B, Large-300M for 2B/4B). Its embedding space is already aligned to Qwen3-VL's language decoder.
2. Matching the same vision encoder that Qwen3-VL uses means text embeddings from Qwen3-VL's language head are directly comparable to image region features without a separate alignment step.
3. SigLIP2 is trained with a sigmoid loss (not softmax) → better calibrated similarity scores.
4. SigLIP2 produces dense local features (via DeepStack in Qwen3-VL), not just global pooled embeddings.

**Alternative:** OpenCLIP ViT-L/14 — same 512-d embedding space as LangSplat/Gen-LangSplat, more existing benchmarks, but not natively aligned to Qwen3-VL.

**Do NOT use:** DINOv2 alone (no language alignment — good for geometry/texture but not for text query matching). Use DINOv2 only as a boundary sharpening regularizer (as LERF does).

---

### 5c. Recommended Feature Dimensionality

**Per-Gaussian semantic feature: 16-dimensional latent** (following Gen-LangSplat ablation).

Full pipeline:
1. Extract **SigLIP2 (or OpenCLIP ViT-B/16) features** from masked image regions → D = 512-d (SigLIP2) or 512-d (OpenCLIP ViT-B/16)
2. Apply a **single frozen shared low-dim projection MLP** (trained once on a large diverse dataset like ScanNet + Objaverse): 512 → 16 (retains >93% cosine similarity per Gen-LangSplat ablation)
3. Store **16-d latent per Gaussian** as extra attribute channels
4. Render via alpha-compositing (same kernel as RGB)
5. At query time: decode 16-d → 512-d via frozen decoder MLP, then cosine similarity to text embedding

**Why 16 and not 3 (LangSplat) or 512 (raw)?**
- 3-d: insufficient capacity, requires scene-specific AE to be meaningful
- 512-d: 35× memory overhead per Gaussian; rendering 512 channels in CUDA is feasible but memory-heavy
- 16-d: compact (adds only ~6% memory over base 3DGS attributes), retains high fidelity, shared across scenes

**Memory comparison per Gaussian:**
- Base 3DGS attributes (pos+rot+scale+opacity+SH color): ~59 floats
- 16-d language feature: +16 floats → +27% overhead (manageable)
- 512-d raw CLIP: +512 floats → +867% overhead (prohibitive)

---

### 5d. Autoencoder: Shared Fixed Projection (Not Per-Scene)

**Architecture recommendation** (follow Gen-LangSplat with extensions):

**Encoder E_φ:** 512 → 256 → 128 → **16**, BatchNorm + ReLU, L2 norm output  
**Decoder D_φ:** 16 → 128 → 256 → **512**, ReLU, L2 norm output

**Training:**
- Dataset: ScanNet (230+ scenes) + Objaverse (1M+ objects) masked CLIP/SigLIP features
- Loss: L_AE = ||f - f̂||_1 + 0.001 × (1 - cos(f, f̂))
- Freeze E_φ and D_φ after training → fixed for all downstream scenes/models

**Key property:** After freezing, the 16-d latent space is consistent across all scenes. A dynamics model predicting future Gaussian states can transform these 16-d vectors in a semantically meaningful shared space.

---

### 5e. Multi-Scale Segmentation Strategy

**Recommendation:** Use SAM2 (or SAM) with 3-level hierarchical masks (subpart/part/whole) as in LangSplat, but with the shared projection applied to each scale's CLIP embedding independently.

Each Gaussian stores **3 × 16 = 48 latent dimensions** (one 16-d vector per scale), or optionally use N2F2's nested encoding (48-d total, with first 16-d encoding whole-object, next 16-d encoding part, last 16-d encoding subpart).

**N2F2 nested approach is preferable** for a dynamics model because:
- A single 48-d vector per Gaussian (instead of 3 separate 16-d vectors)
- Easier for the dynamics model to propagate (one feature vector per Gaussian)
- Scale-aware querying at test time without separate rendering passes

---

### 5f. Feature Consistency Across Dynamic Scenes

**Problem:** In a dynamic scene, Gaussians move over time. Semantic labels must be consistent: the "red mug" should have the same semantic embedding whether it's at position t or t+1.

**Recommendation — Canonical-Space Semantic Features:**

Following the "canonical + deformation field" paradigm (standard in dynamic 3DGS):
1. Store semantic features **in canonical space** (not world space): f_i^{canon} ∈ R^{16} attached to the canonical Gaussian
2. The deformation field warps only geometric attributes (position, rotation, scale) but **semantic features are invariant under deformation** (identity-preserving)
3. During rendering: for each timestep t, deform Gaussians geometrically, but render semantic features from the canonical f_i^{canon} unchanged

This mirrors how Gaussian Grouping uses SH degree 0 (view-independent) for identity encodings.

**Loss for temporal semantic consistency:**
```
L_semantic_temp = Σ_t Σ_i || f_i^{canon} - stop_gradient(avg_t(E_φ(CLIP(mask_i^t)))) ||_1
```
Anchor the canonical semantic features to the average CLIP embedding of all observations of that mask across time.

**Cross-scene consistency** is handled by the shared frozen AE: E_φ maps all scenes' CLIP features into the same 16-d space.

---

### 5g. Dataset-Level Consistency (Not Per-Scene)

**Problem:** Per-scene training (LangSplat) means semantic features from scene A are incomparable to those from scene B. For a generalizable dynamics model trained across the whole dataset, features must be in a shared space.

**Solution:**
1. **Freeze E_φ, D_φ** (trained once on large diverse data) → consistent latent space across all dataset scenes
2. **No per-scene fine-tuning** of E_φ or D_φ
3. Store Gaussian semantic features as **latent codes z_i ∈ R^{16}** using the shared E_φ
4. Language queries: encode text with OpenCLIP/SigLIP2 text encoder → D_φ(z_i) for pixel-level relevancy OR directly train a lightweight text→16-d projection head that maps text embeddings to the 16-d space

---

### 5h. Matching Against Qwen3-VL Language Embeddings

**Qwen3-VL-Embedding-8B** outputs embeddings up to 4096-d (MRL-supported, configurable from 64 to 4096). The model:
- Vision encoder: SigLIP2-SO-400M
- LLM: Qwen3-8B-Instruct (hidden dim 4096)
- Output: 4096-d unified embedding space for text and images
- Similarity: cosine similarity

**Recommended interface:**
1. Store Gaussian semantics as 16-d latent codes z_i (in OpenCLIP/SigLIP2 CLIP space via E_φ)
2. For Qwen3-VL text queries: encode the natural language query with Qwen3-VL-Embedding → 4096-d embedding → project to 512-d CLIP space via a learnable linear head trained to align Qwen3-VL text space with OpenCLIP/SigLIP2 text space
3. Apply frozen E_φ to project 512-d → 16-d → cosine similarity to Gaussian latents

**Alternative (more direct):**
Train a small projection head: Qwen3-VL text embedding (4096-d) → 16-d (the canonical latent space), using paired text-CLIP data. This avoids the detour through 512-d.

---

## 6. Supplementary: Related Generalizable Methods

### 6a. Gen-LangSplat (arXiv 2510.22930)

**Key contribution:** Replaces LangSplat's per-scene AE with a single pre-trained generalized AE.
- Input: 512-d OpenCLIP ViT-B/16 features
- Latent: **16-d** (optimal per ablation, 93% cosine fidelity)
- Training: ScanNet, millions of SAM mask embeddings
- Per-Gaussian storage: learnable z_i ∈ R^{16}
- Language field loss: L_Lang = ||Z(v) - H(v)||_1 + γ(1 - cos(Z(v), H(v)))
- Total: L_total = L_RGB + β·L_Lang (exact β not reported)
- Test-time query: render z → decode D_φ(z) → cosine similarity to text embedding

### 6b. SceneSplat (ICCV 2025 Oral)

**Key contribution:** Vision-language pretrained encoder (PT-v3 backbone) that operates natively on 3DGS attributes.
- Input: raw Gaussian attributes (position, color SH, opacity, scale, quaternion, language labels)
- Output: 768-d per-Gaussian semantic features aligned with SigLIP embeddings
- Pretraining: ScanNet + ScanNet++ + Matterport3D (600 epochs)
- Losses: SimDINO + MAE + iBOT dense feature alignment
- Enables zero-shot semantic understanding on unseen scenes

### 6c. GSemSplat (arXiv 2412.16932)

**Key contribution:** Generalizable semantic 3DGS from only 2 uncalibrated images.
- Built on Splatt3R (weight-shared ViT encoder + transformer decoders)
- Two semantic heads: region-specific (16-d compressed CLIP) + context-aware (16-d)
- Trained on 230 ScanNet++ scenes
- Loss: cosine similarity L = 1 - cos(F̄_R, F_R) per region

### 6d. OVGaussian (arXiv 2501.00326)

**Key contribution:** First method to train a generalizable semantic segmentation network directly on 3D Gaussians.
- Uses MinkowskiNet34C (sparse 3D convolutions on voxelized Gaussians)
- Per-Gaussian semantic vector: **16-d**, SH degree 0
- Training: 288 SegGaussian scenes, 300 epochs on H100
- Losses: semantic alignment (cross-entropy with CLIP text embeddings) + dense visual alignment (cosine with MaskCLIP)
- Generalizes without scene-specific fine-tuning: 43.84% mIoU vs. 33.45% for Gaussian Grouping

### 6e. DGD: Dynamic 3D Gaussians Distillation (arXiv 2405.19321)

**Key contribution:** Semantic features for dynamic Gaussian scenes.
- Foundation models: DINOv2 (384-d) or LSeg-CLIP (512-d)
- Semantic optimization: starts at iter 25,000, ends at 40,000 (after geometric stabilization)
- Loss: total_loss = loss_color + loss_reduce × loss_semantic
  - loss_reduce = 0.5 for DINOv2, 10 for LSeg-CLIP
  - loss_semantic: L1 between teacher 2D features and rendered Gaussian features
- Joint optimization of geometry and semantics in deformable canonical space

---

## 7. Complete Recommended Architecture for the Target System

### 7a. Summary of Design Decisions

| Design Choice | Recommendation | Justification |
|---------------|----------------|---------------|
| Foundation model | SigLIP2-SO-400M (frozen) | Aligned with Qwen3-VL; dense local features |
| Feature extractor | Masked region CLIP features per SAM level | Same as LangSplat; captures object-level semantics |
| Autoencoder | Shared frozen MLP (train once on ScanNet+Objaverse) | Cross-scene generalization; fixes LangSplat's key flaw |
| Latent dimension | 16-d per scale (3 scales → 48-d total) | Gen-LangSplat ablation shows 93% fidelity; N2F2 nesting optional |
| Scale strategy | 3 SAM scales (subpart/part/whole) | Captures multi-granularity semantics |
| Rendering | Alpha-compositing (same as RGB, extra channels) | Differentiable, fast, consistent with 3DGS |
| Dynamic consistency | Canonical-space features, deform geometry only | Semantic identity preserved through motion |
| Qwen3-VL matching | Train text→16-d projection head | Direct alignment; avoids CLIP intermediary |
| 3D consistency loss | Intra-mask smoothing + inter-mask contrastive (OpenGaussian) | No cross-view tracker needed |

### 7b. Per-Gaussian Attribute Layout

```
Gaussian_i = {
    x_i          ∈ R^3       # position
    q_i          ∈ R^4       # rotation quaternion
    s_i          ∈ R^3       # scale (log)
    o_i          ∈ R^1       # opacity (pre-sigmoid)
    SH_i         ∈ R^48      # color SH coefficients (degree 3)
    z_i^s        ∈ R^16      # subpart-level semantic latent
    z_i^p        ∈ R^16      # part-level semantic latent
    z_i^w        ∈ R^16      # whole-level semantic latent
}
# Total: 3+4+3+1+48+48 = 107 floats per Gaussian
# vs. base 3DGS: 3+4+3+1+48 = 59 floats
# Overhead: +81% (48 extra floats; acceptable)
```

### 7c. Training Procedure (Two-Stage)

**Stage 1 — Geometry pretraining (standard 3DGS):**
- Train geometric + color attributes only
- Standard L_rgb = (1-0.2)·L1 + 0.2·L_{D-SSIM}
- Densification: iter 500–15,000

**Stage 2 — Semantic feature training (no densification):**
- Freeze geometry; optimize z_i^s, z_i^p, z_i^w
- Loss:
```
L_semantic = Σ_{l ∈ {s,p,w}} L1(render(z^l, v), E_φ(CLIP(I ⊙ M^l(v))))
           + λ_intra × L_intra_mask
           + λ_inter × L_inter_mask
           + λ_3d × L_3d_kl
```
- λ_intra = 1.0, λ_inter = 1.0, λ_3d = 2.0 (following Gaussian Grouping)

### 7d. Test-Time Language Querying

**Option A — OpenCLIP text path:**
1. Encode query: φ_text = CLIP_text_encoder(query) ∈ R^{512}
2. Project: z_text = E_φ(φ_text) ∈ R^{16}
3. Render per-Gaussian semantic map: Z^l(v) = alpha-composite(z_i^l) ∈ R^{16}
4. Relevancy: R(v) = min_i [softmax(Z^l(v)·z_text / (Z^l(v)·z_text + Z^l(v)·z_canon^i))]

**Option B — Qwen3-VL text path:**
1. Encode query: φ_qwen = Qwen3VL_text_encoder(query) ∈ R^{4096}
2. Project via learned head H: z_text = H(φ_qwen) ∈ R^{16}  [linear or 2-layer MLP]
3. Rest same as Option A

**H is trained** using paired (text, CLIP region embedding) data: minimize cosine distance between H(Qwen3VL_text(t)) and E_φ(CLIP_image(region_for_t)).

---

## 8. Key Sources

- LangSplat paper: https://arxiv.org/abs/2312.16084
- LangSplat code: https://github.com/minghanqin/LangSplat
- Feature-3DGS paper: https://arxiv.org/abs/2312.03203
- Feature-3DGS code: https://github.com/ShijieZhou-UCLA/feature-3dgs
- Gaussian Grouping paper: https://arxiv.org/abs/2312.00732
- Gaussian Grouping code: https://github.com/lkeab/gaussian-grouping
- LERF paper: https://arxiv.org/abs/2303.09553
- N2F2 paper: https://arxiv.org/abs/2403.10997
- OpenGaussian paper: https://arxiv.org/abs/2406.02058
- Gen-LangSplat paper: https://arxiv.org/abs/2510.22930
- SceneSplat code: https://github.com/unique1i/SceneSplat
- GSemSplat paper: https://arxiv.org/abs/2412.16932
- OVGaussian paper: https://arxiv.org/abs/2501.00326
- DGD paper: https://arxiv.org/abs/2405.19321
- Qwen3-VL-Embedding: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
