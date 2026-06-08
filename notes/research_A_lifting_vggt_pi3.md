# Research Brief A — Feed-Forward 3D Geometry Lifting: VGGT & Pi3, and Image→3DGS Heads

**Scope:** Faithful, reimplementation-grade notes on (1) **VGGT** and (2) **Pi3 (π³)** for turning multi-view / video frames into 3D, (3) converting that 3D into 3D Gaussians, and (4) a survey of feed-forward image→3DGS heads (Splatt3R, AnySplat, GS-LRM, pixelSplat, MVSplat). Final section: recommendation for an AgiBot-World-style multi-camera robot-manipulation pipeline on 4×A100-80GB.

**Date compiled:** 2026-06-05. **Convention note:** All camera math below is **OpenCV** (x-right, y-down, z-forward; world→cam extrinsics) unless stated otherwise.

---

## 0. Primary sources (read directly)

- VGGT paper: arXiv 2503.11651 (CVPR 2025 Best Paper). HTML: https://arxiv.org/html/2503.11651v1 ; CVF PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf
- VGGT repo: https://github.com/facebookresearch/vggt — source files cited inline below by `file:symbol`.
- Pi3 paper: arXiv 2507.13347 (ICLR 2026). HTML: https://arxiv.org/html/2507.13347v1
- Pi3 repo: https://github.com/yyfz/Pi3
- Splatt3R: arXiv 2408.13912, repo https://github.com/btsmart/splatt3r
- AnySplat: arXiv 2505.23716 (SIGGRAPH Asia 2025 / ACM TOG), repo https://github.com/InternRobotics/AnySplat
- GS-LRM: arXiv 2404.19702 (ECCV 2024), Adobe — project page https://gs-lrm.github.io
- pixelSplat: arXiv 2312.12337 (CVPR 2024), repo https://github.com/dcharatan/pixelsplat
- MVSplat: arXiv 2403.14627 (ECCV 2024 Oral), repo https://github.com/donydchen/mvsplat

---

# PART 1 — VGGT (Visual Geometry Grounded Transformer)

## 1.1 Architecture

**Four components** (paper §3, `vggt/models/vggt.py`, `vggt/models/aggregator.py`):
1. Frozen DINOv2 tokenizer (patch embedder).
2. Camera tokens + register tokens.
3. Alternating-attention transformer backbone ("Aggregator").
4. Task-specific prediction heads (camera, depth, point-map, track).

### Backbone / tokenizer
- **DINOv2 ViT-Large with registers**, variant string `dinov2_vitl14_reg` (`aggregator.py`). Patch size **14×14**, embed dim **C = 1024**. The DINOv2 patchifier is **frozen** during VGGT training (paper §3.1: "frozen DINOv2"; chosen over a raw conv patchifier for stable training).
- **Total model ≈ 1.2 B parameters** (paper). Trained on **64 A100 GPUs for ~9 days**.

### Input normalization (CRITICAL — do not skip)
Inside `Aggregator.forward` images are normalized with **ImageNet (ResNet) statistics**, registered as buffers:
```
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD  = [0.229, 0.224, 0.225]
images = (images - self._resnet_mean) / self._resnet_std
```
So the model expects **input pixels in [0,1]** (the loader returns [0,1]); ImageNet normalization is applied *internally*. Do NOT pre-apply ImageNet norm yourself — `load_and_preprocess_images()` only does `ToTensor()` → [0,1].

### Tokens
- Per frame: image patch tokens (K = (H/14)·(W/14) tokens of dim 1024) **+ 1 camera token + 4 register tokens**.
- `num_register_tokens = 4`; `patch_start_idx = 1 + 4 = 5` (index where patch tokens begin in the sequence).
- Camera token parameter has **2 variants**: `self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))` and likewise register tokens — one set for the **first frame**, one set for **all other frames**, so the network can distinguish the reference (first) view. (paper §3.1: distinct first-frame tokens.)

### Alternating-attention backbone (the "Aggregator")
- **L = 24 transformer blocks**, configured as `aa_order = ["frame", "global"]`, `aa_block_size = 1`, `depth = 24` → **12 frame-attention + 12 global-attention blocks, strictly alternating** (`aggregator.py`).
- Each block: dim **1024**, **16 heads**.
- **Frame attention**: tokens reshaped to `(B·S, P, C)` → self-attention *within each frame only* (intra-frame texture/appearance; DINOv2 2D positional embeddings carry within-frame position).
- **Global attention**: tokens reshaped to `(B, S·P, C)` → self-attention *across all tokens of all frames* (cross-view correspondence, multi-view fusion). This is where emergent 3D reasoning happens.
- S = number of input views/frames; the same backbone handles 1…hundreds of views.

### Resolution handling
- Training: longest side resized to **518 px max**, aspect ratio randomized in **[0.33, 1.0]** (paper). Inference loader default **target_size = 518** (`vggt/utils/load_fn.py`).
- Loader modes: **`crop`** (default) — set width=518, keep aspect ratio, center-crop height if >518; **`pad`** — longest side=518, pad shorter side to 518 (white pad, value 1.0). Both round H,W to multiples of **14**; resize is **BICUBIC**. Batched frames are white-padded to equal shapes.

## 1.2 Outputs & parameterization (inference)

`predictions = model(images)` (`vggt/models/vggt.py:VGGT.forward`). Heads run under `torch.cuda.amp.autocast(enabled=False)` (fp32) after the aggregator. Keys & shapes (B=batch, S=views):

| Key | Shape | Meaning |
|---|---|---|
| `pose_enc` | `[B, S, 9]` | 9D camera pose encoding (last refinement iter of camera head) |
| `depth` | `[B, S, H, W, 1]` | per-view depth |
| `depth_conf` | `[B, S, H, W]` | depth confidence (aleatoric Σ⁻¹-like) |
| `world_points` | `[B, S, H, W, 3]` | point map (3D, first-camera frame) |
| `world_points_conf` | `[B, S, H, W]` | point-map confidence |
| `track`/`vis`/`conf` | `[B,S,N,2]`/`[B,S,N]`/`[B,S,N]` | 2D tracks + visibility + conf (only if `query_points` given) |
| `images` | `[B, S, 3, H, W]` | echoed input (inference) |

### Camera head — 9D encoding `g = [t, q, fov]`
- Parameterization (paper Eq.; `vggt/utils/pose_enc.py`): **`g = [T(3), quaternion(4), fov_h, fov_w]`** = **9D**. (Translation first, then quaternion, then 2 FOV angles.)
- Head = **4 extra self-attention layers + linear**, applied to the camera token; **iterative refinement** (the dict stores the last iteration).
- **Decode to extrinsic/intrinsic** (`pose_encoding_to_extri_intri`):
  - Extrinsic: `R = quat_to_mat(q)`, then `[R | t]` (3×4, **world→cam, OpenCV**).
  - Intrinsic: `fy = (H/2)/tan(fov_h/2)`, `fx = (W/2)/tan(fov_w/2)`; principal point assumed **centered (cx=W/2, cy=H/2)**; `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`. **Image H,W are required** for the FOV↔focal conversion.
  - Encode (`extri_intri_to_pose_encoding`): `q = mat_to_quat(R)`, `fov_h = 2·atan((H/2)/K[1,1])`, `fov_w = 2·atan((W/2)/K[0,0])`.

### Depth head & Point head — DPT decoders
- Both use **DPT** decoders over the aggregated tokens; output dense maps at input resolution.
- Depth: `Di ∈ R^{H×W}` + aleatoric uncertainty `Σ_i^D ∈ R+^{H×W}`.
- Point map: `Pi ∈ R^{3×H×W}` + uncertainty `Σ_i^P`. **`world_points` are in the first-camera coordinate frame** (see §1.3). Confidence in the dict is what the demo thresholds.

### Tracking head — CoTracker2-style
- Dense feature map `Ti ∈ R^{C×H×W}` per frame; given query points, predicts 2D correspondences across frames + visibility. (Not needed for static lifting.)

### Recovering 3D from outputs (two equivalent routes)
1. **Direct point map**: use `world_points` (already 3D in first-cam frame).
2. **Unproject depth** (`vggt/utils/geometry.py:unproject_depth_map_to_point_map` → `depth_to_world_coords_points`):
   - cam coords: `x_cam=(u-cx)d/fx`, `y_cam=(v-cy)d/fy`, `z_cam=d`.
   - world: invert `[R|t]` via `R_inv=Rᵀ`, `t_inv=-Rᵀt` (closed-form SE3 inverse), then `world = cam @ R_inv.T + t_inv`.
   - valid mask: `depth > 1e-8`.
   - **The repo recommends the depth-branch + intrinsics/extrinsics route over the raw point head for best accuracy** (demo exposes both via `prediction_mode`).

## 1.3 Coordinate frame & normalization (CRITICAL)
- **World = first camera.** Extrinsics of image 1 are identity: `q1=[0,0,0,1]`, `t1=[0,0,0]`. All `world_points` / poses are expressed in **camera-1 frame** (paper §3: "the 3D points Pi(y) are defined in the coordinate system of the first camera g1"). Pi(y) is "viewpoint invariant" only in the sense of being a fixed (first-cam) frame for all views.
- **Scale normalization** (applied to GT during training; model learns to emit this canonical scale): compute the **average Euclidean distance of all 3D points in P to the origin**, and normalize **camera translations t, point map P, and depth D** by that scale. → outputs are in an **arbitrary but internally-consistent metric scale**, NOT metric meters. For robotics you must rescale (e.g. via known camera baseline / robot kinematics).

## 1.4 Loss functions (training reference)
Total (paper §3.3): `L = L_camera + L_depth + L_pmap + λ·L_track`, with `λ = 0.05`.
- **Camera:** `L_camera = Σ_i ||ĝ_i − g_i||_ε` — Huber on 9D pose encoding.
- **Depth (aleatoric):** `L_depth = Σ_i [ ||Σ_i^D ⊙ (D̂_i − D_i)|| + ||Σ_i^D ⊙ (∇D̂_i − ∇D_i)|| − α·log Σ_i^D ]` (uncertainty-weighted L1 on value + gradient; α regularizer).
- **Point map:** identical structure with `Σ_i^P`, `P̂_i`, `P_i`.
- **Track:** `L_track = ΣΣ ||y_{j,i} − ŷ_{j,i}||` (L2 on 2D correspondences) + BCE on visibility.
- The predicted confidences exposed at inference (`depth_conf`, `world_points_conf`) are derived from these learned `Σ` maps; higher = more confident.

## 1.5 Checkpoints & license
- **`facebook/VGGT-1B`** (original, HF): **license `cc-by-nc-4.0` → NON-COMMERCIAL.** ~1.2B params. Publicly downloadable (not gated as of check), but NC.
- **`facebook/VGGT-1B-Commercial`** (released July 29 2025, HF): license **`vggt-aup-license`** — **commercial use permitted, EXCEPT military applications**; access via a LLaMA-style auto-approved application form. Comparable performance to the original.
- Repo **code** `LICENSE.txt` was updated 2025-07-29 to a **commercial-friendly Meta license** (broad royalty-free use, subject to an Acceptable Use Policy). **So: code = commercial OK; for weights you MUST use VGGT-1B-Commercial, not the NC VGGT-1B.**
- Load: `model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial").to(device)`.

## 1.6 Runtime / memory (80GB A100/H100)
- Single frame: paper claims **1.88 GB**; real-world users report **~7–8 GB** for the 1B model at 1 frame (GitHub issue #81 — discrepancy acknowledged).
- Memory grows **~linearly** with view count due to global attention over `S·P` tokens. Community reports: **~40 GB at 200 frames (H100), OOM around ~300 frames on an 80GB A100.** Inference for ~200 frames ≈ **8.75 s on H100**.
- A **May 18 2026 repo memory fix** ("2–3× more input frames" on the same GPU) raises the practical ceiling. Accelerated variants exist (FastVGGT, VGGT-Ω, StreamVGGT/XStreamVGGT with KV-cache) for long sequences.
- Practical guidance: **process in windows of ~32–100 views**; for AgiBot multi-cam, a synchronized 8-cam frame set is trivially within budget.

## 1.7 Failure modes / pitfalls
- **Not metric scale** — outputs are normalized by mean point distance (§1.3). Must rescale for robotics.
- **Symmetric / textureless / specular** surfaces → unreliable point maps; the dedicated 3DGS works (e.g. "Revisiting Depth Representations…", VGD) note VGGT pointmaps have inaccurate regions that hurt downstream 3DGS.
- **Pose head assumes centered principal point** and FOV-based intrinsics — wrong for cameras with significant principal-point offset; if you have true intrinsics, unproject depth with your own K instead of the predicted K.
- **Preprocessing must match**: 518-longest-side, /14 divisibility, [0,1] input (internal ImageNet norm), white padding. Mismatched aspect ratio / normalization degrades results badly.
- Memory blows up at many views (§1.6).

---

# PART 2 — Pi3 (π³): Permutation-Equivariant Visual Geometry Learning

## 2.1 Core idea & how it differs from VGGT
Pi3 removes the **fixed reference view** dependency. It is **fully permutation-equivariant**: no positional/frame-index embeddings, **no special first-frame/reference camera token**. Reorder the inputs → outputs reorder identically, with no accuracy change. Predicts **affine-invariant camera poses** and **scale-invariant per-view local point maps**, then composes them into a global cloud.

## 2.2 Architecture (paper §3, repo)
- **Encoder:** DINOv2 ViT (frozen), three sizes: ViT-S/384 (6 heads), ViT-B/768 (12 heads), **ViT-L/1024 (16 heads)** — same alternating idea as VGGT but **order-agnostic**. Encoder for the released model is **initialized from pretrained VGGT and frozen**; the decoder is trained from scratch.
- **Decoder:** **36 layers** of **alternating view-wise ↔ global self-attention** (vs VGGT's 24), **with NO frame-position embeddings and NO reference token** — this is what makes it permutation-equivariant.
- **Params:** Small ≈ **196.5M**, Base ≈ **390.1M**, **Large ≈ 959M**.

### Heads
- **Camera pose head** (adapted from **Reloc3r**): MLP → average pooling → MLP. Rotation predicted as a **9D representation → orthonormalized to a 3×3 via SVD**; output is a per-image **`T_i ∈ SE(3) ⊂ R^{4×4}`** (a **camera-to-world** transform, see repo) — affine-invariant (defined only up to a global similarity transform).
- **Local point-map head:** decoder branch with self-attention restricted to each image's own features, then **MLP + pixel-shuffle** upsampling → **`X_i ∈ R^{H×W×3}` in that image's own camera frame**, scale-invariant.
- **Confidence head:** same architecture as point head → `C_i ∈ R^{H×W}` (logits; apply **sigmoid**).

## 2.3 Outputs (inference) — repo dict
`model(images)` returns (B=batch, N=views):
| Key | Shape | Meaning |
|---|---|---|
| `points` | `[B, N, H, W, 3]` | **global** cloud (local points unprojected by camera_poses) |
| `local_points` | `[B, N, H, W, 3]` | per-view local point maps (camera frame) |
| `conf` | `[B, N, H, W, 1]` | confidence **logits** → `torch.sigmoid()` |
| `camera_poses` | `[B, N, 4, 4]` | **camera-to-world** 4×4, **OpenCV** |

Note the per-view shape ordering is `...H, W, 3` (channels-last), like VGGT's `world_points`.

## 2.4 Coordinate frame & normalization (CRITICAL)
- **No fixed reference frame.** Local point maps are each in their **own camera coordinate system**. Camera poses + points are defined only **up to an arbitrary global similarity transform** (rigid + single global scale). The whole reconstruction has **one unknown global scale shared across all N views** ("consistent scale factor across all N images").
- **Global cloud assembly:** solve an **optimal global scale `s*`** (ROE solver), scale local points `s*·x_{i,j}`, then transform to world via `T_i`. The repo's `points` key already does this composition (`points = unproject(local_points, camera_poses)`).
- Like VGGT: **not metric**; gauge-free (similarity-invariant).

## 2.5 Loss functions (training reference)
- **Points (depth-weighted L1, optimal scale ŝ):** `L_points = Σ_i Σ_j (1/z_{i,j})·||ŝ·x̂_{i,j} − x_{i,j}||_1`.
- **Normal:** `L_normal = Σ_i Σ_j arccos(n̂_{i,j}·n_{i,j})`.
- **Confidence:** BCE (target 1 if recon error < ε else 0).
- **Camera (relative, all pairs):** `L_cam = 1/(N(N−1)) Σ_{i≠j}[L_rot(i,j) + λ·L_trans(i,j)]`, with `L_rot = arccos((Tr(R_{i←j}ᵀ R̂_{i←j})−1)/2)`, `L_trans = Huber_δ(s*·t̂_{i←j} − t_{i←j})`. Using **relative pairwise** poses is what keeps the camera loss permutation-equivariant.
- **Total:** `L = L_points + λ_n·L_normal + λ_c·L_conf + λ_cam·L_cam`.

## 2.6 Training & data
- Two-stage (DUSt3R-style): Stage 1 **224×224**, 100 epochs, 64 imgs/GPU (16 A100); Stage 2 random res **100k–255k px**, 100 epochs, 48 imgs/GPU (**64 A100**). Encoder = frozen VGGT init.
- **15 datasets:** GTA-SfM, CO3D, WildRGB-D, Habitat, ARKitScenes, TartanAir, ScanNet, ScanNet++, BlendedMVG, MatrixCity, MegaDepth, Hypersim, Taskonomy, Mid-Air, + an internal dynamic dataset.

## 2.7 Checkpoints, license, runtime
- **Checkpoints (HF):** `yyfz233/Pi3` and `yyfz233/Pi3X` (`Pi3X` is the recommended newer variant), `model.safetensors`.
- **License:** **code = BSD-3-Clause (commercial OK)**; **weights = CC BY-NC 4.0 (NON-COMMERCIAL).** (Code license confirmed from the repo LICENSE; weights NC per README.) → For commercial use you'd need to retrain weights, or get permission.
- Load: `from pi3.models.pi3x import Pi3X; model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()`.
- **Preprocessing:** input tensor `[B, N, 3, H, W]`, values in **[0,1]** (DINOv2/ImageNet norm applied internally as in VGGT). Use **bf16** (CC ≥ 8.0) or fp16. Frame sampling examples use interval 10 for video.
- **Runtime/memory:** repo gives no explicit table; architecture is VGGT-class (slightly larger decoder, 36 vs 24 layers) so expect **similar or slightly higher** memory than VGGT-1B per view. No fixed reference frame → robust to arbitrary multi-cam ordering (good for synchronized rigs).

---

# PART 3 — Converting feed-forward geometry → 3D Gaussians

## 3.1 Minimal/naïve route (works with raw VGGT or Pi3, no extra training)
This is what the VGGT demo does for point clouds, extended to Gaussians:
1. **Geometry source:** prefer **unprojected depth + predicted (or true) intrinsics/extrinsics** over the raw point head (repo recommendation). Gives one 3D point per pixel → one Gaussian per pixel (or per subsampled pixel).
2. **Confidence filtering** (`visual_util.py:predictions_to_glb`): threshold by **percentile**:
   ```
   conf_threshold = 0.0 if conf_thres==0 else np.percentile(conf, conf_thres)   # default conf_thres = 10 (10th pct)
   conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
   ```
   Optionally also drop near-black/near-white pixels and sky-segmented pixels.
3. **Color:** taken **directly from the input image pixel** at that ray: `colors_rgb = (image_pixels * 255).astype(uint8)` (then masked identically to vertices). For 3DGS set the **SH DC term** = `RGB2SH(color)` (i.e. `(c-0.5)/0.28209479177387814`), higher SH = 0.
4. **Initial Gaussian heuristics (standard 3DGS init, adapt to feed-forward):**
   - **xyz** = the 3D point (world / first-cam frame).
   - **scale** = function of local point spacing: `s = log(mean_knn_dist)` (3DGS uses `scale = log(sqrt(mean dist to 3 nearest neighbors))`); for per-pixel grids, a cheap proxy is `scale ≈ depth · pixel_footprint / focal`. Store as **log-scale** (activation `exp`).
   - **rotation** = identity quaternion `[1,0,0,0]` (will be optimized).
   - **opacity** = constant, stored as **inverse-sigmoid of 0.1** (3DGS default `inverse_sigmoid(0.1)`); activation `sigmoid`.
   - **SH degree** 0–3 (start DC-only).
5. Then either **render directly** (rough) or run a short **3DGS optimization** with the poses fixed.

## 3.2 Learned route — VGGT/Pi3 backbone + a Gaussian head (preferred quality)
Several works attach a learned Gaussian head to VGGT-style features:
- **AnySplat** (most relevant; see Part 4) — VGGT geometry encoder + DPT Gaussian head + differentiable voxelization, trained with only photometric + VGGT pseudo-geometry supervision (no SfM/MVS GT). Open weights `lhjiang/anysplat` (HF).
- **VGGS** (AAAI; AllenXiangX/VGGS), **VGD** (arXiv 2510.19578, surround-view driving), **VG3T**, **"Revisiting Depth Representations for Feed-Forward 3DGS"** (arXiv 2506.05327) — all use VGGT pointmaps as 3D supervision/initialization; the last notes pointmap inaccuracies are the main quality bottleneck and proposes fixes. VGGT pointmaps can be **precomputed offline ~0.3 s/scene** and cached.

## 3.3 Practical tip
Pi3/VGGT outputs are **gauge-free (similarity scale)** — fix the gauge before any photometric Gaussian training (e.g. anchor scale to a known camera baseline) so the rasterizer's near/far and scale activations behave.

---

# PART 4 — Survey: feed-forward image→3DGS heads we can reuse

For each: predicts per-pixel Gaussians? + exact parameterization + license/checkpoints.

## 4.1 Splatt3R (arXiv 2408.13912; btsmart/splatt3r)
- **Per-pixel Gaussians: YES.** Built on **MASt3R** (`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`), 2-view, **512** res. Adds a **third head** on top of MASt3R that predicts, per point: **covariance (rotation quaternion + scale), spherical-harmonics color, opacity, and a mean offset Δ** (the Gaussian center = MASt3R 3D point + learned offset). No explicit pose/intrinsic/mono-depth prediction (inherits MASt3R geometry). Trains only the Gaussian head + a loss-masking trick (only supervise pixels with valid GT).
- **License:** **CC BY-NC-SA 4.0 (NON-COMMERCIAL).**
- **Checkpoint:** `huggingface.co/brandonsmart/splatt3r_v1.0` (`epoch=19-step=1200.ckpt`).

## 4.2 AnySplat (arXiv 2505.23716; InternRobotics/AnySplat) — closest to our target
- **Per-pixel Gaussians: YES → voxelized.** Pipeline: **VGGT-style transformer geometry encoder → 3 DPT heads** (Gaussian params, depth, camera pose). Per-pixel Gaussians are **unprojected** (`batchify_unproject_depth_map_to_point_map`) then **voxelized** (`voxelizaton_with_fusion`, softmax/confidence-weighted fusion of per-voxel features+positions → sparse per-voxel Gaussians).
- **Gaussian head:** `VGGT_DPT_GS_Head`, `output_dim = raw_gs_dim + 1`, `raw_gs_dim = 1 + gaussian_adapter.d_in`. Channel 0 → **opacity/density via sigmoid**; opacity uses a pdf→opacity map `0.5*(1-(1-pdf)^k + pdf^(1/k))` with warm-up. Remaining → `GaussianAdapter`.
- **GaussianAdapter parameterization** (`src/model/encoder/common/gaussian_adapter.py`): `d_in = 7 + 3·d_sh` split as **scales(3) | rotation(4) | SH(3·d_sh)**.
  - **rotation:** unit quaternion, `rotations / (||rotations|| + eps)`.
  - **scale:** two variants — base `scale_min + (scale_max−scale_min)·sigmoid(s)` then ×depth/projection; `UnifiedGaussianAdapter`/`Unet3dGaussianAdapter`: `0.001·softplus(s)` then `clamp_max(0.3)`.
  - **SH:** `d_sh = (sh_degree+1)²`, 3 channels (RGB); higher-degree coeffs initialized small (`0.1·0.25^degree`).
- **Distinctive:** trains with **NO SfM/MVS GT** — geometry supervised by **VGGT pseudo-priors** + photometric loss; jointly predicts intrinsics/extrinsics. Strong at **dense (32+) views**; "high-fidelity in under one day on 8–16 GPUs."
- **License:** not explicitly stated in README (verify the repo LICENSE before commercial use; uses VGGT in acknowledgements → check VGGT weight provenance). **Checkpoint:** `lhjiang/anysplat` (HF).

## 4.3 GS-LRM (arXiv 2404.19702; Adobe) — best parameterization reference, but NOT released
- **Per-pixel Gaussians: YES.** Simple transformer: patchify posed images → concat multi-view tokens → transformer → **per-pixel 12-channel Gaussian** decode.
- **Parameterization (12 ch/pixel):** **RGB(3) | scale(3) | rotation quaternion(4) | opacity(1) | ray-distance t(1).** Center = **ray origin + t·ray-direction** (uses known camera pose). **scale = exp(·) clipped to max**; **opacity = sigmoid**; **rotation = normalized quaternion**; **color = plain RGB (no SH).**
- **Transformer:** 16 heads, MLP hidden 4096, GeLU; (object & scene models; scene at RealEstate10K res, object at 256). **0.23 s on a single A100** for 2–4 posed views.
- **License/availability:** **Adobe; official code & checkpoints were NEVER publicly released.** Only an unofficial partial reimpl exists (`InternRobotics/gs-lrm-unofficial`, stage-1 scene training code, **no checkpoints, no license stated**). → Use as a **design reference only**, not a drop-in.

## 4.4 pixelSplat (arXiv 2312.12337; dcharatan/pixelsplat)
- **Per-pixel Gaussians: YES, probabilistic depth.** 2-view, **256×256**, epipolar-attention encoder. Predicts a **discrete distribution over depth buckets** (in disparity space, near/far). Gaussian center `μ = o + (b_z + δ_z)·d_u` (origin + sampled-bucket depth + offset, along ray). **Opacity α = the bucket's sampled probability** (reparameterization trick → differentiable depth). Per pixel: one **covariance** (rotation+scale) + **SH** color. Union of both views' Gaussians (no fusion).
- **License:** repo is open-source (verify exact license in repo; paper itself only carries the arXiv license). **Checkpoints:** RealEstate10K & ACID provided by authors.

## 4.5 MVSplat (arXiv 2403.14627; donydchen/mvsplat)
- **Per-pixel Gaussians: YES, cost-volume depth.** Builds a **plane-sweep cost volume** for depth, then per-pixel Gaussians: **xyz from depth along ray, rotation quaternion, scale (activation), opacity, SH color.** **~12M params (10× smaller, 2× faster than pixelSplat).** 2-view, **256×256**, photometric (MSE+LPIPS) supervision only.
- **License:** **MIT** (commercial OK). **Checkpoints:** `re10k.ckpt`, `acid.ckpt` (Google Drive). Built on pixelSplat + UniMatch.

### Quick comparison
| Method | Backbone | Views | Res | Gaussian center | Rot | Scale act | Opacity act | Color | License | Ckpt |
|---|---|---|---|---|---|---|---|---|---|---|
| Splatt3R | MASt3R | 2 | 512 | MASt3R pt + Δ | quat | (sigmoid-range) | — | SH | **CC BY-NC-SA** | ✅ HF |
| AnySplat | VGGT | many | var | unproj depth (voxelized) | unit quat | softplus·0.001, clamp 0.3 | sigmoid+pdf-map | SH `(deg+1)²` | unstated/verify | ✅ HF `lhjiang/anysplat` |
| GS-LRM | plain ViT | 2–4 posed | 256/RE10K | o+t·d | norm quat | exp+clip | sigmoid | **RGB** | Adobe, **not released** | ❌ |
| pixelSplat | epipolar ViT | 2 | 256 | o+(b_z+δ)·d | quat | (act) | **= bucket prob** | SH | open (verify) | ✅ |
| MVSplat | cost-volume | 2 | 256 | depth·ray | quat | (act) | sigmoid | SH | **MIT** | ✅ |

---

# PART 5 — Recommendation for our pipeline (AgiBot World: 8+ synced cams, 30 fps; 4×A100-80GB)

### Geometry lifter: **Pi3 as primary, VGGT-1B-Commercial as the commercial-safe fallback.**

**Why Pi3 for a multi-camera rig:**
- **Permutation-equivariant** → robust to arbitrary camera ordering across the 8-cam synchronized rig and across time; no brittle "first camera = world" assumption. This matters when no single view is a natural reference and views are symmetric/redundant.
- Outputs both **local point maps** (per-cam, good for per-camera fusion) and a **globally-composed cloud**, plus **camera-to-world poses** — convenient for fusing with known extrinsics from the robot calibration.
- Slightly larger/deeper (36-layer decoder, ~959M) but VGGT-class memory; an 8-cam frame is trivially in budget on one A100.

**Why keep VGGT in the loop:**
- **VGGT-1B-Commercial** is the only one of the two with a **commercial-OK weight license** (Pi3 weights are CC BY-NC). If this becomes a product, Pi3 weights are a blocker — VGGT-1B-Commercial (commercial, ex-military) is the safe choice.
- Larger ecosystem of VGGT→3DGS heads (AnySplat, VGGS, VGD) reuse VGGT features directly.
- VGGT has a 9D pose head giving intrinsics too (useful if rig intrinsics drift); Pi3 gives poses but you supply/recover intrinsics.

**License summary (decisive for productization):**
| Model | Code | Weights | Commercial? |
|---|---|---|---|
| VGGT-1B (original) | commercial (Meta lic.) | **cc-by-nc-4.0** | ❌ weights NC |
| **VGGT-1B-Commercial** | commercial | **vggt-aup (commercial, no military)** | ✅ |
| Pi3 / Pi3X | BSD-3 | **cc-by-nc-4.0** | ❌ weights NC |

### Gaussian head: start with **AnySplat** (VGGT-native, open weights, handles many views, voxelized → bounded primitive count), keep **MVSplat (MIT)** as the clean, license-safe, retrainable fallback; use **GS-LRM's 12-channel scheme as the parameterization reference** for any head we train ourselves (RGB or SH DC + exp-scale + sigmoid-opacity + normalized quaternion + ray-distance center).

### Feasibility on 4×A100-80GB
- **Per multi-cam frame (8–16 views @ 518px):** comfortably fits on **one** A100 (VGGT ~40GB at 200 views → 8–16 views ≈ a few GB). 4 GPUs → run **4 independent frame-windows / 4 cameras-groups / 4 timesteps in parallel**, or batch.
- **30 fps temporal handling:** do **NOT** feed all 30·N frames into one global-attention pass (OOM ~300 frames, quadratic-ish cost). Instead: (a) **temporally subsample** (e.g. keyframe every K frames, interval-10 like Pi3's video example), and/or (b) use **windowed inference** (32–100 frames per window) with streaming variants (StreamVGGT/XStreamVGGT KV-cache, FastVGGT) for long horizons. The synchronized 8-cam *spatial* set per timestep is the natural unit.
- **Scale recovery:** both models are gauge-free (similarity scale). Anchor to **known rig extrinsics/baseline** or robot kinematics to get metric scale before 3DGS training; fix the gauge before any photometric optimization.
- **Preprocessing contract (both):** input [0,1], longest side 518, dims divisible by 14, ImageNet norm applied internally, white-pad to equal shapes, BICUBIC; use predicted intrinsics only if you lack true ones (prefer the robot's calibrated K + unproject depth).

### Concrete plan
1. **Lift:** Pi3X per synchronized 8-cam timestep (+VGGT-1B-Commercial as commercial fallback). Cache pointmaps offline (~0.3 s/scene VGGT).
2. **Fuse + gauge:** rescale with known rig baseline; merge per-cam local clouds via predicted (or calibrated) poses.
3. **Gaussianize:** AnySplat head for fast feed-forward 3DGS (or naïve per-pixel init §3.1 + short 3DGS opt for max fidelity).
4. **Confidence filter** at ~10th percentile (VGGT/Pi3 `conf`→sigmoid for Pi3), color from source pixels, opacity `inverse_sigmoid(0.1)`, scale from kNN spacing, identity rotation.
5. **Productization gate:** if commercial → VGGT-1B-Commercial weights + MIT/own-trained Gaussian head; avoid Pi3/Splatt3R/original-VGGT weights (all NC).

---

## Appendix — exact constants cheat-sheet
- VGGT input norm: mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, input [0,1]. Patch 14. Backbone `dinov2_vitl14_reg`, dim 1024, 16 heads, 24 blocks (12 frame + 12 global), 4 register + 1 camera token (2 first-vs-rest variants), `patch_start_idx=5`.
- VGGT pose enc: `[t(3), quat(4), fov_h, fov_w]`; `fx=(W/2)/tan(fov_w/2)`, `fy=(H/2)/tan(fov_h/2)`, cx=W/2, cy=H/2.
- VGGT unproject: `x=(u-cx)d/fx, y=(v-cy)d/fy, z=d`; world via `R_inv=Rᵀ, t_inv=-Rᵀt`; valid `depth>1e-8`.
- VGGT demo filter: `np.percentile(conf, conf_thres)`, default `conf_thres=10`, mask `(conf>=thr)&(conf>1e-5)`.
- Pi3: 36-layer decoder, no frame embeddings, no reference token; rotation 9D→SVD; outputs `points/local_points/conf(logits→sigmoid)/camera_poses(4×4 cam-to-world, OpenCV)`.
- GS-LRM 12ch: RGB(3)|scale(3,exp+clip)|quat(4,norm)|opacity(1,sigmoid)|ray-t(1); center `o+t·d`.
- AnySplat adapter: `d_in=7+3·d_sh`; scale `0.001·softplus`→clamp 0.3; quat normalized; SH `(deg+1)²`.
