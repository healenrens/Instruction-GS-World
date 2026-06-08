# Research A2: Feed-Forward Geometry for Dynamic Monocular Video
## Exhaustive Technical Brief for Reimplementation

**Context:** AgiBot World robot manipulation videos, 30 fps, monocular pinhole head camera 480×640, dynamic scenes (arms + objects move). Goal: per-frame consistent pointmaps/depth + camera poses → lifted into 3D Gaussians + supervision for a dynamics model.

**Date:** 2026-06-05

---

## 0. Foundations: DUSt3R and MASt3R

### 0.1 DUSt3R (Wang et al., CVPR 2024; arXiv:2312.14132)

**Core idea:** Cast multi-view 3D reconstruction as regression of pointmaps, removing the hard constraints of projective geometry. A single transformer network takes an image pair and outputs two pointmaps — one per image — both expressed in the first image's camera frame.

#### 0.1.1 Pointmap Parameterization (Paper §3.1)

A **pointmap** X ∈ ℝ^{H×W×3} associates every pixel (u,v) with an absolute 3D coordinate (x,y,z). For a camera with intrinsics K and depth map D:

```
X_{i,j} = K^{-1} · [i·D_{i,j},  j·D_{i,j},  D_{i,j}]^T
```

For stereo pair (I^1, I^2) the network f_θ outputs:

```
(X^{1,1}, C^{1,1},  X^{2,1}, C^{2,1}) = f_θ(I^1, I^2)
```

where X^{v,1} ∈ ℝ^{H×W×3} is the pointmap of view v expressed **in camera frame of view 1**, and C^{v,1} ∈ ℝ^{H×W} is the associated **confidence** map.

Pointmap transformations obey (Eq.1):
```
X^{n,m} = P_m · P_n^{-1} · h(X^n)
```
where P_m, P_n are 4×4 world-to-camera matrices, h(·) converts to homogeneous.

#### 0.1.2 Confidence-Weighted Regression Loss (Paper §3.2, Eq. 4)

Scale-invariant confidence-weighted loss:

```
ℒ_conf = Σ_{v∈{1,2}} Σ_{i∈D^v}  C_i^{v,1} · ℓ_regr(v,i)  −  α · log C_i^{v,1}
```

where:
- **C_i^{v,1}** = confidence at pixel i for view v, parameterized as 1 + exp(C̃_i^{v,1}) > 1 (softplus-like, ensures positivity and lower bound of 1)
- **ℓ_regr(v,i)** = ‖(1/z) X_i^{v,1} − (1/z̄) X̄_i^{v,1}‖₂  (Eq. 2: Euclidean after scale normalization)
- **z, z̄** = average L2 norm of predicted / GT points to origin: z = (1/|D^1|+|D^2|) Σ_{v,i} ‖X_i^{v,1}‖₂  (Eq. 3)
- **α** = regularization hyperparameter (prevents confidence collapse to 0; forces high confidence only where error is low)

The log term acts as regularizer: if C→∞ the loss is dominated by the ℓ_regr weighted by huge C; if C→0 the −log(C) term blows up. Optimum: C ∝ 1/ℓ_regr.

#### 0.1.3 Global Alignment (Paper §3.4, Eq. 5)

Given a view graph G = (V,E) with pairwise predictions {X^{v,e}, C^{v,e}} for each edge e∈E, recover globally consistent pointmaps χ^n for each camera n and per-edge scales σ_e:

```
χ* = argmin_{χ, P, σ}  Σ_{e∈E} Σ_{v∈e} Σ_{i=1}^{HW}  C_i^{v,e} · ‖χ_i^v − σ_e · P_e · X_i^{v,e}‖
```
subject to ∏_e σ_e = 1 (prevents trivial zero-scale solution).

Optimized via **LBFGS** on a flat set of poses P_e ∈ SE(3) and scales σ_e.

#### 0.1.4 Architecture

- **Encoder:** ViT-Large (24 layers, 1024-dim, 16×16 patches), initialized from CroCo v2 pretrained weights
- **Decoder:** Two ViT-Base decoders (12 layers each) with cross-attention between the two image branches
- **Head:** DPT (Dense Prediction Transformer) for final H×W×3 pointmap and H×W confidence output
- **Image preprocessing:** Random center crop, color jitter; resolutions (224×224) then (512×384, 512×336, 512×288, 512×256, 512×160); 16×16 patches; RoPE positional embeddings
- **Training data:** 8 datasets, ~8.5M pairs (Habitat, CO3Dv2, MegaDepth, ARKitScenes, BlendedMVS, Waymo, ScanNet++, StaticThings3D)

#### 0.1.5 Coordinate Frame, Scale, Outputs

- Both X^{1,1} and X^{2,1} in **camera-1 frame** (NOT world frame by default)
- Scale is **up-to-unknown-scale** at pairwise level; global alignment recovers consistent up-to-scale
- Metric scale only recovered if GT scale in training data
- **No tracking output** — DUSt3R is pairwise only
- **License:** CC BY-NC-SA 4.0 (Naver Labs) — **non-commercial research only**
- **Checkpoint:** `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` on HuggingFace naver/DUSt3R

---

### 0.2 MASt3R (Leroy et al., ECCV 2024; arXiv:2406.09756)

MASt3R extends DUSt3R by adding a **dense local feature head** to enable reciprocal nearest-neighbor matching.

#### 0.2.1 Additional Head and Loss

An additional MLP head (on top of the DPT decoder tokens) outputs a **descriptor map** D^v ∈ ℝ^{H×W×24} (d=24 dimensional dense features).

**Matching loss** uses reciprocal matching M = {(i,j) | j=NN₂(D^1_i) and i=NN₁(D^2_j)} with:
```
NN_A(D^B_j) = argmin_i ‖D^A_i − D^B_j‖₂
```
Loss weight: β=1 for matching loss, α=0.2 for confidence.

#### 0.2.2 Coarse-to-Fine Strategy

Initial matching at coarse resolution → refine with local offset regression. Critical for accuracy (Table 7 in paper).

#### 0.2.3 Outputs

- All DUSt3R outputs: X^{1,1}, X^{2,1} ∈ ℝ^{H×W×3} + confidences C^{v,1}
- PLUS dense descriptor maps D^v ∈ ℝ^{H×W×24} for feature matching
- Same coordinate frame as DUSt3R (camera-1 frame)

#### 0.2.4 License & Checkpoints

- **Code:** CC BY-NC-SA 4.0 (github.com/naver/mast3r) — non-commercial only
- **Checkpoint NOTICE:** Must agree to all constituent training dataset licenses (mapfree is very restrictive)
- **Main checkpoint:** `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` on HuggingFace naver/MASt3R

---

## 1. MonST3R — Motion DUSt3R

**Paper:** Zhang et al., "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion," ICLR 2025; arXiv:2410.03825  
**Code:** github.com/Junyi42/monst3r  
**License:** CC BY-NC-SA 4.0 — **non-commercial research only**

### 1.1 Core Idea and Architecture

MonST3R adapts DUSt3R to **dynamic scenes** by the key insight that a pointmap can represent per-timestep geometry: at time t, X^t_{i,j} is the 3D location of pixel (i,j) at that instant, whether or not it belongs to a moving object. The network architecture is identical to DUSt3R (ViT-Large encoder + ViT-Base decoder + DPT head) but the decoder and head are **finetuned** on dynamic video data.

**What is frozen vs. finetuned (Paper §3.2):**
- **Frozen:** ViT-Large encoder (preserves CroCo geometric pre-training)
- **Finetuned:** Both ViT-Base decoders + DPT prediction heads
- Training: 25 epochs, 20,000 image pairs/epoch, AdamW lr=5×10⁻⁵, batch=4, 2× RTX 6000 GPUs, ~1 day

**Training datasets (Table 2, §4.1):**
- PointOdyssey: 10,000 pairs/epoch (dynamic, synthetic, articulated motion)
- TartanAir: 5,000 pairs/epoch (static, synthetic, drone)
- Spring: 1,000 pairs/epoch (dynamic, synthetic)
- Waymo: 4,000 pairs/epoch (dynamic, real-world driving with LiDAR)
- Temporal stride: 1–9 frames (larger strides upweighted)

### 1.2 Per-Frame Pointmap Output

For an image pair (I^t, I^{t'}) at different timestamps:
```
(X^{t;t'}, C^{t;t'},  X^{t';t't'}, C^{t';t't'}) = f_MonST3R(I^t, I^{t'})
```
- **X^{t;t'}** ∈ ℝ^{H×W×3}: 3D locations of pixels in I^t, in camera frame of I^t, at time t
- **X^{t';t't'}** ∈ ℝ^{H×W×3}: 3D locations of pixels in I^{t'}, also in camera frame of I^t
- Both confidence maps C ∈ ℝ^{H×W}
- For dynamic pixels: X^{t;t'} and X^{t';t't'} can differ even at same pixel location because the scene moved between t and t'

### 1.3 Static/Dynamic Separation (Paper §3.3)

**Static mask computation (Eq. 2):**
```
S^{t→t'} = [ α > ‖F^{t→t'}_{cam} − F^{t→t'}_{est}‖₁ ]
```
where:
- **F^{t→t'}_{cam}** = camera-induced optical flow (rigid background flow computed from estimated camera pose)
- **F^{t→t'}_{est}** = optical flow from an external estimator (e.g., RAFT or FlowFormer)
- Pixels where the residual is below threshold α are classified as **static**
- Dynamic pixels: the residual exceeds α, indicating independent motion

The static mask is computed iteratively: initial camera pose from network, compute F_cam, compare to F_est, update mask, refine pose.

**Camera pose from RANSAC+PnP (§3.3):**
Relative pose P=[R|T] estimated via RANSAC on the pairwise pointmaps. Because RANSAC draws random samples, mostly-static scenes bias samples toward static pixels, making the pose estimate robust to dynamic objects.

### 1.4 Video Optimization / Global Alignment (Paper §3.4)

For a T-frame video, build a video graph:
- **Sliding temporal window** of size w (default w=9) with stride sampling
- For a 60-frame video: ~600 image pairs (not O(T²) all-pairs)

**Joint optimization objective (Eq. 6):**
```
X̂ = argmin  L_align(X, σ, P_W) + w_smooth · L_smooth(X) + w_flow · L_flow(X)
```

Components:
- **L_align** (Eq. 3): Σ ‖C^{t;e} · (X^t − σ_e P^{t;e} X^{t;e})‖₁ — standard DUSt3R alignment
- **L_smooth** (Eq. 4): Frobenius norm on rotation differences + L2 on translation differences (temporal smoothness)
- **L_flow** (Eq. 5): ‖F̂^{a→b} − F^{a→b}‖₁ weighted by (1−M^a) — optical flow consistency on static regions only

Solver: Adam, 300 iterations, lr=0.01, ~1 minute for 60-frame video on single RTX 6000 GPU.

### 1.5 Camera Pose Recovery

**Intrinsics** (§3.3): focal length f^t computed from X^{t;t't} (in its own camera frame) by fitting pinhole model to the pointmap.
**Extrinsics**: accumulated from pairwise R|T estimates into global trajectory. Output in first-frame world coordinate.

### 1.6 Output Summary

| Output | Shape | Frame / Scale |
|--------|-------|---------------|
| Per-frame pointmap X^t | H×W×3 | World frame (first frame), up-to-scale |
| Per-frame confidence C^t | H×W | — |
| Camera pose P^t | 4×4 SE(3) | World frame (first frame origin) |
| Intrinsics K^t | 3×3 | Per-frame focal length |
| Static mask S^t | H×W | Binary |
| Depth map D^t | H×W | Up-to-scale (derived from X^t + K^t) |

**Scale:** Up-to-unknown-scale (no metric depth without metric GT in training). The global optimization recovers a consistent scale across the video.

### 1.7 Cross-Frame Correspondences / Tracking

MonST3R does **NOT** output explicit per-pixel correspondences or tracks. Implicit correspondences can be recovered by:
1. Projecting X^t into frame t' using the recovered camera pose and checking pixel proximity
2. Using the flow consistency in L_flow (optical flow gives 2D correspondences)
3. Third-party: use RAFT/FlowFormer tracks alongside MonST3R depth → lift 2D tracks to 3D

**Important:** MonST3R alone does NOT give per-Gaussian trajectories. You need to augment with a tracker.

### 1.8 Inference Cost & GPU Requirements

- **Input preprocessing:** Max dimension 512px (16:9 → 512×288, etc.)
- **Pairwise inference:** ~30 sec for 60-frame video (~600 pairs)
- **Global alignment:** ~1 min for 60-frame video
- **GPU memory:** ~33GB VRAM for 65-frame 16:9 video (batchified mode); ~23GB with `--not_batchify`
- **Memory scales quadratically** with window size × num_frames for the pair graph

### 1.9 Image Preprocessing

- Resize so max dimension ≤ 512 pixels while maintaining aspect ratio
- Center crop augmentation during training at multiple resolutions: (512,288), (512,384), (512,336)
- Color jitter augmentation
- RoPE positional embeddings (no explicit normalization/standardization required beyond standard ImageNet-style)

### 1.10 Checkpoint

`MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`  
Available: HuggingFace `Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt`, Google Drive  
Download: `cd data && bash download_ckpt.sh`

---

## 2. CUT3R — Continuous 3D Perception with Persistent State

**Paper:** Wang et al., "Continuous 3D Perception Model with Persistent State," CVPR 2025; arXiv:2501.12387  
**Code:** github.com/CUT3R/CUT3R  
**License:** CC BY-NC-SA 4.0 — **non-commercial research only**

### 2.1 Core Innovation: Recurrent State

Unlike DUSt3R/MonST3R which operate on discrete image pairs, CUT3R processes a **video stream** online by maintaining a persistent **state** s_t that encodes accumulated scene understanding.

**State representation (§3.1):**
- 768 tokens, each 768-dimensional = 589,824 learnable values
- Initialized before first frame; updated at each new observation
- Represents the "memory" of the entire video seen so far

**State update rule (Eq. 2):**
```
[z'_t, F'_t], s_t = Decoders([z, F_t], s_{t-1})
```
where:
- **F_t** = image tokens from ViT encoder for frame t
- **s_{t-1}** = state from previous step
- **z** = learnable pose token (1 token, captures ego-motion)
- **Decoders** = two interconnected transformer decoders with cross-attention between {image tokens, state tokens} at each block
- **s_t** = updated state; **F'_t** = enriched image tokens for prediction heads

### 2.2 Exact Output Tensors (Eqs. 3–5)

For each frame t, CUT3R outputs three things simultaneously:

```
X̂_t^{self}, C_t^{self}  =  Head_self(F'_t)          [pointmap in camera frame of t]
X̂_t^{world}, C_t^{world}  =  Head_world(F'_t, z'_t)  [pointmap in world frame]
P̂_t  =  PoseHead(z'_t)                               [camera pose, 6-DoF]
```

| Output | Shape | Meaning |
|--------|-------|---------|
| X̂_t^{self} | H×W×3 | 3D points in own camera frame |
| C_t^{self} | H×W | Confidence for X̂_t^{self} |
| X̂_t^{world} | H×W×3 | 3D points in world frame (frame-1 origin) |
| C_t^{world} | H×W | Confidence for X̂_t^{world} |
| P̂_t | 7-vector (quaternion + translation) | World-to-camera pose |

For **virtual view queries** (§3.2): a raymap R ∈ ℝ^{H×W×6} (Plücker coordinates: origin + direction per pixel) is encoded and passed to decoders, yielding pointmap X̂_r, confidence C_r, color Î_r.

### 2.3 Coordinate Frame Convention

- **World frame** = coordinate system of the first image I^1 (identical to MonST3R/DUSt3R convention)
- **X̂_t^{world}** is in this fixed world frame — NO post-processing alignment step needed
- The state implicitly maintains the alignment across all frames
- Output is metric-scale (the model trains on metric datasets)

### 2.4 Dynamic Content Handling

CUT3R uses **fully implicit handling** — no explicit static/dynamic masking. The recurrent state learns to separate camera motion from object motion during training. Training data includes dynamic datasets (BEDLAM, DynamicStereo, PointOdyssey) mixed with 32 total datasets.

Key quote from paper: "our method seamlessly handles videos of dynamic scenes, estimating accurate camera parameters and dense point clouds for moving parts."

**Limitation:** Moving objects do contribute to the world-frame pointmap — there is no explicit segmentation of dynamic vs. static. The world-frame pointmap for dynamic objects reflects their position at each frame's observation time.

### 2.5 Online vs. Offline Mode (§4.4)

- **Online (default):** Sequential processing; state updated left-to-right. 16.58 FPS at 512×144 on A100.
- **Offline (revisiting):** After processing all frames, freeze final state and reprocess sequence from start. Improves accuracy by incorporating future context. ~2× time cost.
- Causal constraint: online mode cannot use future frame information.

### 2.6 Cross-Frame Correspondences / Tracking

CUT3R does **NOT** produce explicit per-pixel cross-frame correspondences or tracks. The accumulation of world-frame pointmaps provides dense geometry per frame in a common frame, but there is no direct mechanism to identify which world-frame point at time t corresponds to which point at time t'. Must be combined with a 2D/3D tracker.

### 2.7 Inference Cost

- **Online:** 16.58 FPS at 512×144 resolution on A100 (Table 2 in paper)
- **Memory:** Linear in sequence length (state fixed size, image processing sequential)
- Training uses sequences up to 64 frames; inference tested on longer sequences
- **GPU requirement:** Training: 8× A100 80GB. Inference: likely works on 1× A100/3090 (not specified in paper)

### 2.8 Pose Loss (§3.3, Eq. 8)

```
ℒ_pose = Σ_t ( ‖q̂_t − q_t‖₂  +  ‖τ̂_t/ŝ − τ_t/s‖₂ )
```
where q is quaternion, τ is translation, s is scale normalization (analogous to DUSt3R's z).

### 2.9 Checkpoints

| Model | Resolution | Sequence Length | Head |
|-------|-----------|----------------|------|
| `cut3r_224_linear_4.pth` | 224×224 | up to 16 | Linear |
| `cut3r_512_dpt_4_64.pth` | 512 variants | 4–64 | DPT |

Download: Google Drive links in README

### 2.10 Image Preprocessing

- Stage 1-2 training: 224×224 images
- Stage 3: max side 512px, varied aspect ratios
- ViT encoder: 16×16 patches
- No explicit ImageNet normalization cited beyond CroCo standard

---

## 3. MegaSaM — Deep Visual SLAM for Casual Dynamic Video

**Paper:** Li et al., "MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos," CVPR 2025; arXiv:2412.04463  
**Code:** github.com/mega-sam/mega-sam  
**License:** Apache 2.0 (code) + CC-BY 4.0 (other materials) — **commercial use permitted**  
**Note:** "This is not an official Google product"

### 3.1 Architecture: Three-Stage Pipeline

MegaSaM is a **SLAM-based** system (not a pointmap regression network). It extends differentiable bundle adjustment to dynamic scenes.

**Stage 1 — Frontend (keyframe SLAM):**
- Frame selection and sliding-window BA for keyframe registration
- Outputs: rough camera poses for keyframes, optical flow correspondences

**Stage 2 — Backend (global BA):**
- Global bundle adjustment across all frames
- Uncertainty-aware regularization with dynamic masking

**Stage 3 — Depth Refinement:**
- Optional consistent video depth optimization (CVD)
- Fuses monocular depth priors with BA results

### 3.2 Dynamic Content Handling (Paper §3.2.1)

A separate **motion probability network** F_m predicts per-pixel dynamic masks:

```
m_i ∈ ℝ^{H/8 × W/8}  =  F_m({I_i} ∪ N(i))
```
where N(i) is a neighborhood of temporally adjacent frames.

These masks are element-wise multiplied with pairwise confidence weights:

```
w̃_{ij}  =  ŵ_{ij}  ⊙  m_i
```

This **down-weights dynamic pixels** in the BA loss, making pose estimation robust to moving objects.

**Training for F_m:**
1. Pretrain ego-motion on static scenes (TartanAir, 163 scenes)
2. Freeze flow network, finetune F_m on dynamic synthetic videos (Kubric: 5K static + 11K dynamic)
3. Loss: binary cross-entropy on dynamic mask predictions

### 3.3 Optimization Objectives

**Reprojection cost (Eq. 2):**
```
C(Ĝ, d̂, f̂)  =  Σ_{(i,j)∈P}  ‖û_{ij} − u_{ij}‖²_{Σ_{ij}}
```
where û_{ij} = π(Ĝ_{ij} ∘ π^{-1}(p_i, d̂_i, K^{-1}), K) is the reprojection.

**Frontend cost (Eq. 9):**
```
C = Σ_{(i,j)∈P}  ‖û_{ij} − u_{ij}‖²_{Σ_{ij}}  +  w_a Σ_i ‖d̂_i − D^{align}_i‖²
```
Alignment term pulls disparity toward monocular depth prior.

**Consistent video depth (Eq. 11):**
```
C_cvd  =  w_flow · C_flow  +  w_temp · C_temp  +  w_prior · C_prior
```

**Solver:** Levenberg–Marquardt with differentiable Schur complement (Eqs. 5–6).

### 3.4 Outputs

| Output | Shape | Meaning |
|--------|-------|---------|
| Camera poses Ĝ_i | SE(3) 4×4 | World frame trajectory |
| Focal length f̂ | scalar | Shared across video |
| Disparity d̂_i | H/8 × W/8 | Low-res (during BA) |
| Depth D̂_i | H×W | Full-res after refinement |
| Uncertainty maps | H×W | Aleatoric depth uncertainty |
| Dynamic mask m_i | H/8 × W/8 | Probability of pixel being dynamic |

**Coordinate convention:** Standard SE(3) / pinhole model. World = first frame origin.

### 3.5 Cross-Frame Correspondences

MegaSaM uses **learned optical flow** (convolutional GRU, Eq. 1) for correspondences within BA. The flow network outputs:
```
(û^{k+1}_{ij}, ŵ^{k+1}_{ij})  =  F(I_i, I_j, û^k_{ij}, ŵ^k_{ij})
```
These 2D correspondences are used only internally for BA. No explicit 3D tracking output.

### 3.6 Monocular Depth Prior (Key for Low-Parallax Videos)

```
D^{align}_i  =  α̂ · D^{rel}_i  +  β̂
```
- D^{rel}_i: relative depth from DepthAnything-V2
- Metric prior: UniDepth predictions used for initial focal length f̂ (median across video)
- α̂, β̂: per-video scale/shift estimated from median alignment

### 3.7 Inference Cost

- Average ~1.0 s per frame (on Sintel: 20–50 frames; DyCheck: 180–500 frames)
- Depth optimization: 1.3 FPS at 336×144
- No per-video finetuning required
- Input resolution: 672×288 (for depth output)

### 3.8 Image Preprocessing

- No explicit undistortion (assumes pinhole)
- Focal length initialized from UniDepth median estimate
- First two poses fixed to ground truth for gauge freedom removal (or set to identity)
- CUDA 11.8 / PyTorch 2.0.1 / xformers required

### 3.9 Scale

**Metric depth** (via UniDepth prior) — output depth is approximately metric, not up-to-scale.

---

## 4. St4RTrack — Simultaneous 4D Reconstruction and Tracking

**Paper:** Feng et al., "St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World," ICCV 2025; arXiv:2504.13152  
**Code:** github.com/HavenFeng/St4RTrack  
**License:** Non-commercial scientific research only (explicit restriction in repo)  
**Checkpoint:** `St4RTrack_Seqmode_reweightMax5.pth` on HuggingFace `yupengchengg147/St4rTrack`

### 4.1 Core Idea: Dual-Branch Pointmap Network

St4RTrack is the **most directly relevant method for per-Gaussian trajectories**. It jointly reconstructs 3D geometry AND tracks 3D points across time, in a single forward pass.

**Key insight:** Two pointmaps are defined at the same spatial origin but different times. A **tracking branch** predicts where the physical content of frame i moves at time j. A **reconstruction branch** predicts the geometry of frame j.

### 4.2 Pointmap Notation (Paper §3.1)

Notation: **X^b_t^a** = "3D pointmap of physical content from frame b, at time t, expressed in coordinate system of frame a."

For a pair (I^i, I^j), the network learns:
```
f_St4R(I^i, I^j)  →  (X^i_j,  X^j_j,  C^i_j,  C^j_j)
```

- **X^i_j ∈ ℝ^{H×W×3}** (tracking pointmap): 3D locations of content visible in frame i, BUT at time j (i.e., where those pixels moved to at time j), expressed in frame i's coordinates
- **X^j_j ∈ ℝ^{H×W×3}** (reconstruction pointmap): 3D locations of content visible in frame j at time j, also in frame i's coordinates

**3D scene flow** is immediately derivable:
```
flow_{i→j}  =  X^i_j − X^i_i
```
where X^i_i is obtained by running f_St4R(I^i, I^i) or by the self-reconstruction pass.

### 4.3 Architecture (§3.2)

- **Encoder:** Shared ViT (same as MASt3R)
- **Decoder:** Siamese dual-branch transformer decoder with alternating self-attention + cross-attention
- "Continuous information flow between two branches is crucial for generating spatial-aligned 3D pointmaps in a shared coordinate system"
- Initialized from **MASt3R checkpoint**

### 4.4 Coordinate Frame

First frame I^1 defines the world coordinate system. For a video of T frames:
```
Process: f(I^1, I^1), f(I^1, I^2), ..., f(I^1, I^T)
```
Using frame 1 as the anchor enables **long-range correspondences** by chaining: pixel at (i,j) in I^1 → its 3D location at time t is X^1_t(i,j) directly. No drift accumulation.

### 4.5 Training (§4.1)

**Synthetic datasets:**
- PointOdyssey: 9,800 sequences
- Dynamic Replica: 8,500 sequences
- Kubric: 5,700 sequences
- Frame sampling: 24 frames with stride 1–6

**Configuration:** AdamW lr=5×10⁻⁵, batch=1 per GPU, 4× A100 80GB, 50 epochs (~1 day)

**Supervision:**
- Reconstruction branch: per-frame depth maps + GT camera poses
- Tracking branch: mesh vertices in world coordinates (sparse masked supervision)
- Reprojection loss (Eqs. 5–7): 3D tracks project into frame j via PnP; compared against CoTracker3 pseudo-GT

**Adaptation loss (Eq. 11):**
```
L_reproj  =  L_traj  +  λ₁ · L_depth  +  λ₂ · L_align
```

### 4.6 Cross-Frame Correspondences ← KEY FEATURE

**St4RTrack provides EXPLICIT 3D tracking.** For each pixel in frame 1, you get its 3D position at every subsequent time step → this IS the per-point trajectory.

To get trajectories for N pixels in I^1 across T frames:
1. Run f(I^1, I^t) for each t=1..T → get X^1_t(i,j) for all (i,j)
2. This gives pixel-level 3D tracks without any post-processing

### 4.7 Inference Cost

- **30 FPS** on RTX 4090 (pair-wise forward pass)
- **Test-time adaptation (optional):** ~5 minutes on 4× A100 for 500 optimization steps
- Pair-wise approach works on arbitrarily long videos (no fixed-length constraint)

### 4.8 Image Preprocessing

- Focal length estimated using Weiszfeld algorithm (fast iterative solver)
- Standard pinhole assumption (square pixels, centered principal point)
- Input resolution: not specified, likely inherits MASt3R preprocessing (max dim 512px)

---

## 5. D²USt3R — Static-Dynamic Aligned Pointmaps

**Paper:** Lee et al., "D²USt3R: Enhancing 3D Reconstruction with 4D Pointmaps for Dynamic Scenes," NeurIPS 2025; arXiv:2504.06264  
**Code:** github.com/cvlab-kaist/DDUSt3R  
**License:** Not specified (NeurIPS 2025, research-grade)

### 5.1 Core Idea: Static-Dynamic Aligned Pointmaps (SDAP)

D²USt3R introduces **SDAP**: a single pointmap that simultaneously captures static background and dynamic object geometry. Instead of predicting only the geometry at the current timestamp, SDAP aligns dynamic pixels via optical flow to bring them into the reference frame.

**Formal definition (§4.2):**
- For static pixels: aligned via camera pose transformation (rigid motion)
- For dynamic pixels: aligned via backward optical flow b (non-rigid motion)

**Alignment equations (Eq. 4–5):**
- Occlusion mask (forward-backward consistency):
```
p₂' = p₁ + f(p₁)
p₁' = p₂' + b(p₂')
M_occ = [|p₁' − p₁| > t]
```
- Dynamic mask (residual motion):
```
f_cam = π(D·K·R·K^{-1}·p + K·T) − p
M_dyn = [‖f_cam − f‖ > τ]
```

### 5.2 Outputs (§4.2, 4.4)

| Output | Shape | Meaning |
|--------|-------|---------|
| X^{1,1} | H×W×3 | Pointmap of I^1 in camera-1 frame |
| X^{2,1} | H×W×3 | Pointmap of I^2 in camera-1 frame (SDAP-aligned) |
| C^{v,1} | H×W | Confidence maps |
| M̂_dyn | H×W×1 | Predicted dynamic mask (logits) |
| Optical flow field | H×W×2 | 2D flow (RAFT-based head) |

### 5.3 Loss Functions (Eqs. 6–9)

**Static region loss:**
```
L_static  =  Σ_v Σ_i  C^{v,1} · ‖(1/z)X^{v,1}_i − (1/z̄)X̄^{v,1}_i‖  −  α·log(C^{v,1}_i)
```
Applied where (1−M_dyn) = 1.

**Dynamic region loss (Eq. 8):**
```
L_dyn  =  Σ_i  M²_dyn,i · (1−M²_occ,i) · C²_i · ‖(1/z̄₁)X̄^{1,1}_{i+b(i)} − (1/z₁)X²_i‖  −  α·log(C²_i)
```
Aligns dynamic pixel at location i+b(i) in I^1 to corresponding point in I^2.

**Total loss:** L_total = L_static + L_dyn + L_dyn_mask + L_flow

### 5.4 Architecture Differences from MonST3R/DUSt3R

- Freezes encoder, finetunes decoder + DPT head
- Adds two auxiliary heads: dynamic mask prediction (DPT-based) and optical flow (RAFT-based with cross-attention instead of 4D correlation volumes)
- Based on MonST3R/DUSt3R codebase

### 5.5 Cross-Frame Correspondences

**Indirect tracking:** Dynamic region alignment via SDAP provides flow-based 2D correspondences. 3D correspondences derivable from flow + depth but NOT native 3D tracking output.

**Limitation:** Pairwise only — extends to video via global optimization (MonST3R-style), not built-in.

### 5.6 Coordinate Frame & Scale

Same as DUSt3R: camera-1 frame, up-to-scale.

---

## 6. Easi3R — Training-Free Attention Adaptation of DUSt3R

**Paper:** Chen et al., "Easi3R: Estimating Disentangled Motion from DUSt3R Without Training," ICCV 2025; arXiv:2503.24391  
**Code:** github.com/Inception3D/Easi3R  
**License:** CC BY-NC-SA 4.0 — **non-commercial research only**

### 6.1 Core Idea: Attention Disentanglement Without Training

The key insight is that DUSt3R's cross-attention layers **implicitly encode epipolar constraints**: tokens that violate epipolar geometry receive low cross-attention values. Dynamic objects violate rigid epipolar geometry → they get low attention. By extracting and re-weighting these attention maps, dynamic regions can be identified without any training on dynamic data.

### 6.2 Attention Adaptation Mechanism (Paper Eqs. 4–10)

**First inference pass — attention extraction:**

Cross-attention maps from decoder blocks l:
```
A^{a←b}_l  =  Q^a_l · (K^b_l)^T / √c
```

Spatial aggregation (Eq. 5): average across decoder layers and spatial dimensions → A^{b=src}, A^{a=ref} ∈ ℝ^{h×w}

Temporal aggregation (Eq. 6):
```
A^{b=src}_μ  =  Mean({A_i^{b=src}})    [across frame pairs]
A^{b=src}_σ  =  Std({A_i^{b=src}})
```

**Dynamic score computation (Eq. 9):**
```
A^{a=dyn}  =  (1 − A^{a=src}_μ) · A^{a=src}_σ · A^{a=ref}_μ · (1 − A^{a=ref}_σ)
```

**Thresholding:** Binary mask M^t = [A^{t=dyn} > α] via Otsu thresholding.  
Temporal consistency: k-means clustering (k=64) on encoder features F^t_0 across frames.

**Second inference pass — attention re-weighting:**

Mask out cross-attention for dynamic tokens (Eq. 10):
```
softmax(Ã^{a←b}_l)  =  { 0            if M^{a←b}   (dynamic pixel)
                         { softmax(A)   otherwise
```
where M^{a←b} = (1−M^a) ⊗ (M^b)^T (outer product masking).

This forces the reconstruction to be based only on static tokens → cleaner camera pose + static geometry.

### 6.3 Outputs

| Output | Shape | Meaning |
|--------|-------|---------|
| X^{a→a}, X^{b→a} | H×W×3 | Static pointmaps in reference frame |
| M^t | h×w | Binary dynamic mask |
| Camera poses | SE(3) | Per-frame via DUSt3R global alignment |
| (Video graph optimization) | — | Same as MonST3R sliding window |

### 6.4 Video Pipeline (Eq. 11 + 2)

Uses MonST3R-style sliding window video graph. Flow consistency loss:
```
L_flow  =  Σ (1−M^a) · ‖F̂_i^{a→b} − F_i^{a→b}‖₁
```
Weighted only on static regions (masked by M^a).

### 6.5 Cross-Frame Correspondences

No explicit 3D tracking. Static scene geometry accumulated into consistent point cloud. Dynamic masks identify which pixels are unreliable.

### 6.6 Key Advantage

**Zero additional training.** Uses any pretrained DUSt3R or MonST3R checkpoint. Minimal inference overhead (two forward passes instead of one, plus clustering).

### 6.7 Inference Cost

Described as "lightweight" — approximately 2× DUSt3R inference time. No specific runtime numbers given.

---

## 7. Geo4D — Video Diffusion for 4D Reconstruction

**Paper:** Jiang et al., "Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction," ICCV 2025 Highlight; arXiv:2504.07961  
**Code:** github.com/jzr99/Geo4D (research purposes)  
**License:** Not specified explicitly (research-only suggested by README)

### 7.1 Core Idea: Video Diffusion as Dynamic Prior

Geo4D fine-tunes **DynamiCrafter** (a latent video diffusion model) to output multi-modal geometric representations simultaneously. The video generator provides a strong prior over object motion and camera dynamics.

**Training objective (Eq. 2):**
```
min_θ  𝔼  ‖ε^{1:N} − ε_θ(z^{1:N}_t, t, y)‖²₂
```
where z^{1:N} = VAE-encoded geometric outputs, y = conditioning (first frame + CLIP embedding).

### 7.2 Multi-Modal Geometric Outputs (§3.2, Eq. 1)

Per-frame, simultaneously outputs:

| Modality | Shape | Meaning |
|----------|-------|---------|
| Point map X^i | H×W×3 | 3D coords in frame-1 reference frame |
| Disparity map D^i | H×W×1 | Inverse depth |
| Ray map r^i | H×W×6 | Plücker coordinates (origin d_uv + moment m_uv) |

All modalities predicted from the latent diffusion output via the VAE decoder.

**Coordinate frame:** All point maps in reference frame of camera 1. Static parts of different frames align in this frame.

### 7.3 Multi-Modal Alignment Algorithm (§3.3, Eqs. 4–10)

Three-stage optimization to recover camera parameters and fuse modalities:

**1. Point map alignment (Eq. 4):**
```
‖X^i_{uv} − λ_p^g · P_p^g · X^{i,g}_{uv} / σ^{i,g}_{uv}‖₁
```
Recovers K_p^i, R_p^i, o_p^i (camera center), D_p^i (disparity)

**2. Disparity alignment (Eq. 5):**
```
‖D^i_p − λ^g_d · D^{i,g}_d − β^g_d‖₁
```
Per-clip scale/shift

**3. Ray map alignment (Eqs. 6–8):**
Camera center: `argmin_p Σ ‖p × d^{i,g}_{uv} − m^{i,g}_{uv}‖²`  
Rotation: RQ-decomposition

**Joint loss (Eq. 10):**
```
L_all  =  α₁L_p + α₂L_d + α₃L_c + α₄L_s
```
L_s = temporal smoothness on rotation and camera centers.

### 7.4 Sliding Window for Long Videos (§3.3)

- Divide into V=16 frame clips with stride s=4
- Starting indices: S = {0, s, 2s, ..., ⌊(N−V)/s⌋s} ∪ {N−V}
- Each clip processed independently by diffusion (DDIM, 5 steps)
- Results fused via joint multi-modal alignment

### 7.5 Dynamic Content Handling

**Implicit** — no explicit dynamic masking. The video diffusion model's learned prior encodes typical object motion patterns. Dynamic objects form a "3D trace of the motion" in the point maps. No segmentation.

### 7.6 Inference Cost

- ~1.9 sec/frame (vs MonST3R ~2.41 sec/frame at comparable quality)
- Diffusion: 5 DDIM steps per clip (fast)
- Training: 4× H100, ~1 week

### 7.7 Cross-Frame Correspondences

**Implicit only.** Temporal consistency enforced via L_s. No explicit per-pixel correspondences output.

### 7.8 Checkpoint

Two models: fine-tuned VAE (`gdown 10SPKkOpou2lKl9bwkgx1d6YocYkmSxQl`) and full model (`gdown 11K0ubqytun-SA5RIOgR7ejNIR8B4uois`).

---

## 8. Driv3R — Dense 4D Reconstruction for Autonomous Driving

**Paper:** Zheng et al., "Driv3R: Learning Dense 4D Reconstruction for Autonomous Driving," arXiv:2412.06777  
**Code:** github.com/Barrybarry-Smith/Driv3R  
**License:** MIT License — **commercial use permitted**

### 8.1 Core Idea: Memory-Based Multi-View Temporal Integration

Driv3R extends Spann3R (spatial memory) to handle **multi-camera + temporal** sequences for autonomous driving. Key difference from MonST3R: optimization-free (no post-hoc BA) via memory pool.

### 8.2 Memory Pool Architecture (§3.2, Eqs. 1–2)

Per-sensor memory with cross-attention:
```
f*_{t,c}  =  softmax(q_{t,c} K^T / √s · V)  +  q_{t,c}
```
- **Working memory:** 5 most recent frames (stage 1) or 5 most similar frames by cosine similarity (stage 2)
- **Long-term memory:** Pruned by accumulated attention weights
- Separate memory pool per sensor/camera

### 8.3 4D Flow Predictor (§3.3, Eqs. 3–4)

Uses RAFT optical flow + camera motion to compute per-pixel dynamic probability:
```
E^i_{12}  =  K^{(e)}_{i_2} (T^{(e)}_{i_2})^{-1} P^{(e)}_{i_1}  −  K^{(e)}_{i_1} (T^{(e)}_{i_1})^{-1} P^{(e)}_{i_1}
Ω'_t  =  (1/N) [Σ‖F^i_{12} − E^i_{12}‖ + Σ‖F^i_{21} − E^i_{21}‖]
```
Refined with SAM2 segmentation → binary dynamic masks {Ω₁, ..., Ω_T}.

### 8.4 Outputs (§3.2, Eq. 2; §3.4, Eq. 5)

Per-frame per-camera:
- **P_{t,c}** ∈ H×W×3: pixel-wise 3D pointmap (in relative camera frame)
- **C_{t,c}** ∈ H×W: confidence map

World frame conversion (Eq. 5):
```
P^{(w)}_{t,c}  =  T_{t,c} K_{t,c}^{-1} K^{(e)}_{t,c} (T^{(e)}_{t,c})^{-1} P_{t,c}
```
where T_{t,c}, K_{t,c} are GT world camera parameters. This step requires **known camera calibration** to produce world-frame points.

### 8.5 Dynamic Content Handling

Stage 2 training supervised on dynamic regions: L = L^dynamic_conf + L^dynamic_scale. Uses R3D3 depth predictions as pseudo-GT for moving objects.

### 8.6 Inference Cost

| Metric | Driv3R | MonST3R | DUSt3R |
|--------|--------|---------|--------|
| FPS | 4.55 | 0.19 | 0.40 |
| GPU Memory | 7.84 GB | 5.98 GB | 5.84 GB |
| Time (5 frames, 6 cams) | 13.18 s | 312.44 s | 178.31 s |

**15× faster** than methods requiring global alignment.

### 8.7 Critical Limitation for Our Use Case

Driv3R requires **known multi-camera calibration** (GT extrinsics T_{t,c}) to produce world-frame pointmaps. Input images are split into 224×224 patches (from 1600×900 nuScenes images). **Not designed for monocular uncalibrated single-camera input.**

### 8.8 Cross-Frame Correspondences

Memory pool implicit correspondences. No explicit 3D tracking output. 4D flow predictor gives 2D optical flow which can be lifted to 3D via pointmaps.

---

## 9. Additional Relevant Methods

### 9.1 Dynamic Point Maps (Oxford VGG; arXiv:2503.16318)

**Core:** Extends DUSt3R with 4 output pointmaps per pair: P_i(t_j, π₁) for i∈{1,2}, t_j∈{t₁,t₂}, all in camera-1 frame. Enables direct scene flow computation: flow = P₁(t₂,π₁) − P₁(t₁,π₁). Code/models at robots.ox.ac.uk/~vgg/research/dynamic-point-maps/. Research use only (Oxford).

### 9.2 DePT3R (arXiv:2512.13122)

**Core:** Single-forward-pass joint dense 3D point tracking + reconstruction. Outputs pointmaps X̂^t_{t'} (content of frame t at time t') AND motion maps M̂^q_t (=X̂^q_t^1 − X̂^t^t^1). Efficient: 12GB handles 268K query points. Code: github.com/StructuresComp/DePT3R. License not specified.

### 9.3 VGGT (Wang et al., CVPR 2025; arXiv:2503.11651) — For Comparison

**Static scene assumption:** VGGT explicitly assumes static scenes in training. Its performance degrades on dynamic content — shallow transformer layers capture some motion cues but the model cannot robustly decouple moving objects from static background. Extensions: VGGT4D, PAGE-4D try to adapt VGGT to dynamic scenes but with workarounds. **Not recommended for dynamic robot video.**

---

## 10. Critical Synthesis: Recommendation for Robot Manipulation Pipeline

### 10.1 Problem Requirements Recap

Given: AgiBot World robot videos (30fps, 480×640 pinhole, uncalibrated, highly dynamic — arm + objects).

Need:
1. **(i)** Per-frame depth/pointmaps in a single consistent world frame
2. **(ii)** Camera trajectory (head camera pose vs. world)
3. **(iii)** Per-point 3D correspondence/tracking over time → per-Gaussian motion = ground-truth 3D scene flow

### 10.2 Why Static-Scene Methods (VGGT, Pi3) Fail

- VGGT and Pi3D are trained on static-scene datasets. Their multi-view consistency assumptions require that the same 3D point appears in multiple views — violated when the scene (arm, objects) moves.
- If you run VGGT/Pi3 on a window of 10-20 frames with substantial arm motion: the "static scene" constraint will pull the arm pointmaps into inconsistent positions, creating ghost artifacts and corrupted depth.
- The only valid use: sliding windows short enough that arm displacement is < ~5% of field of view. This is very restrictive (1-3 frames at 30fps for typical robot motion).

### 10.3 Method Comparison

| Method | World-Frame Pointmaps | Camera Pose | Explicit 3D Tracking | Dynamic Handling | License | Inference Speed |
|--------|-----------------------|-------------|---------------------|------------------|---------|----------------|
| **MonST3R** | ✓ (global opt) | ✓ RANSAC+PnP | ✗ (implicit) | Static mask + flow | CC BY-NC-SA | ~1 min/60 frames |
| **CUT3R** | ✓ (native world frame) | ✓ (native) | ✗ | Implicit (learned) | CC BY-NC-SA | 16.58 FPS online |
| **MegaSaM** | ✓ (BA-based, metric) | ✓ (metric, SE(3)) | ✗ | F_m dynamic mask | Apache 2.0 | ~1 FPS |
| **St4RTrack** | ✓ | ✓ | ✓ EXPLICIT | Dual-branch | NC research | 30 FPS |
| **D²USt3R** | ✓ (pairwise) | via global opt | Indirect (flow) | SDAP alignment | NC (unspecified) | Similar to MonST3R |
| **Easi3R** | ✓ | ✓ | ✗ | Attention masking | CC BY-NC-SA | ~2× DUSt3R |
| **Geo4D** | ✓ (diffusion) | ✓ | ✗ | Implicit (diffusion) | Research only | ~1.9 s/frame |
| **Driv3R** | ✓ (needs GT calib) | requires GT | ✗ | 4D flow + SAM2 | MIT | 4.55 FPS |

### 10.4 Recommended Pipeline

#### RECOMMENDED APPROACH A: MonST3R + St4RTrack (BEST FOR GROUND-TRUTH TRAJECTORIES)

**Step 1 — Camera Pose + Static Background Geometry: MonST3R**
```
Input: all T frames of video
Process:
  - Sliding window pairs (w=9) → MonST3R pairwise inference
  - Video graph global optimization (300 iter, Adam)
Output:
  - Camera poses P^t ∈ SE(3), t=1..T [world frame = frame 1]
  - Static background pointmap X_static ∈ ℝ^{H×W×3} per frame
  - Static mask S^t ∈ {0,1}^{H×W}
  - Intrinsics K^t (estimated from pointmap)
Runtime: ~30s pairwise + ~1min alignment for 60 frames
GPU: ~23-33GB VRAM
```

**Step 2 — 3D Scene Flow / Per-Pixel Tracking: St4RTrack**
```
Input: all T frames (using I^1 as anchor)
Process:
  - For each t=2..T: run f_St4R(I^1, I^t) → (X^1_t, X^j_t)
  - Chain or directly query: X^1_t(i,j) = 3D position of pixel (i,j) from I^1 at time t
Output:
  - Tracking pointmap X^1_t ∈ ℝ^{H×W×3} for each t [anchor frame coordinates]
  - 3D scene flow: Δ^t(i,j) = X^1_t(i,j) − X^1_1(i,j)
Runtime: 30 FPS → 1/30 s per frame pair on RTX 4090
GPU: RTX 4090 compatible
```

**Step 3 — Fusion**
```
- Use MonST3R P^t to transform St4RTrack (anchor-frame) outputs into global world frame
- Static regions: use MonST3R X_static (smoother, more accurate for background)
- Dynamic regions: use St4RTrack X^1_t (explicit temporal tracking)
- Per-pixel assignment via static mask S^t
```

**Limitation:** Both are CC BY-NC-SA (non-commercial). For commercial: MegaSaM (Apache 2.0) + separate tracker.

#### RECOMMENDED APPROACH B: CUT3R (Online, Lower Latency)

- CUT3R at 16 FPS gives world-frame pointmaps natively without post-optimization
- For a robot controller that needs real-time 3D: CUT3R is the practical choice
- **No explicit tracking** — must be augmented with CoTracker3 or similar
- License: CC BY-NC-SA

#### RECOMMENDED APPROACH C: MegaSaM (Commercial Use, Metric Depth)

- Only method with **metric depth** (via UniDepth prior) and **Apache 2.0 license**
- Dynamic masking (F_m) is explicit and trained
- No explicit tracking, no pointmaps (depth + poses only → can be lifted)
- Recommended if: (a) metric scale required, (b) commercial use needed

### 10.5 How to Get Per-Gaussian 3D Trajectories (Ground Truth for Dynamics Model)

**The fundamental challenge:** A dynamics model trained on 3D Gaussians needs, for each Gaussian g at time t, its 3D position x_g(t) and velocity ẋ_g(t) (or displacement Δx_g(t) = x_g(t+1) − x_g(t)).

**Procedure:**

1. **Initialize Gaussians** from MonST3R world-frame pointmap at t=0: each pixel (i,j) → one Gaussian with center x_g = X^0(i,j), color from I^0(i,j)

2. **Assign trajectories via St4RTrack:**
   - For each Gaussian g at pixel (i,j) in I^1:
     - St4RTrack gives X^1_t(i,j) directly = 3D position of that pixel's content at time t
     - In the MonST3R world frame: apply P^1 rotation to get x_g(t)
   - Result: trajectory {x_g(t)}_{t=0}^{T-1} for every Gaussian

3. **Handle occlusions and merging:**
   - Pixels occluded at time t: X^1_t will have low confidence C^1_t
   - Use confidence threshold: only use tracks where C^1_t > τ
   - For new objects entering field of view: need to re-initialize Gaussians from X^t_t

4. **3D scene flow ground truth:**
   - v_g(t) = x_g(t+1) − x_g(t) = X^1_{t+1}(i,j) − X^1_t(i,j)
   - This is the per-Gaussian velocity / scene flow
   - This can be used as ground-truth supervision for the dynamics model

5. **Alternatively:** Use Dynamic Point Maps (Oxford VGG) which directly outputs scene flow = P₁(t₂,π₁) − P₁(t₁,π₁) per pixel pair without post-processing.

### 10.6 Comparison: MonST3R+St4RTrack vs. VGGT/Pi3D on Static Windows

| Criterion | VGGT/Pi3 (per short window) | MonST3R + St4RTrack |
|-----------|---------------------------|---------------------|
| Handles dynamic arm | ✗ (assumes rigid) | ✓ (explicit masking) |
| Per-frame depth | ✓ | ✓ |
| Consistent world frame | Need stitching | ✓ native |
| Camera pose | ✓ (DBA) | ✓ (RANSAC+align) |
| Per-Gaussian tracking | ✗ (no temporal) | ✓ (St4RTrack) |
| Max frames (reasonable) | 10-20 per window | 60-300+ |
| Window stitching needed | Yes (complex drift) | No |
| License | MIT (VGGT) | CC BY-NC-SA |
| Inference speed | Very fast (VGGT) | Moderate |

**VERDICT:** For dynamic robot manipulation video, MonST3R + St4RTrack dominates VGGT/Pi3D on every dimension relevant to the task, except license. The only case to prefer VGGT/Pi3D short windows is if arm motion in each window is negligible (robot is nearly static), which is rare in meaningful manipulation demos.

### 10.7 Practical Recommendations for AgiBot World

**For the head camera (480×640, pinhole, uncalibrated, 30fps):**

1. **Preprocessing:** Resize to 480×640 → 512×288 (letterbox/crop to fit MonST3R's training resolution). No undistortion needed (already pinhole per problem spec).

2. **MonST3R inference:**
   - Window size w=9, stride 2 → ~4T pairs for T frames
   - GPU: Need 23+ GB VRAM (RTX 3090/4090 or A100)
   - For 300-frame sequence: ~8 min pairwise + ~5 min alignment

3. **St4RTrack inference:**
   - Anchor on frame 0 (or reset anchor every 50 frames for long sequences)
   - Run f(I^0, I^t) for t=1..T at 30 FPS
   - Output: per-pixel 3D trajectories relative to I^0

4. **Camera intrinsics recovery:**
   - MonST3R recovers K^t per frame
   - Focal length estimable from pointmap directly (Weiszfeld algorithm)
   - For 480×640 sensor: expect ~600-700px focal length for typical robot head cameras

5. **Scale:** MonST3R outputs are up-to-scale. For metric scale (needed to define Gaussian sizes): use MegaSaM alongside, or calibrate with known object size (robot arm link length).

---

## 11. Summary Table: All Methods

| Method | arXiv | Conference | Dynamic Type | Tracks | License | Metric |
|--------|-------|-----------|-------------|--------|---------|--------|
| DUSt3R | 2312.14132 | CVPR 2024 | ✗ static only | ✗ | CC BY-NC-SA | ✗ |
| MASt3R | 2406.09756 | ECCV 2024 | ✗ static only | ✗ (features) | CC BY-NC-SA | ✗ |
| MonST3R | 2410.03825 | ICLR 2025 | ✓ explicit mask | ✗ | CC BY-NC-SA | ✗ |
| CUT3R | 2501.12387 | CVPR 2025 | ✓ implicit | ✗ | CC BY-NC-SA | ✓ |
| MegaSaM | 2412.04463 | CVPR 2025 | ✓ explicit mask | ✗ | Apache 2.0 | ✓ |
| St4RTrack | 2504.13152 | ICCV 2025 | ✓ dual-branch | ✓ EXPLICIT | NC research | ✗ |
| D²USt3R | 2504.06264 | NeurIPS 2025 | ✓ SDAP | indirect | unspecified | ✗ |
| Easi3R | 2503.24391 | ICCV 2025 | ✓ attention | ✗ | CC BY-NC-SA | ✗ |
| Geo4D | 2504.07961 | ICCV 2025 | ✓ implicit | ✗ | research | ✗ |
| Driv3R | 2412.06777 | — | ✓ 4D flow | ✗ | MIT | ✗ |
| Dyn. Pt Maps | 2503.16318 | — | ✓ 4 pointmaps | via flow | research | ✗ |
| DePT3R | 2512.13122 | — | ✓ | ✓ EXPLICIT | unspecified | ✗ |

---

## References

- DUSt3R: Wang et al. arXiv:2312.14132; github.com/naver/dust3r
- MASt3R: Leroy et al. arXiv:2406.09756; github.com/naver/mast3r
- MonST3R: Zhang et al. arXiv:2410.03825; github.com/Junyi42/monst3r
- CUT3R: Wang et al. arXiv:2501.12387; github.com/CUT3R/CUT3R
- MegaSaM: Li et al. arXiv:2412.04463; github.com/mega-sam/mega-sam
- St4RTrack: Feng et al. arXiv:2504.13152; github.com/HavenFeng/St4RTrack
- D²USt3R: Lee et al. arXiv:2504.06264; github.com/cvlab-kaist/DDUSt3R
- Easi3R: Chen et al. arXiv:2503.24391; github.com/Inception3D/Easi3R
- Geo4D: Jiang et al. arXiv:2504.07961; github.com/jzr99/Geo4D
- Driv3R: Zheng et al. arXiv:2412.06777; github.com/Barrybarry-Smith/Driv3R
- Dynamic Point Maps: arXiv:2503.16318; robots.ox.ac.uk/~vgg/research/dynamic-point-maps/
- DePT3R: arXiv:2512.13122; github.com/StructuresComp/DePT3R
- VGGT: Wang et al. arXiv:2503.11651; vgg-t.github.io
