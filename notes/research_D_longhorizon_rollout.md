# Long-Horizon Autoregressive Rollout of Dynamic 3D Scenes / World Models

**Purpose:** Faithful technical reference for reimplementing long-horizon (10 s / 100+ frames at 10 Hz) rollout of per-Gaussian scene states conditioned on language, without drift or collapse.

---

## Table of Contents

1. [Drift / Error-Accumulation Mitigation Taxonomy](#1-drift--error-accumulation-mitigation-taxonomy)
2. [Per-System Training Recipes](#2-per-system-training-recipes)
   - 2.1 GameNGen
   - 2.2 DIAMOND
   - 2.3 Diffusion Forcing (DF)
   - 2.4 History-Guided Video Diffusion / DFoT
   - 2.5 Rolling Forcing
   - 2.6 Self Forcing
   - 2.7 Genie / Genie 2
   - 2.8 LIVE
   - 2.9 Navigation World Models (NWM)
   - 2.10 iVideoGPT
   - 2.11 Cosmos (NVIDIA)
   - 2.12 DreamerV3
3. [3D / Gaussian-Specific Dynamics](#3-3d--gaussian-specific-dynamics)
   - 3.1 Spacetime Gaussians (STG)
   - 3.2 EvoGS
   - 3.3 ODE-GS
   - 3.4 GaussianPrediction
   - 3.5 GWM (Gaussian World Model)
   - 3.6 GaussianGPT
4. [Temporal Consistency & Anti-Drift Regularizers for 3D Gaussians](#4-temporal-consistency--anti-drift-regularizers-for-3d-gaussians)
5. [Appearing / Disappearing Content in Gaussian Rollout](#5-appearing--disappearing-content-in-gaussian-rollout)
6. [Critical Synthesis: Recommended Rollout Recipe for Instruct-GS-World](#6-critical-synthesis-recommended-rollout-recipe-for-instruct-gs-world)

---

## 1. Drift / Error-Accumulation Mitigation Taxonomy

Autoregressive models face **exposure bias**: trained on ground-truth context, they must condition on their own noisy predictions at inference. Over 100+ steps errors compound, causing visual collapse. The field has converged on several orthogonal mitigation strategies:

### 1.1 Scheduled Sampling (Bengio et al., NIPS 2015)

**Mechanism:** During training, with probability $\epsilon_t$ replace ground-truth context tokens with the model's own predictions. $\epsilon_t$ is annealed from 0 → 1 over training (linear, inverse-sigmoid, or exponential decay). The model learns to correct its own errors.

**Limitation:** Does not change the test distribution of the *denoised* outputs; discrete scheduled sampling can be pathological with diffusion models.

### 1.2 Input Noise Augmentation of Context Frames (GameNGen, DIAMOND)

**Mechanism:** During training, corrupt the conditioning (context) frames by adding Gaussian noise $\mathbf{n} \sim \mathcal{N}(0, \sigma^2 I)$ before encoding them. The model receives a noise-level embedding alongside the corrupted context so it can learn to undo the corruption. At inference the self-generated previous frames act like slightly corrupted ground truth; if the noise level during training matches the expected self-generation error, the model is robust.

**Key insight (GameNGen):** The generated frame is similar to a lightly-corrupted version of the true next frame; training with matched noise teaches the model to denoise from this distribution instead of only from clean context.

**Limitation:** Choosing the right noise magnitude is empirical. Too high → model ignores context, too low → no robustness improvement.

### 1.3 Teacher-Forcing vs. Free-Running

**Teacher forcing:** At every step, feed the ground-truth previous frame. Perfect at training but collapses at inference (exposure bias).

**Fully free-running:** At every step during training, feed the model's own last prediction. Exact match to inference, but gradients through many AR steps are expensive and unstable.

**Hybrid / TBPTT:** Truncated backpropagation through time — unroll $k$ steps with the model's own predictions, compute loss, backprop only through the last $k$ steps.

### 1.4 Diffusion Forcing (Independent Per-Token Noise)

See Section 2.3 for full treatment. The core idea: each token in a sequence gets its own independently-sampled noise level $\sigma_i$. Because noising is equivalent to partial masking, the model learns to condition on tokens at arbitrary noise levels, which at inference enables generating new tokens while conditioning on (slightly noised) previous ones without catastrophic exposure bias.

### 1.5 Rolling / Sliding-Window Diffusion

Generate a window of $W$ tokens jointly via diffusion, slide by $S < W$ tokens, regenerate the new tokens while conditioning on the overlapping old tokens. Error resets every $S$ steps rather than compounding forever.

### 1.6 Cycle Consistency (LIVE)

Enforce that a forward rollout followed by a backward rollout recovers the initial state. This bounds the per-step error in a recoverable range.

### 1.7 Latent-Space Rollout (DreamerV3, RSSM)

Roll out entirely in a compact latent space (e.g., 32-cat × 32 RNN state in DreamerV3). Rendering only at selected keyframes. Errors are smaller in the regularized latent space; the decoder absorbs residuals.

### 1.8 Attention Sink + Global Context Anchor (Rolling Forcing)

Cache the KV states of the very first $L_\text{glo}$ frames and inject them into every future attention computation. This anchors color tone, exposure, and scene identity for arbitrarily long rollouts.

### 1.9 EMA of Model Weights

Maintain an Exponential Moving Average of weights with decay $\beta \in [0.999, 0.9999]$. Use EMA weights at inference. Smooths training instability. Used universally (DIAMOND, GameNGen, Cosmos).

---

## 2. Per-System Training Recipes

### 2.1 GameNGen

**Paper:** "Diffusion Models Are Real-Time Game Engines" (Valevski et al., arXiv 2408.14837, ICLR 2025)

**Architecture:** Stable Diffusion v1.4 (U-Net, unfrozen) fine-tuned on DOOM gameplay. 4M → 381M parameters for CS:GO version.

**Two-Phase Training:**
1. **RL Data Collection:** PPO agent trained 50 M env steps; 70 M (frame, action) pairs collected.
2. **Diffusion Model Training:** 700 k steps, batch 128, lr = 2e-5, Adafactor, gradient clip 1.0, 128 TPU-v5e, context dropout $p = 0.1$.

**Context Length:** 64 frames (~3.2 s at 20 FPS). Ablation over $N \in \{1,2,4,8,16,32,64\}$.

**Noise Augmentation (KEY):**
```
α ~ Uniform(0, α_max), α_max = 0.7
Discretize α into 10 embedding buckets
Context frames → encoded latents → add Gaussian noise at level α → feed to U-Net
U-Net also receives α as an input embedding
```
The maximum noise level 0.7 was found empirically. Context frames are concatenated channel-wise to noised prediction latents.

**Action Conditioning:** Each action → linear embedding → single token replacing text cross-attention tokens.

**Velocity Parameterization Loss:**
$$\mathcal{L} = \mathbb{E}\left[\|v(\epsilon, x_0, t) - v_\theta(x_t, t, \{\phi(o_{i<n})\}, \{A_\text{emb}(a_{i<n})\})\|_2^2\right]$$

**Inference:** DDIM, 4 steps, CFG guidance weight 1.5 (applied only on observation condition, not actions), 50 ms per frame, 20 FPS on single TPU.

**Stability Results:** After 5–10 minutes of autoregressive generation, human raters perform at chance (50–58%) distinguishing simulation from real game. PSNR 29.43, LPIPS 0.249 in teacher-forcing; FVD 114/186 at 16/32 AR steps.

**Anti-Drift Mechanism:** The noise augmentation is critical — without it, the model suffers fast visual degradation. The model learns to "denoise" its own slightly-noisy previous frames exactly as it denoises diffusion noise.

---

### 2.2 DIAMOND

**Paper:** "Diffusion for World Modeling: Visual Details Matter in Atari" (Alonso et al., arXiv 2405.12399, NeurIPS 2024)

**Architecture:** Standard U-Net 2D for vector field $\mathbf{F}_\theta$. History frames concatenated channel-wise to noisy next observation. Actions injected via adaptive group normalization inside residual blocks.

**EDM Preconditioning (critical for stability):**
$$\mathbf{F}_\theta(x_t, \sigma) = c_\text{skip}(\sigma) x_t + c_\text{out}(\sigma) F_\theta(c_\text{in}(\sigma) x_t, c_\text{noise}(\sigma))$$
The four scalar preconditioners maintain unit variance for any noise level $\sigma$. This prevents the network from learning the identity function at high noise (a DDPM failure mode).

**Training Loss (EDM formulation):**
$$\mathcal{L} = \mathbb{E}_\sigma \left[\lambda(\sigma) \|\mathbf{F}_\theta(x + \sigma\epsilon; \sigma) - x\|_2^2\right], \quad \lambda(\sigma) = ({\sigma^2 + \sigma_\text{data}^2}) / (\sigma \sigma_\text{data})^2$$

**Context Frames:** Buffer of $L$ past clean observations concatenated channel-wise. The paper conditions on **clean** past frames (no noise augmentation of context), relying instead on the EDM formulation's superior high-noise behavior for stability.

**Anti-Drift Mechanism:** EDM's clean target prediction at high $\sigma$ means the model never gets stuck predicting the identity — it always reverts to predicting a clean frame. This provides strong signal even when the denoising problem is underspecified, preventing error cascade.

**Inference:** Minimal denoising steps (n=3) sufficient due to EDM efficiency. Mean human normalized score 1.46 on Atari 100k.

**Scale:** 4M parameters (Atari), 381M (CS:GO). ~2.9 days on single RTX 4090, 12 GB VRAM.

---

### 2.3 Diffusion Forcing (DF)

**Paper:** "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion" (Chen et al., arXiv 2407.01392, NeurIPS 2024)

**Core Training Scheme:**
- Train a causal sequence model (RNN or masked transformer) to denoise tokens $\mathbf{x}_{1:T}$ where each token $\mathbf{x}_i$ is independently noised at noise level $\sigma_i \sim p(\sigma)$.
- The model observes the noise levels $\{\sigma_i\}$ and must predict all tokens simultaneously.
- Loss: standard DDPM $\epsilon$-prediction but per-token:
$$\mathcal{L}_\text{DF} = \mathbb{E}_{\sigma_{1:T}} \sum_{i=1}^T \lambda(\sigma_i) \|\epsilon_i - \epsilon_\theta(\mathbf{x}_{1:T}^{\sigma_{1:T}}, \sigma_{1:T})\|^2$$

**Two-Dimensional Noise Schedule Table:** A $(T, K)$ table where rows are sequence positions and columns are denoising steps. For a 3-frame sequence with 3 DDIM steps, the schedule progresses:
$$[x_1^{T/3}, x_2^{2T/3}, x_3^T] \to [x_1^0, x_2^{T/3}, x_3^{2T/3}] \to \ldots$$

**Causal Masking:** Implemented via causal RNN or masked attention — token $i$ can only attend to tokens $1, \ldots, i$. This allows the model to generate $x_{t+1}$ conditioned on already-generated $x_{1:t}$ (at noise level 0) and the noisy future $x_{t+2:T}$.

**Rollout Beyond Training Length:**
1. Generate segment $[x_1, \ldots, x_T]$ fully.
2. Set the RNN hidden state from the end of segment 1 as the initial state for segment 2.
3. Generate $[x_{T+1}, \ldots, x_{2T}]$ without re-initializing, enabling arbitrarily long rollouts.
No sliding window needed — continuous hidden state carries temporal context.

**Why It Prevents Drift:** At inference, previously generated clean tokens ($\sigma_i = 0$) act as perfect conditioning; the model was trained on tokens at all noise levels including 0, so it learned to condition on them correctly without distribution shift.

**Inference Variant (Stabilized Long Rollout):** Optionally add a small noise $\sigma_\text{hist} > 0$ to previously generated tokens before re-conditioning. This matches the training distribution more closely if the model was trained with $\sigma_\text{hist}$ lower-bound on historical noise.

**Video Generation:** Demonstrated generation of sequences much longer than training horizon while baselines diverge.

---

### 2.4 History-Guided Video Diffusion / DFoT

**Paper:** "History-Guided Video Diffusion" (Song et al., arXiv 2502.06764, ICML 2025)

**Architecture:** Diffusion Forcing Transformer (DFoT) — a video diffusion transformer with per-frame independent noise levels during training (same as DF §2.3 but at scale with transformers).

**Training Objective:** "noise-as-masking" — frames with $\sigma_i = \sigma_\text{max}$ are fully masked (pure noise), frames with $\sigma_i = 0$ are clean history. The model learns to condition on variable-length history.

**History Guidance at Inference (Three Variants):**

**(a) Vanilla History Guidance (HG-v):**
Use arbitrary-length history as CFG conditioning variable:
$$\tilde{\epsilon}_\theta(x_t | h) = \epsilon_\theta(x_t | \emptyset) + w \cdot [\epsilon_\theta(x_t | h) - \epsilon_\theta(x_t | \emptyset)]$$
where $h$ = history frames and $w$ = guidance weight. Even this simple variant significantly improves temporal consistency.

**(b) Temporal History Guidance (HG-t):**
Combine scores from different history windows $h_{t-k:t}$ for varying $k$:
$$\tilde{\epsilon} = \epsilon_\theta(x_t | \emptyset) + \sum_k w_k [\epsilon_\theta(x_t | h_{t-k:t}) - \epsilon_\theta(x_t | \emptyset)]$$

**(c) Fractional History Guidance (HG-f):**
Condition on history windows corrupted by varying noise levels. Combines temporal and frequency guidance for compositional generalization to out-of-distribution history lengths.

**Long Rollout:** Because the model is trained with variable-length history (including very short), it can roll out "extremely long videos" without distribution shift as history accumulates. The history guidance amplifies the conditioning signal without requiring architectural changes.

---

### 2.5 Rolling Forcing

**Paper:** "Rolling Forcing: Autoregressive Long Video Diffusion in Real Time" (Liu et al., arXiv 2509.25161, ICLR 2026)

**Core Innovation:** Jointly denoise a rolling window of $L_\text{win}=15$ latent frames with progressively increasing noise levels, rather than one frame at a time.

**Noise Schedule for Rolling Window:**
```
Uniform 5-step schedule: [t_5, t_4, t_3, t_2, t_1] = [1000, 800, 600, 400, 200]
Window length L_win = T = 5 (denoising steps)
Newest frame: noise level t_T (pure noise)
Oldest frame in window: noise level t_1 (nearly clean)
```
At each denoising step, the entire window is refined jointly — later frames benefit from forward-pass information about earlier frames ("mutual refinement").

**Window Configuration:**
- $L_\text{glo} = 3$ global context frames (cached from start)
- $L_\text{tem} = 3$ temporal context (recent clean frames)
- $L_\text{win} = 15$ active denoising window

**Attention Sink for Global Context:**
Initial $L_\text{glo}$ frames have their KV states cached *without* RoPE applied. At each future step, RoPE indices are adjusted so global frames appear positioned immediately before the temporal context. This anchors:
- Global exposure / color tone
- Semantic identity of scene elements

Without this, color drift and identity drift appear after ~100 frames.

**Training Strategy:** Mixed:
1. Frame-by-frame with Self Forcing objective (§2.6)
2. Rolling-window denoising with gradient computed only on non-overlapping windows (indices $i \equiv j \pmod{T}$), reducing forward passes from $N$ to $\lceil N/T \rceil$.

**Why Better than Pure Diffusion Forcing:** DF injects noise into past frames at inference, depriving the model of clean references. Rolling Forcing maintains clean references in the rolling window while still preventing strict causality (which would compound errors frame-by-frame).

**Performance:** 16 FPS on single GPU, multi-minute streams.

---

### 2.6 Self Forcing

**Paper:** "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion" (Huang et al., arXiv 2506.08009, NeurIPS 2025 Spotlight)

**Core Idea:** During training, perform autoregressive rollout using the model's own predictions (via rolling KV cache), then optimize a video-level loss on the entire self-generated sequence. This exactly matches inference conditions.

**Training Procedure:**
1. For each training sample, run autoregressive rollout: generate frame $\hat{x}_{t+1}$ conditioned on previous self-generated frames $\hat{x}_{1:t}$ using KV cache.
2. Compute **video-level** diffusion loss over all generated frames simultaneously (not frame-by-frame).
3. Use stochastic gradient truncation (stop gradient through rolled-out KV states older than $k$ steps) to control memory.
4. Optionally use a few-step diffusion model (4-step DDIM distillation) for faster training rollouts.

**Why It Works:** Teacher-forced models condition on $p_\text{data}(x)$ but must condition on $p_\text{model}(x)$ at test time — a distribution mismatch. Self Forcing trains on $p_\text{model}(x)$ exactly, eliminating exposure bias.

**Performance:** 480P video at ~0.8 s initial latency, then streaming ~16 FPS (H100) / ~10 FPS (RTX 4090).

---

### 2.7 Genie / Genie 2

**Genie Paper:** "Generative Interactive Environments" (Bruce et al., arXiv 2402.15391, 2024)

**Architecture (11B parameters):**
- **VQ-VAE Tokenizer (200M, ST-transformer, 12 enc + 20 dec layers):** 1024-code codebook, embedding dim 32, patch size 4. Processes 16 frames at 160×90 → discrete tokens $z$ per frame.
- **Latent Action Model (LAM, 300M):** Infers latent action between consecutive frame pairs. Codebook size 8 (|A|=8 discrete actions). Discarded at inference; user provides actions.
- **Dynamics Model (10.1B, 48 layers, MaskGIT):** Given $z_{1:t-1}$ and $a_{1:t-1}$, predicts $z_t$. Masked input (Bernoulli masking 0.5–1.0). Inference: 25 MaskGIT steps per frame, temperature 2.

**Training:** 125k steps, batch 512, AdamW lr 3e-5 → 3e-6, 256 TPUv5p. 30k hours of 2D platformer gameplay.

**History Length:** 16 frames (1.6 s at 10 FPS). Acknowledged limitation: "challenging to get consistent environments over long horizons" beyond 16 frames.

**Genie 2 (Dec 2024, blog post — no arxiv paper):**
- Autoregressive latent diffusion model on 3D environments.
- Transformer dynamics model trained with causal mask (LLM-style).
- Inference: frame-by-frame, single action + preceding latent frames → next latent frame.
- CFG for action controllability.
- Long-term memory: retains context over extended horizons (mechanism undisclosed, likely sliding window with long-context transformer).

---

### 2.8 LIVE

**Paper:** "LIVE: Long-horizon Interactive Video World Modeling" (Huang et al., arXiv 2602.03747, Feb 2026)

**Architecture:** 774M DiT, 18-step ODE sampling, 16× spatial VAE downsampling, 256×256 resolution.

**History Length:** 32 frames (fixed during training and evaluation).

**Cycle-Consistency Anti-Drift Objective:**

**(a) Forward Rollout:** Given $T$ training frames with $p$ prompt frames, generate remaining $T-p$ frames:
$$\tilde{x}_{p+1:T} \sim p_\theta(x_{p+1:T} | x_{1:p}, c_{1:T})$$

**(b) Reverse Rollout:** Temporally reverse the generated sequence and conditions, inject per-frame random noise:
$$\tilde{x}_{p+1:T,\text{rev}} \leftarrow (\tilde{x}_T, \ldots, \tilde{x}_{p+1}), \quad c_\text{rev} \leftarrow (c_T, \ldots, c_1)$$

**(c) Cycle Loss:** Apply diffusion loss to the recovered prompt frames:
$$\mathcal{L}_\text{LIVE} = \mathbb{E}\left[\frac{1}{T} \sum_{k=1}^T \|\epsilon_k - \epsilon_\theta^k\|^2\right]$$

**Why Bounded Error Accumulation:** Minimizing the cycle loss $\mathcal{L}_\text{LIVE}$ forces the forward distortion $\mathcal{D}(x_k, \tilde{x}_k)$ to stay within a "recoverable" range. Any frame sequence that cannot be reversed to the original prompt receives a large gradient signal. The model learns to keep errors bounded rather than accumulating monotonically.

**Curriculum:** Progressively decrease prompt ratio $p/T$ from 1.0 to lower values across training, gradually increasing the rollout length the cycle-consistency must cover.

**Comparison:** Outperforms noise augmentation (Diffusion Forcing) approaches on multi-minute rollouts for RealEstate10K, Minecraft, UE Engine.

---

### 2.9 Navigation World Models (NWM)

**Paper:** "Navigation World Models" (Bar et al., arXiv 2412.03572, CVPR 2025, Best Paper HM)

**Architecture:** Conditional Diffusion Transformer (CDiT), 1B parameters.

**Training Data:** Diverse egocentric videos of human and robotic agents.

**Conditioning:** Past visual observations + navigation actions → future observation frames.

**History Conditioning:** Sliding window of recent past frames; exact history length not publicly detailed.

**Drift Prevention:** Diffusion-based generation inherently provides high-quality per-step outputs; long-horizon planning is done via trajectory simulation and scoring rather than blind rollout (the model evaluates candidate trajectories rather than unconditioned generation).

**Application:** Planning — simulate future trajectories then rank/evaluate them against goal. Does not claim infinite-horizon stability; 10–30 step planning horizon.

---

### 2.10 iVideoGPT

**Paper:** "iVideoGPT: Interactive VideoGPTs are Scalable World Models" (Wu et al., arXiv 2405.15223, NeurIPS 2024)

**Architecture:** GPT-like autoregressive transformer (LLaMA-style, RoPE), 138M / 436M parameters.

**Tokenization:** Conditional VQGAN — context frames → $16 \times 16 = 256$ tokens per frame; future frames → $4 \times 4 = 16$ tokens (16× compression via temporal redundancy conditioning).

**History Conditioning:** 2 context frames for most tasks (BAIR, RoboNet).

**Action Conditioning:** Linear projection of actions added to slot token embeddings.

**Training:**
- Cross-entropy on next-token prediction, across 1.5M trajectories (Open X-Embodiment + Something-Something v2).
- Loss only on future frame tokens (not context frames).
- Reward prediction: linear head on last hidden state, MSE loss.

**Long-Horizon:** Token-by-token AR naturally extends to long rollouts; compressive tokenization (16× fewer future tokens) reduces error surface.

**Results:** FVD 60.8 (BAIR), PSNR 24.5; competitive with state-of-the-art action-conditioned methods.

---

### 2.11 Cosmos (NVIDIA)

**Paper:** "Cosmos World Foundation Model Platform for Physical AI" (arXiv 2501.03575, Jan 2025)

**Two Model Families:**

**(a) Diffusion WFM:**
- Tokenizer: Cosmos-1.0-Tokenizer-CV8x8x8 (continuous, 8× temporal, 8×8 spatial compression, 16-dim latent, Haar wavelet preprocessing).
- Training loss: EDM denoising score matching.
- Augmented noise on conditional frames for Video2World: $P_\text{mean} = -3.0, P_\text{std} = 2.0$ (log-normal noise schedule).
- Context window: up to 121 frames (~5 s at 24 FPS).
- 3D-factorized RoPE for arbitrary resolution/length generalization.

**(b) Autoregressive WFM (4B / 12B):**
- Tokenizer: Cosmos-1.0-Tokenizer-DV8x16x16 (discrete, 8× temporal, 16×16 spatial, integer codes).
- GPT-style next-token prediction.
- Diffusion decoder (Cosmos-1.0-Diffusion-7B-Decoder) for quality refinement: maps discrete → continuous tokens, reducing accumulated discretization errors.
- Progressive training stages: low-res (10,240 tokens / 57 frames at 512p) → high-res (56,320 tokens / 121 frames at 720p).

**Anti-Drift Strategy:** The diffusion decoder stage "restores" fine details lost in discrete tokenization, providing a quality reset every generation chunk. The AR model's exposure bias errors in discrete space are at least partially corrected by the diffusion decoder.

---

### 2.12 DreamerV3

**Paper:** "Mastering Diverse Domains through World Models" (Hafner et al., 2023, Nature 2025)

**Not a video model, but the gold standard for latent-space rollout stability.**

**RSSM Architecture:**
$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \quad \text{(deterministic GRU)}$$
$$z_t \sim q_\phi(z_t | h_t, o_t) \quad \text{(posterior, 32 categoricals × 32 classes)}$$
$$\hat{z}_t \sim p_\phi(\hat{z}_t | h_t) \quad \text{(prior)}$$

**Symlog Transform:** Observations encoded as $\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$. Prevents gradients from exploding through large-magnitude observations.

**Training Stability Tricks:**
- KL balance ($\alpha = 0.8$ weight on posterior, 0.2 on prior)
- Free bits (minimum KL of 1 nat per latent dimension)
- 1% uniform mixing (unimix) in categorical distributions — prevents logit collapse
- Percentile return normalization for critic
- Symexp two-hot encoding for continuous scalars
- Block GRU + RMSNorm + SiLU
- LaProp optimizer + adaptive gradient clipping

**Latent Rollout:** Up to 15-step imagination in latent space for actor-critic training. Gradients flow through RSSM dynamics. Beyond ~15 steps, returns are estimated with critic to avoid truncation bias.

---

## 3. 3D / Gaussian-Specific Dynamics

### 3.1 Spacetime Gaussians (STG)

**Paper:** "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis" (Li et al., arXiv 2312.16812)

**Time-Dependent Opacity (Temporal RBF):**
$$\sigma_i(t) = \sigma_i^s \cdot \exp\left(-s_i^t |t - \mu_i^t|^2\right)$$
- $\sigma_i^s$: spatial (view-independent) opacity
- $\mu_i^t$: temporal center (peak visibility timestamp)
- $s_i^t$: temporal spread (sharpness of temporal window)

**Polynomial Motion:**
$$\mu_i(t) = \sum_{k=0}^{n_p} b_{i,k} (t - \mu_i^t)^k, \quad n_p = 3 \text{ (cubic)}$$
$$q_i(t) = \text{polynomial quaternion, } n_q = 1 \text{ (linear)}$$

**Training:** L1 + D-SSIM on rendered frames. Densification: guided ray sampling for sparse regions + aggressive pruning of low-opacity Gaussians.

**Appearing/Disappearing:** STG naturally handles transient content — Gaussians with narrow $s_i^t$ are only visible in a short temporal window, effectively appearing and disappearing without any special mechanism.

---

### 3.2 EvoGS

**Paper:** "EvoGS: 4D Gaussian Splatting as a Learned Dynamical System" (arXiv 2512.19648)

**Velocity-Field Formulation:**
$$\frac{d\mathbf{x}_i}{dt} = \mathbf{v}_\theta(\mathbf{x}_i(t), \mathbf{f}_i(t), t)$$
where $\mathbf{x}_i = [\mathbf{p}_i, \mathbf{R}_i, \mathbf{S}_i, \mathbf{c}_i, \alpha_i]$ (position, rotation, scale, color, opacity) and $\mathbf{f}_i(t)$ are spatial-temporal features from HexPlane factorization.

**Integration:** RK4:
$$\mathbf{x}_i(t_1) = \text{RK4}(\mathbf{x}_i(t_0), t_0, \Delta t, \mathbf{v}_\theta)$$
Bidirectional: backward rollout via $\Delta t < 0$.

**Training Loss:**
$$\mathcal{L} = \mathcal{L}_\text{photo} + \lambda_\text{coh} \mathcal{L}_\text{coh} + \lambda_\text{anchor} \mathcal{L}_\text{anchor} + \lambda_\text{tv} \mathcal{L}_\text{tv}$$
- $\mathcal{L}_\text{coh}$: velocity coherence — nearby Gaussians must move similarly
- $\mathcal{L}_\text{anchor}$: $\sum \|\mathbf{x}(t^{(a)}) - \hat{\mathbf{x}}(t^{(a)})\|_2^2$ for sparse waypoints
- $\mathcal{L}_\text{tv}$: total variation on spatiotemporal feature planes

**Temporal Anchor Points:** Three sparse anchors (start, midpoint, end) act as re-initialization states preventing long-term drift during extrapolation. Without anchors, ODE integration error compounds.

---

### 3.3 ODE-GS

**Paper:** "ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting" (arXiv 2506.05480)

**Architecture:**
- **Encoder (Transformer):** 128 dim, 8 heads, 5 layers. Encodes observed Gaussian trajectory sequence $\gamma_k = \{G_k(t_j)\}_{j=1}^{N_c}$ → latent $z(t_0) \in \mathbb{R}^{64}$.
- **Neural ODE:** $\dot{z} = f_\theta(z(t))$, 4-layer MLP, 64 hidden units. Solved with DOPRI5 (rtol=1e-3, atol=1e-4).
- **Decoder:** 5-layer transformer, 128 dim. $\hat{G}_k(t) = \delta_\psi(z(t))$.

**Training (Two-Stage):**
1. Train interpolation model (deformation MLP) for 40k iterations.
2. Freeze; train ODE model for 40 epochs:
$$\mathcal{L} = \mathcal{L}_e + s_t(\lambda_\text{latent} \cdot R_\text{latent} + \lambda_\text{traj} \cdot R_\text{traj})$$
   - $\mathcal{L}_e$: L1 between predicted and target Gaussian parameters
   - $R_\text{latent}$: finite-difference penalty on latent velocity (prevents oscillation)
   - $R_\text{traj}$: penalizes 3D position acceleration
   - $\lambda_\text{traj} = 10^{-1}$, $\lambda_\text{latent} = 10^{-5}$
   - $s_t$: adaptive weight that *increases* as $\mathcal{L}_e$ decreases (stabilizes late training)

**Context for Extrapolation:** $N_c = 30$ observed frames; variable extrapolation length $N_e = 10$.

**Results:** 19.8% improvement in PSNR over leading baselines on D-NeRF, NVFi, HyperNeRF.

---

### 3.4 GaussianPrediction

**Paper:** "GaussianPrediction: Dynamic 3D Gaussian Prediction for Motion Extrapolation and Free View Synthesis" (arXiv 2405.19745)

**Key Innovation: Hyper-Canonical Space**
$$\mathcal{C}_h = \{(\mu, m) | \mu \in \mathbb{R}^3, m \in \mathbb{R}^d\}$$
Combined spatial coordinates + motion feature embedding. K-means clusters in this joint space → key points $K = \{k_i\}$.

**Pipeline:**
1. **Deformation MLP** $D$: produces $(\Delta\mu^t, \Delta q^t)$ offsets from canonical space. Novel `lifecycle` property $\psi(G_i, t)$ handles irreversible deformations.
2. **Motion Distillation:** Key point motions $(T_k^t, Q_k^t)$ computed from Gaussian motions via hash-encoded time-independent weights.
3. **GCN Prediction:** Single-layer MLP decodes relational GCN features → next key point positions:
$$\Delta\mu_i^t = \sum w^T_{i\leftarrow k} \cdot T_k^t, \quad \Delta q_i^t = \sum w^Q_{i\leftarrow k} \cdot Q_k^t$$

**Training:** 3-step: 30k (static init) + 10k (key point weights) + 20-30k (GCN training) iterations.

**Prediction Horizon:** Evaluated on held-out frames $t > 0.8 \cdot T_\text{max}$.

---

### 3.5 GWM (Gaussian World Model for Robotics)

**Paper:** "GWM: Towards Scalable Gaussian World Models for Robotic Manipulation" (Lu et al., arXiv 2508.17600, ICCV 2025)

**Architecture:**
- **3D Gaussian VAE:** FPS downsample to $N=2048$ Gaussians; cross-attention encoder to latent $\mathbf{x} \in \mathbb{R}^{N \times D}$; transformer decoder for reconstruction.
- **Dynamics DiT:** EDM preconditioning; actions as cross-attention keys/values.
- **VAE Loss:** $\mathcal{L}_\text{VAE} = \text{Chamfer}(\hat{G}, G) + \|C(\hat{G}) - C(G)\|_1$
- **DiT Loss:** EDM formulation with adaptive preconditioning.

**Gaussian Count Management:** FPS to fixed $M=512$ latent points. VAE always outputs same dimension regardless of input Gaussian count variation.

**Action Conditioning:** Actions embedded → cross-attention into DiT layers.

**Multi-Step Rollout:** Iterative application (single-step model applied repeatedly). No explicit long-horizon stabilization reported.

---

### 3.6 GaussianGPT

**Paper:** "GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation" (arXiv 2603.26661, 2026)

**Tokenization:**
- Sparse 3D convolutional autoencoder with LFQ (Lookup-Free Quantization): codebook size 4096, voxel size 0.025m → 20cm latent voxels.
- Training: RGB + perceptual re-rendering loss + occupancy + LFQ entropy regularization.

**Transformer:** GPT-2 medium, 16,384-token context. 3D RoPE on actual voxel coordinates (not sequence index). Position tokens alternate with feature tokens.

**Generation:** BOS → alternating (position, feature) tokens → EOS. Scene completion = prefix prompt.

**Relevance for Dynamic Scenes:** GaussianGPT is static scene generation but the serialization + causal transformer paradigm is directly applicable to dynamic Gaussian sequences.

---

## 4. Temporal Consistency & Anti-Drift Regularizers for 3D Gaussians

### 4.1 As-Rigid-As-Possible (ARAP) Loss

Used in DynaSurfGS, many deformable-GS methods:
$$\mathcal{L}_\text{ARAP} = \sum_i \sum_{j \in \mathcal{N}(i)} w_{ij} \|(\mathbf{p}_j^{t+1} - \mathbf{p}_i^{t+1}) - \mathbf{R}_i^t (\mathbf{p}_j^t - \mathbf{p}_i^t)\|^2$$
where $\mathbf{R}_i^t$ is the estimated rotation for Gaussian $i$'s local neighborhood, $\mathcal{N}(i)$ is the $K$-nearest-neighbor set, and $w_{ij}$ are distance-based weights.

**Interpretation:** Punishes non-rigid stretching of the local neighborhood. Neighboring Gaussians should maintain their relative distances modulo a rigid rotation.

### 4.2 Velocity Coherence Loss (EvoGS)

$$\mathcal{L}_\text{coh} = \sum_i \sum_{j \in \mathcal{N}(i)} \|\mathbf{v}_\theta(\mathbf{x}_i) - \mathbf{v}_\theta(\mathbf{x}_j)\|^2$$
Encourages spatial smooth velocity fields. Nearby Gaussians should have similar velocities.

### 4.3 Acceleration / Jerk Smoothness

Second- and third-order finite-difference penalties on Gaussian positions:
$$\mathcal{L}_\text{acc} = \sum_i \|\mathbf{p}_i^{t+1} - 2\mathbf{p}_i^t + \mathbf{p}_i^{t-1}\|^2$$
$$\mathcal{L}_\text{jerk} = \sum_i \|\mathbf{p}_i^{t+2} - 3\mathbf{p}_i^{t+1} + 3\mathbf{p}_i^t - \mathbf{p}_i^{t-1}\|^2$$
Used in velocity-centric 4DGS and physics-informed methods to suppress jitter and overshoot.

### 4.4 Group Motion Consistency (SpeeDe3DGS, GroupFlow)

Group spatially adjacent Gaussians, constrain them to share SE(3) transforms:
$$\mathcal{L}_\text{group} = \sum_g \sum_{i,j \in g} \|(\mathbf{R}_g, \mathbf{t}_g)(\mathbf{p}_i^0) - \mathbf{p}_i^t\|^2$$
Reduces degrees of freedom → fewer parameters to overfit → better generalization across time.

### 4.5 Temporal Pruning (SpeeDe3DGS)

Remove Gaussians with low temporal sensitivity (low gradient variance across time). This prevents the model from wasting capacity on static regions and ensures the remaining Gaussians are all meaningfully dynamic.

---

## 5. Appearing / Disappearing Content in Gaussian Rollout

### 5.1 Temporal Opacity RBF (STG approach, recommended)

Each Gaussian has a temporal existence window:
$$\alpha_i(t) = \alpha_i^s \cdot \exp\left(-s_i^t (t - \mu_i^t)^2\right)$$
Gaussians with $\alpha_i(t) < \alpha_\text{thresh}$ are culled at rendering time.

**For Autoregressive Prediction:** At step $t$, the model predicts:
- $\Delta \mu_i^t$ (temporal center offset)
- $\Delta s_i^t$ (temporal sharpness update)
- Whether to introduce a new Gaussian (birth)
- Whether opacity has decayed below threshold (death = implicit)

### 5.2 Opacity-Gated Existence (4DGS, recommended)

The model predicts opacity $\alpha_i^t$ directly. At each rollout step:
1. Apply predicted $\Delta\alpha_i^t$ to each Gaussian's opacity.
2. Gaussians with $\alpha_i^t < 0.005$ (threshold) are pruned from the active set.
3. A **birth module** proposes new Gaussians:
   - Location: where rendering error is high (detected via auxiliary depth/RGB prediction heads)
   - Initialization: mean position is sampled from the predicted depth map; covariance initialized to isotropic $\sigma_\text{init}^2 I$; opacity initialized to $\alpha_\text{init} = 0.5$

### 5.3 Learned Lifecycle (GaussianPrediction approach)

The deformation MLP includes a `lifecycle` property $\psi(G_i, t)$ that models irreversible deformations. For prediction rollout:
- The GCN predicts future lifecycle values alongside positions.
- Gaussians with $\psi_i^t = 0$ are considered "dead" (removed from active set).

### 5.4 Fixed-Size Population with Soft Masking (practical approach)

Maintain a **maximum fixed population** $N_\text{max}$ Gaussians, using a learned opacity $\alpha_i^t \in [0, 1]$ as a soft existence mask. This avoids variable-size set operations in the transformer:
- Neural network predicts $\Delta\alpha_i$ as part of the Gaussian state update.
- Gaussians with small $\alpha_i$ contribute negligibly to rendering (effectively dead) but remain in the state tensor.
- Periodically (every $K$ steps), dead Gaussians ($\alpha < \alpha_\text{thresh}$) are reallocated to seed new Gaussians from predicted high-error regions.

---

## 6. Critical Synthesis: Recommended Rollout Recipe for Instruct-GS-World

### 6.1 Problem Specification

**Task:** Given a Gaussian scene state $\mathcal{G}_t = \{(\mu_i, q_i, s_i, c_i, \alpha_i)\}_{i=1}^N$ (position, rotation, scale, spherical harmonics, opacity) and a language instruction $L$, predict $\mathcal{G}_{t+1}, \ldots, \mathcal{G}_{t+H}$ for $H = 100+$ steps at 10 Hz.

**Key challenges:**
1. Gaussian count $N$ varies (appears/disappears)
2. Language conditioning must persist for 10+ seconds
3. Error accumulates in position, rotation, opacity over 100+ steps
4. Training data rarely exceeds 5–10 s; must generalize to longer

---

### 6.2 Recommended Architecture

```
Input at step t:
  - G_t: (N, D_gauss) per-Gaussian state [mu, q, s, c_SH, alpha]
  - G_{t-K:t-1}: K past states (history)
  - L: language embedding (CLIP / T5-XXL)

Model:
  1. Gaussian Encoder: PointNet++ or sparse 3D conv → (N, D_latent)
  2. Temporal Aggregator: 8-layer causal Transformer over history K
  3. Language Cross-Attention: T5 cross-attention in every transformer layer
  4. Per-Gaussian Decoder: MLP → (delta_mu, delta_q, delta_s, delta_c, delta_alpha, p_birth)
  5. Birth/Death Head: lightweight MLP on pooled features → N_new Gaussians to spawn

Key design choices:
  - Fixed population N_max = 10,000-50,000 with soft opacity masking
  - Positional encoding: 3D Fourier features on Gaussian centers (not sequence index)
  - Language instruction re-injected every step via cross-attention (not just at t=0)
```

---

### 6.3 Training Recipe (Step-by-Step)

#### Phase 1: Short-Horizon Supervised Pre-Training (Steps 1–100k)

**Input noise augmentation (critical, from GameNGen):**
```python
sigma_context = np.random.uniform(0, 0.3)  # noise on history states
G_hist_noisy = G_hist + sigma_context * torch.randn_like(G_hist)
# Encode sigma_context as input to model (10-bin embedding)
```

**Teacher forcing:** At each step, feed ground-truth $\mathcal{G}_{t-1}$ as context. Rollout 4-step sequences.

**Loss:**
$$\mathcal{L}_\text{pred} = \sum_{i} \|\Delta\mu_i^t - \widehat{\Delta\mu_i^t}\|^2 + \lambda_q \|q_i^t - \hat{q}_i^t\|^2 + \lambda_\alpha |\alpha_i^t - \hat{\alpha}_i^t|$$

**Anti-drift regularizers:**
$$\mathcal{L}_\text{ARAP} = \sum_{(i,j) \in \text{kNN}} w_{ij} \|(\Delta\mu_j - \Delta\mu_i) - R_i (\mu_j^0 - \mu_i^0)\|^2$$
$$\mathcal{L}_\text{vel} = \sum_i \sum_{j \in \mathcal{N}(i)} \|\mathbf{v}_i - \mathbf{v}_j\|^2$$
$$\mathcal{L}_\text{acc} = \sum_i \|\Delta\mu_i^{t+1} - 2\Delta\mu_i^t + \Delta\mu_i^{t-1}\|^2$$

Total: $\mathcal{L} = \mathcal{L}_\text{pred} + 0.1 \cdot \mathcal{L}_\text{ARAP} + 0.01 \cdot \mathcal{L}_\text{vel} + 0.001 \cdot \mathcal{L}_\text{acc}$

#### Phase 2: Scheduled Sampling / Self-Forcing (Steps 100k–300k)

Gradually transition from teacher-forcing to free-running:
```python
# Linear scheduled sampling schedule
p_teacher = max(0.1, 1.0 - step / 300000)

if random() < p_teacher:
    G_context = G_ground_truth
else:
    G_context = G_predicted  # model's own previous output
```

Alternatively (preferred for diffusion models): **Self Forcing** (§2.6) — roll out K=8 steps with model's own predictions, backprop with truncated BPTT through last 4 steps, optimize video-level rendering loss.

#### Phase 3: Diffusion Forcing Long-Horizon Fine-Tuning (Steps 300k–500k)

If using a diffusion-based Gaussian state model:
1. Assign independent noise levels $\sigma_i^t \sim p(\sigma)$ to each Gaussian at each timestep.
2. Train model to denoise sequence under causal attention.
3. At inference, condition on previously generated Gaussians at $\sigma=0$, generate next Gaussians from $\sigma=\sigma_\text{max}$.

For a regression (non-diffusion) model: apply the per-step **noise augmentation** from GameNGen:
```python
sigma_aug = 0.3 * (1 - step / 500000)  # decay augmentation over training
G_context = G_model_output + sigma_aug * randn_like(G_model_output)
```

#### Phase 4: Anti-Drift Cycle-Consistency Fine-Tuning (optional, Steps 500k–600k)

Implement LIVE-style cycle consistency (§2.8):
1. Forward rollout: $\mathcal{G}_{1:T}^\text{model}$ from ground-truth $\mathcal{G}_{1:p}$.
2. Reverse rollout: $\hat{\mathcal{G}}_{1:p}$ from reversed $\mathcal{G}_{T:p+1}^\text{model}$.
3. Cycle loss: $\mathcal{L}_\text{cycle} = \|\hat{\mathcal{G}}_{1:p} - \mathcal{G}_{1:p}\|^2$.

---

### 6.4 History Length Recommendation

Based on analysis of GameNGen (64 frames / 3.2 s), Genie (16 frames), LIVE (32 frames):

- **Minimum history:** 16 frames (1.6 s at 10 Hz)
- **Recommended:** 32 frames (3.2 s at 10 Hz) — covers one complete action-response cycle
- **Maximum practical:** 64 frames (6.4 s at 10 Hz) — diminishing returns vs. memory cost

Use **compressed history encoding** (iVideoGPT-style): maintain 32 full-resolution recent frames, compress older frames by 4×–8× in a learned summary state (like DreamerV3's GRU hidden state).

---

### 6.5 Inference Rollout Procedure

```
Algorithm: Long-Horizon Gaussian Rollout
Input: G_0 (initial scene), L (language), H (horizon=100+)
Output: G_1, ..., G_H

1. Initialize:
   - Active set A = {i : alpha_i^0 > alpha_thresh}
   - History buffer: H_buf = [G_0]
   - Global anchor: G_anchor = G_0 (cached, never updated)
   - Language embedding: l = encode(L)

2. For t = 1 to H:
   a. Context = H_buf[-K:]  # last K states
   b. Noisy context = Context + sigma_aug * randn  # tiny noise for robustness
      (sigma_aug = 0.05 at inference, matched to training schedule)
   c. Predict delta state: d_G = model(Noisy_context, l, G_anchor)
   d. Update active Gaussians: G_pred = apply_delta(G_{t-1}, d_G)
   
   e. Opacity update and death:
      alpha_i^t = clip(alpha_i^{t-1} + d_alpha_i, 0, 1)
      PRUNE: remove i where alpha_i^t < 0.005
   
   f. Birth:
      IF t % K_birth == 0:
         new_Gs = birth_head(G_pred, rendering_error_map)
         A = A ∪ new_Gs (up to N_max)
   
   g. Apply anti-drift momentum:
      v_i^t = beta_v * v_i^{t-1} + (1-beta_v) * (mu_i^t - mu_i^{t-1})
      # Smooth velocity via EMA
      mu_i^t = mu_i^{t-1} + v_i^t  # reapply smoothed velocity if drift detected
   
   h. Append G_pred to H_buf; truncate to K frames
   i. Emit G_pred for rendering at step t
```

**Key inference parameters:**
- `sigma_aug = 0.05` (small but nonzero — matches training noise)
- `beta_v = 0.9` (velocity EMA for momentum smoothing)
- `K_birth = 10` (birth check every 10 steps = 1 Hz)
- `alpha_thresh = 0.005`
- `N_max = 20,000`

---

### 6.6 Gaussian Count Management Over 100+ Steps

**Protocol:**
1. **Fixed maximum size $N_\text{max}$** with soft opacity masking. All tensors are $(N_\text{max}, D)$ — dead Gaussians have $\alpha = 0$, contribute nothing to rendering.
2. **Periodic compaction:** Every 50 steps, reclaim dead Gaussian slots. Replace slots with new Gaussians seeded at predicted high-reconstruction-error regions (detected via auxiliary rendering loss head).
3. **Gradual opacity decay schedule:** During training, teach the model to gracefully decay opacity of disappearing objects via time-conditioned opacity prediction.
4. **Avoid hard birth/death operations at every step** — they break gradient flow. Use soft masking during training; hard pruning only at inference.

**Birth seeding at inference:**
```python
# Every K_birth steps
render_error = compute_photometric_error(G_pred, reference_image_if_available)
high_error_regions = render_error > error_threshold  # spatial mask
new_mu = sample_from_depth_map(predicted_depth, mask=high_error_regions, N=N_new)
new_alpha = 0.1  # start small, let model grow
new_s = s_init_isotropic
# Replace N_new dead Gaussian slots
```

---

### 6.7 Anti-Drift Regularizers Summary for Rollout Training

| Regularizer | Formula | Lambda | Applied at |
|---|---|---|---|
| ARAP rigidity | $\sum_{ij} w_{ij} \| (\Delta\mu_j - \Delta\mu_i) - R_i(\mu_j^0 - \mu_i^0)\|^2$ | 0.1 | Every step |
| Velocity coherence | $\sum_{ij \in kNN} \|v_i - v_j\|^2$ | 0.01 | Every step |
| Acceleration smoothness | $\|\mu^{t+1} - 2\mu^t + \mu^{t-1}\|^2$ | 0.001 | Steps 2+ |
| Opacity TV | $|\alpha^{t+1} - \alpha^t|$ | 0.01 | Every step |
| Cycle consistency | $\|\hat{G}_{1:p} - G_{1:p}\|^2$ | 1.0 | Phase 4 only |

---

### 6.8 Diffusion Forcing vs. Regression + Scheduled Sampling: Which to Use?

**Recommendation for Instruct-GS-World:** Use **regression (deterministic or stochastic MLP/transformer) + noise augmentation + scheduled sampling** for the following reasons:

1. **Gaussian states are low-dimensional per-entity** (~15 floats per Gaussian), not pixel images. Diffusion forcing was designed for high-dimensional token sequences where uncertainty modeling is crucial.

2. **Regression is faster:** single forward pass per step vs. $K$ denoising steps in diffusion. At 10 Hz over 100 steps, inference speed is critical.

3. **Noise augmentation (GameNGen-style) is simpler and effective:** Add $\sigma \sim U(0, 0.3)$ noise to context Gaussian states during training; encode $\sigma$ as input. Decays to $\sigma \approx 0.05$ at final inference.

4. **If uncertainty matters** (e.g., ambiguous instructions leading to multiple valid futures): Use a **latent diffusion model** over per-Gaussian state updates (like GWM §3.5) — model $p(\Delta G_{t+1} | G_{t-K:t}, L)$ with DiT, run 4–8 DDIM steps per step.

5. **Diffusion Forcing** is most beneficial when training and inference lengths differ greatly. If you can train on 32-frame windows and roll out to 100+, use DF with the RNN hidden-state carryover trick (§2.3).

---

### 6.9 Language Conditioning for Persistent Instructions

**Strategy (from NWM and Cosmos):**
- Encode instruction with T5-XXL or CLIP → $l \in \mathbb{R}^{D_L}$.
- Inject into every transformer block via cross-attention (not just once at the start).
- Use instruction dropout ($p = 0.1$) for CFG at inference.
- At inference, apply CFG: $\Delta G = \Delta G_\text{uncond} + w (G_\text{cond} - G_\text{uncond})$, $w = 2$–7.

**For 10+ second rollouts:** Language embeddings should be re-injected at every step (already naturally handled if cross-attention is in every transformer block). Instructions describe semantics that should accumulate, not just a one-time impulse.

---

### 6.10 Global Anchor for Scene Identity

**Implement attention sink (from Rolling Forcing §2.5):**
```
At t=0: cache KV states of initial G_0 (compressed to 128 points via FPS)
At every step t: prepend anchor KV states to attention (without RoPE offset)
RoPE indices: anchor treated as position "immediately before" current context
```

This prevents the model from drifting away from the initial scene identity (color, scale, room layout) over long rollouts — the most common failure mode in video world models.

---

## Key References

1. **GameNGen** (arXiv 2408.14837): noise augmentation σ_max=0.7, 64 context frames, 4-step DDIM
2. **DIAMOND** (arXiv 2405.12399): EDM preconditioning, clean context, 3 denoising steps
3. **Diffusion Forcing** (arXiv 2407.01392): independent per-token noise, causal RNN, infinite horizon via hidden-state carryover
4. **History-Guided Video Diffusion / DFoT** (arXiv 2502.06764): flexible history CFG, time-frequency guidance
5. **Rolling Forcing** (arXiv 2509.25161): joint denoising window, attention sink, L_glo=3 global anchor frames
6. **Self Forcing** (arXiv 2506.08009): KV-cache rollout during training, video-level loss, truncated BPTT
7. **Genie** (arXiv 2402.15391): 16-frame VQ-VAE + MaskGIT dynamics
8. **Genie 2** (DeepMind blog, Dec 2024): autoregressive latent diffusion, causal transformer
9. **LIVE** (arXiv 2602.03747): cycle-consistency objective, 32-frame history, 774M DiT
10. **Navigation World Models** (arXiv 2412.03572): CDiT, 1B, trajectory simulation for planning
11. **iVideoGPT** (arXiv 2405.15223): 16× compression tokenization, 2-frame context
12. **Cosmos** (arXiv 2501.03575): discrete AR + diffusion decoder; continuous diffusion WFM; P_mean=-3.0 noise augmentation
13. **DreamerV3** (2023/Nature 2025): RSSM + symlog + KL balance for stable latent rollouts
14. **Spacetime Gaussians** (arXiv 2312.16812): temporal RBF opacity, cubic polynomial motion
15. **EvoGS** (arXiv 2512.19648): RK4 velocity ODE, coherence + anchor losses
16. **ODE-GS** (arXiv 2506.05480): latent ODE, λ_traj=0.1, λ_latent=1e-5, 30-frame context
17. **GaussianPrediction** (arXiv 2405.19745): hyper-canonical space, GCN key-point prediction
18. **GWM** (arXiv 2508.17600): 3D VAE + EDM DiT for Gaussian future prediction
19. **GaussianGPT** (arXiv 2603.26661): VQ autoencoder + causal transformer with 3D RoPE
20. **FramePack** (arXiv 2504.12626): compressed context packing, inverted anti-drift sampling
21. **DynaSurfGS** (arXiv 2408.13972): ARAP regularization for dynamic Gaussians
