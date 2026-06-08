# Research E1 — MetaQuery-style task-conditioned tokens from a FROZEN MLLM, and LMM→segmentation/region-token methods

**Purpose.** Foundation + faithful recipe for our system: freeze **Qwen3-VL-2B** (Cosmos-Reason2-2B, hidden 2048) and extract (a) **semantic/grounding** tokens and (b) **motion-intent** tokens to condition the downstream 3DGS dynamics network (`igsw/dynamics`). This is the literature half of Module E; it pairs with `research_E_DESIGN_draft.md` (which already defines `region_tokens [R,d]` + `motion_tokens [Q,d]` and per-Gaussian anchor-uv sampling).

**Verification protocol.** Every paper below was checked against the arXiv API (`export.arxiv.org/api/query`) and/or primary HTML/project pages. Each entry carries a **VERIFIED / UNVERIFIED** tag. Where the user's brief stated a detail that the source contradicts, it is flagged **[CORRECTION]**. Where a source is silent on a load-bearing detail, it is flagged **[SOURCE SILENT]** — not invented.

Date of verification: 2026-06-05.

---

## 0. TL;DR of what was verified vs. what the brief got wrong

| Item | Brief claim | Verified reality | Tag |
|---|---|---|---|
| AFUN exists | yes, ID `2606.02551` **or** `2504.06256` | **YES**, AFUN = `arXiv:2606.02551` (1 Jun 2026). The *other* ID `2504.06256` is **MetaQueries** (a different real paper). Neither ID was wrong — they were two different real papers. | VERIFIED |
| AFUN backbone | Qwen3-VL (size unspecified, user guessed 2B) | **Qwen3-VL-8B** is the main backbone; **2B is an ablation**. | VERIFIED **[CORRECTION]** |
| AFUN MetaQuery count | (unstated) | **64 total = 32 semantic + 32 motion**. | VERIFIED |
| AFUN trainable | ~32.21M, only queries+MLP+motion decoder | **32.21M**, frozen = {Qwen3-VL, SAM3, Sonata}; trained = {query tokens, 2-layer proj MLP, 6-layer motion decoder}. | VERIFIED |
| MetaQuery mechanism | append learnable queries to frozen MLLM, read final hidden states, train queries+connector | **CONFIRMED** verbatim by primary source. N=256, 24-layer connector, MLLM frozen. | VERIFIED |
| LISA / GLaMM / PixelLM / VideoGLaMM / u-LLaVA / LISA++ | all real, `<SEG>`→decoder | **ALL VERIFIED**, IDs + mechanisms below. | VERIFIED |
| Groma / Ferret | region-token mechanisms | **BOTH VERIFIED**. | VERIFIED |

**One-line strategic read:** The exact pattern the brief wants — *append learnable query tokens to a frozen Qwen3-VL, read their last-layer hidden states, project to a downstream module, train only the queries + projector + downstream head* — is **literally the AFUN recipe**, and AFUN is itself **MetaQuery applied to affordance** (semantic queries → SAM3 masks, motion queries → 3D motion decoder). Our task is structurally identical (semantic→grounding, motion→3D dynamics), so AFUN is the single closest precedent and we should mirror it.

---

## 1. AFUN — **VERIFIED** (`arXiv:2606.02551`)

- **Title:** *AFUN: Towards an Affordance Foundation Model for Functionality Understanding*
- **Authors:** Zhaoning Wang, Yi Zhong, Jiawei Fu, Henrik I. Christensen, Jun Gao
- **Submitted:** 1 June 2026. Categories cs.RO, cs.CV.
- **arXiv API confirms the entry exists** (queried `id_list=2606.02551`; title/authors/date returned). HTML mirror: https://arxiv.org/html/2606.02551 ; abs: https://arxiv.org/abs/2606.02551
- **The other ID the user supplied, `2504.06256`, is NOT AFUN — it is the MetaQueries paper (§2).** Both are real; they are different works. AFUN cites/uses the MetaQuery mechanism.

### 1.1 Task
From a **single RGB-D observation + language task description**, predict:
1. a **task-conditional functional mask** ("where to interact"), and
2. a **3D post-contact motion curve** ("how to interact").

### 1.2 Frozen backbones (all three frozen throughout training)
- **VLM:** **Qwen3-VL-8B** (main). Ablations: Qwen3-VL-2B, "Qwen3.5-9B" variant. **[CORRECTION]** — the brief's "2B" is the *ablation*, not the headline config. Good news for us: **2B is explicitly ablated**, so a 2B build is precedented.
- **Segmentation model:** **SAM3** (mask decoder reused).
- **3D feature encoder:** **Sonata** (point-cloud network).

### 1.3 MetaQuery tokens (the core mechanism)
- Two learnable sets, **appended to the VLM input prompt together** and processed through the Qwen3-VL transformer:
  - semantic set `mq^s = {⟨mq^s_0⟩ … ⟨mq^s_{Ns-1}⟩}`, **Ns = 32**
  - motion set `mq^m = {⟨mq^m_0⟩ … ⟨mq^m_{Nm-1}⟩}`, **Nm = 32**
  - **64 MetaQuery tokens total.**
- **Read-out:** "The **last hidden states** of each set of tokens are then fed into the downstream segmentation and motion model, respectively." (i.e. take the final-layer hidden states at the 64 query positions.)
- **[SOURCE SILENT]** The paper does **not** specify the attention mask for the query tokens (causal vs bidirectional) nor any explicit positional-encoding / mrope handling for them. It only says "appended to the input prompt and processed through the transformer." → For our impl we adopt the **explicit** MetaQueries choice (causal mask over the whole sequence; §2/§5), since AFUN inherits that lineage.

### 1.4 Semantic branch → SAM3
- The 32 semantic query hidden states are **mapped by a two-layer MLP into SAM3's language-feature space**, then fed to **SAM3's mask decoder**, which predicts per-detection **boxes + masks** (the functional mask).

### 1.5 Motion branch → 3D post-contact motion
- **3D input path:** point cloud (unprojected from depth) → **frozen Sonata** → features **projected back to image space and pooled to "geo features."** (Depth does **not** enter Qwen3-VL; only the RGB image does.)
- **Motion decoder:** **6 transformer layers** with **self-attention to the encoded geo features** and **cross-attention to (the per-object features from SAM3) + (the 32 motion MetaQuery tokens).**
- **Output representation:** an **anchored 3D Bézier spline curve** with control points `{P_k}_{k=1}^K` predicted in **relative 3D coordinates**.

### 1.6 What is trained — **32.21M total**
Trainable = {MetaQuery tokens (64×2048), the 2-layer projection MLP to SAM3, the 6-layer motion decoder}. Frozen = {Qwen3-VL, SAM3, Sonata}.

### 1.7 Three-stage training + losses
- **Stage I — MetaQuery↔SAM3 alignment:** MSE between projected semantic-query features and SAM3 text features (on Visual Genome).
- **Stage II — affordance segmentation:** SAM3 losses only — box regression (ℓ1 + GIoU), presence classification, per-query mask (focal BCE + Dice), plus a semantic-segmentation term.
- **Stage III — joint:** `L = λ_sam3·L_sam3 + λ_curve·L_curve`; the motion term uses a **point-sampling loss (Curve-GCN style) with T = 16 sampled points per curve**.

### 1.8 Reported gains (for context, not load-bearing)
+23.9 / +26.3 segmentation metric gains; 12.7–61.3% contact-point hit-rate gain over baselines.

### 1.9 Why AFUN ≈ our problem (direct mapping)
| AFUN | Instruct-GS-World (us) |
|---|---|
| 32 semantic queries → MLP → SAM3 → functional mask | **SEM_QUERY** → grounding (our `role_masks_2d` / `region_tokens`) — could go to SAM/SAM2 **or** straight into dynamics cross-attn |
| 32 motion queries → 6-layer decoder (+geo, +SAM feats) → 3D Bézier curve | **MOTION_QUERY** → our `igsw/dynamics` (SC-GS control set), which already outputs continuous 3D motion — so we do **not** need a separate Bézier head; the dynamics net *is* the motion decoder |
| frozen Qwen3-VL-8B (2B ablated) | frozen Qwen3-VL-2B (Cosmos-Reason2-2B) |
| Sonata for 3D | **Pi3** already gives us 3D Gaussians + per-Gaussian uv anchors (Stream A) |

---

## 2. MetaQueries — **VERIFIED** (`arXiv:2504.06256`)

- **Title:** *Transfer between Modalities with MetaQueries*
- **Authors:** Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, Saining Xie.
- **Submitted:** 8 April 2025 (arXiv API confirmed). Project page: https://xichenpan.com/metaquery/ ; HTML: https://arxiv.org/html/2504.06256v1 ; OpenReview PDF: https://openreview.net/pdf/0ed3644e169be9c9603fd6e341777f869a88d2a7.pdf
- **This is the ID the user attached to "AFUN" — it is actually MetaQueries.** (Real, just mislabeled.)

### 2.1 Exact mechanism (this is our template)
1. **Learnable queries** `Q ∈ R^{N×D}`, randomly initialized, **D = MLLM hidden dim**. **Main setting N = 256**; ablated over N ∈ {1,4,16,32,64,128,256,512,1024}.
2. **Concatenation:** queries are **appended to the MLLM input token sequence**; the whole sequence (multimodal tokens + queries) is processed by the **frozen** MLLM.
3. **Attention:** **"we continue to use causal masking for the entire sequence rather than specifically enabling full attention for Q."** → queries attend to all preceding image+text tokens via the model's native causal mask; **no special mask needed.** (This is the explicit design AFUN is silent about.)
4. **Read-out:** take the MLLM **output hidden states at the query positions** → these are the "conditions" `C`. The frozen MLLM is used as a **"feature resampler."**
5. **Connector (trainable):** the chosen design is **Enc-Proj** = a **24-layer transformer encoder operating in the MLLM hidden dim, with bi-directional attention** (same block as the base LLM but bidirectional), **then a projection** to the diffusion input dim. (Project-page figures: 24 layers, inner dim ~896 for the small model; ~316M params for the connector reported in the paper text.) Alternative Proj-Enc (project first, then encoder) was worse.
6. **Downstream:** connector output **replaces** the diffusion model's original conditioning ("simply replace its original condition with our C"). Diffusion decoders tried: **SD v1.5** and **Sana-1.6B**.

### 2.2 What is trained vs. frozen
- **MLLM: completely frozen** in the headline result.
- **Trained:** {queries `Q`, connector, diffusion decoder}.
- **Loss:** plain **diffusion denoising objective** on **25M image–caption pairs** (8 epochs); **no task-specific losses** needed. (Image editing / subject-driven use a later instruction-tuning stage.)
- **Tunable variants reported:** Table 2 compares **Frozen MLLM vs. MLLM-tuning vs. E2E-tuning**. E2E (unfrozen) is marginally better on FID, but the paper's thesis is that **freezing preserves the MLLM's understanding** at near-equal generation quality. → supports our decision to keep Qwen3-VL frozen.

### 2.3 The 3 facts the brief asked us to confirm (all TRUE)
- ✅ queries appended to MLLM input,
- ✅ read their **final hidden states**,
- ✅ train **only queries + connector** (MLLM frozen).

---

## 3. LMM → segmentation / region tokens

All entries below: VERIFIED via arXiv API + HTML. Pattern family = **"embedding-as-mask"**: add a special token to the LLM vocabulary; when the LLM emits it, take that token's **last-layer hidden state**, project it, and feed it as a **prompt to a mask decoder** (SAM-style) — except PixelLM (own decoder) and the region-token methods (Groma/Ferret) which encode regions on the *input* side.

### 3.1 LISA — **VERIFIED** (`arXiv:2308.00692`, ICLR'24)
*LISA: Reasoning Segmentation via Large Language Model.* HTML: https://ar5iv.labs.arxiv.org/html/2308.00692
- **Mechanism — embedding-as-mask:** vocabulary is extended with a single **`<SEG>`** token. The LLM (LLaVA) generates `<SEG>`; its **last-layer hidden state** `h_seg` is taken, passed through a projection **MLP γ with channel sizes [256, 4096, 4096]**, and used as the **prompt embedding for the SAM mask decoder**. SAM's image encoder (ViT-H) produces the dense features; the decoder outputs the mask `M̂`.
- **Trained vs frozen:** **SAM ViT-H vision encoder = FROZEN.** LLM = **LoRA** fine-tuned (rank **not stated** in paper) + **LLM word embeddings trainable** (for the new token) + **projection γ trainable** + **SAM mask decoder fully fine-tuned**.
- **Losses:** `L = λ_txt·L_txt + λ_mask·L_mask`, `L_mask = λ_bce·BCE + λ_dice·DICE`, with **λ_txt=1.0, λ_mask=1.0, λ_bce=2.0, λ_dice=0.5**. `L_txt` = autoregressive CE.
- **Relevance to us:** this is the canonical "single special token → SAM prompt." Our **SEM_QUERY** can be exactly this if we route it to SAM2; the [256,4096,4096] MLP and BCE+Dice weights are directly reusable (scale 4096→256 to SAM2 prompt dim; our LLM hidden is 2048 not 4096).

### 3.2 GLaMM — **VERIFIED** (`arXiv:2311.03356`, CVPR'24)
*GLaMM: Pixel Grounding Large Multimodal Model.* HTML: https://arxiv.org/html/2311.03356v3 ; code: https://github.com/mbzuai-oryx/groundingLMM
- **5 components:** Global Image Encoder, Region Encoder, LLM, Grounding Image Encoder, Pixel Decoder.
- **Mask path:** uses `<p>…</p>` to delimit a grounded phrase and **`<SEG>`** for its mask. The `<SEG>` hidden state → **L-P (language-to-prompt) projection `g`** → **pixel decoder `P`** (SAM-decoder-like) with features from a **SAM-based grounding image encoder `V`**: `M = P(g(l_seg), V(x_img))`.
- **Region (input) encoder:** for region prompts, builds a **feature pyramid from 4 CLIP global-encoder layers + RoIAlign → 14×14 → region token** projected to language space (`Rx = R(Ix, r)`).
- **Trained vs frozen:** global + grounding image encoders **frozen**; **LLM LoRA-tuned (α=8)**; region encoder, V-L & L-P projections, and pixel decoder **fully fine-tuned**.
- **Losses:** per-pixel **BCE + DICE** for masks + autoregressive **CE** for text.
- **New task:** Grounded Conversation Generation (GCG); dataset **GranD** (7.5M concepts / 810M regions).

### 3.3 PixelLM — **VERIFIED** (`arXiv:2312.02228`, CVPR'24)
*PixelLM: Pixel Reasoning with Large Multimodal Model.* HTML: https://arxiv.org/html/2312.02228v2 ; page: https://pixellm.github.io/
- **Distinctive: NO SAM.** Uses a **segmentation codebook** `C_seg = {c^ℓ_n}` (N tokens per group × **L visual scales**) — **learnable tokens added to the LLM vocab and generated autoregressively**; N=1 base, N>1 for multiple targets (with a **token-fusion** op).
- **Lightweight pixel decoder:** L attention blocks, one per scale. From codebook hidden states `h={h^ℓ}`, decoder computes `m^ℓ = Attn^ℓ(h^ℓ, f'^ℓ_img)` over **multi-scale CLIP features**, coarse→fine, each scale modulated by the previous mask: `f'^ℓ_img ⊙ (σ(m^{ℓ+1})+1)`.
- **Trained vs frozen:** CLIP encoder **frozen**; trainable = {pixel decoder, **LoRA**, segmentation codebook, projections p_{V→T}, p_{V→D}}.
- **Losses:** `L = L_txt + λ_ref·L_ref + λ_dice·L_dice`; **target-refinement loss** up-weights overlapping pixels (α=2.0).
- **Relevance:** PixelLM proves you can decode masks from special-token hidden states **without SAM**, using a tiny decoder over multi-scale features — relevant if we want grounding maps cheaply in-loop instead of running SAM2 every step.

### 3.4 VideoGLaMM — **VERIFIED** (`arXiv:2411.04923`, CVPR'25)
*VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos.* HTML: https://arxiv.org/html/2411.04923v1 ; page: https://mbzuai-oryx.github.io/VideoGLaMM/
- **3 parts:** LLM + **dual spatio-temporal vision encoder** (CLIP ViT-L/14 spatial + **InternVideo2** temporal) + **spatio-temporal pixel decoder (SAM2-based)**.
- **Mechanism:** grounding triggered by **`<SEG>`**; the **SAM2-based decoder consumes L→V-transformed LLM embeddings** + multi-scale features from the frozen ViT to produce temporally consistent masks.
- **Alignment via tunable V-L and L-V adapters** (the trainable bridges; ViT frozen).
- **Data:** 38k video-QA triplets, 83k objects, 671k masks.
- **Relevance:** closest to our **video** setting; confirms the `<SEG>`→**SAM2** route with temporal decoding (we already use SAM-family weights; AgiBot is video).

### 3.5 u-LLaVA — **VERIFIED** (`arXiv:2311.05348`)
*u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model.* HTML: https://arxiv.org/html/2311.05348v2 ; code: https://github.com/OPPOMKLab/u-LLaVA
- **Mechanism:** special tokens `<img_beg>`, `<vid_beg>`, `<tag>`, `<loc>`, **`<seg>`**. The **`<seg>` hidden states are mapped by a projector into SAM as text embeddings** to drive segmentation. Grounding parses the object label from LLM output, feeds a grounding module, and cross-checks against the mask to get the box.
- **Training:** joint instruction tuning with **task-specific projectors + decoders**, end-to-end; 277K mask-based multi-task dataset. Video supported by adding **two video tokens + minimal trainable params** (relevant: cheap modality extension).

### 3.6 LISA++ — **VERIFIED** (`arXiv:2312.17240`)
*LISA++: An Improved Baseline for Reasoning Segmentation with Large Language Model.* HTML: https://arxiv.org/html/2312.17240v3
- **Same architecture as LISA**; two additions: (1) **instance** segmentation — emits **multiple `<SEG>` tokens**, one per **instance** (LISA used one `<SEG>` per category), trained with **bipartite matching (DETR/MaskFormer-style)**; (2) **Segmentation-in-Dialogue (SiD)** — `<SEG>` tokens placed at natural points in free-form text (no fixed "it is `<SEG>`" template). Achieved purely by **reconstructing instruction-tuning data** (COCO, ADE20K) — no arch change.
- **Relevance:** the **one-token-per-instance + bipartite matching** trick is how to emit a *variable number* of grounded entities — useful if SEM_QUERY must cover several task roles/objects.

---

## 4. Region-token methods (encode regions on the INPUT side)

### 4.1 Groma — **VERIFIED** (`arXiv:2404.13013`, ECCV'24)
*Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models.* page: https://groma-mllm.github.io/ ; code: https://github.com/FoundationVision/Groma
- **Mechanism — localized visual tokenization:** beyond global image tokens, a **general-purpose region proposer (Deformable DETR)** finds ROIs; a **lightweight encoder** turns each ROI into a **region token**. Image encoder = **DINOv2**; LLM = **Vicuna-7B**.
- **Key idea:** the LLM **grounds by referring to region tokens** (it outputs a region-token reference), **avoiding direct coordinate regression** by the LLM. Region tokens are inserted into both user instructions and model responses.
- **Relevance:** an alternative to MetaQuery on the **input** side — pre-tokenize candidate regions, let the frozen LLM *select/reference* them. Could feed our dynamics `region_tokens` directly.

### 4.2 Ferret — **VERIFIED** (`arXiv:2310.07704`, ICLR'24)
*Ferret: Refer and Ground Anything Anywhere at Any Granularity.* HTML: https://arxiv.org/html/2310.07704v1
- **Hybrid region representation:** each region = **discrete coordinates ⊕ continuous features** (jointly), enabling free-form shapes (points / boxes / scribbles / masks).
- **Spatial-aware visual sampler:** randomly samples N points inside the region's binary mask, bilinear-interpolates each point's feature, then cascades **sample → gather → pool** blocks (handles varying sparsity/shape).
- **Relevance:** the **continuous-feature-per-region sampler** is a clean way to turn our **per-Gaussian anchor-uv set** (Stream A) into a region/continuous feature without a proposer — i.e. sample frozen-ViT features at Gaussian uv's.

---

## 5. Concrete faithful recipe for our system (frozen Qwen3-VL-2B, in-loop, `transformers`)

**Target:** add **SEM_QUERY** (e.g. 32 tokens) + **MOTION_QUERY** (e.g. 32 tokens) learnable tokens to a **frozen** Qwen3-VL-2B; read their **last-layer hidden states**; train **only** the query embeddings + small projectors (+ the downstream dynamics/grounding head). This is the **AFUN/MetaQuery recipe** with our dynamics net standing in for AFUN's motion decoder. Mirrors `research_E_DESIGN_draft.md` §1 Stream B.

### 5.1 Verified Qwen3-VL-2B facts that drive the implementation
From the HF `config.json` of `Qwen/Qwen3-VL-2B-Instruct` (corroborated by a second source):
- `hidden_size = 2048` (matches Cosmos-Reason2-2B / our memory). **→ query dim D = 2048.**
- `image_token_id = 151655`, `video_token_id = 151656`, `vision_start_token_id = 151652`, `vision_end_token_id = 151653`.
- `deepstack_visual_indexes = [5, 11, 17]` (ViT features injected into LLM layers 5/11/17 — **see gotcha 5.5(c)**).
- `mrope_section = [24, 20, 20]`, `mrope_interleaved = true`, `rope_theta = 5e6` (**interleaved M-RoPE**).

From `modeling_qwen3_vl.py` (HF `transformers`, main; v4.57-dev):
- **Forward accepts `inputs_embeds` directly** ("specify exactly one of input_ids or inputs_embeds") → **we can build our own embedding sequence**.
- Image injection is **`inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_embeds)`**, where `special_image_mask = (input_ids == config.image_token_id)`. So image features are scattered onto the **image-placeholder positions**, not appended.
- `get_rope_index(...)` returns **`position_ids` of shape `(3, batch, seq_len)`** (temporal/height/width) for M-RoPE, plus `mrope_position_deltas`.
- All-layer hidden states are recordable (`output_hidden_states=True`; `_can_record_outputs` includes `hidden_states`).
- DeepStack: `_deepstack_process` does `hidden_states[visual_pos_masks] += visual_embeds` at the deepstack layers (only affects **visual** positions).

### 5.2 Step-by-step token injection (the right way for Qwen3-VL)

Because Qwen3-VL **merges image features by `masked_scatter` over `image_token_id` positions** (it does *not* simply concatenate a vision prefix), the clean approach is to **let the processor build the normal multimodal sequence, then append our query tokens after it in embedding space.**

```python
# 0) frozen model + processor
model = Qwen3VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16)
model.eval();  [p.requires_grad_(False) for p in model.parameters()]   # FREEZE all

D = model.config.text_config.hidden_size        # 2048
sem_q  = nn.Parameter(torch.randn(32, D) * 0.02)   # SEM_QUERY  (trainable)
mot_q  = nn.Parameter(torch.randn(32, D) * 0.02)   # MOTION_QUERY(trainable)

# 1) normal multimodal batch (image + instruction) via the processor
inputs = processor(text=[prompt], images=[frame0], return_tensors="pt")  # input_ids, pixel_values, image_grid_thw, attention_mask

# 2) build inputs_embeds the way the model does, THEN append queries
#    (a) get image features and scatter them onto image-token positions
inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
image_embeds  = model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)   # frozen ViT (+deepstack feats)
img_mask = (inputs.input_ids == model.config.image_token_id).unsqueeze(-1)
inputs_embeds = inputs_embeds.masked_scatter(img_mask.to(inputs_embeds.device), image_embeds.to(inputs_embeds.dtype))
#    (b) append the learnable queries at the END (after image+text)
B = inputs_embeds.size(0)
q = torch.cat([sem_q, mot_q], 0).unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
inputs_embeds = torch.cat([inputs_embeds, q], dim=1)        # [B, L+64, D]
attn = torch.cat([inputs.attention_mask, torch.ones(B, 64)], dim=1)
```

**Position IDs (M-RoPE) — the load-bearing gotcha (see 5.5):** compute `position_ids` for the real tokens with the model's own helper, then assign the 64 query positions **monotonically continuing the max text position on all 3 rope axes**:

```python
pos, deltas = model.model.get_rope_index(inputs.input_ids, inputs.image_grid_thw, attention_mask=inputs.attention_mask)
# pos: [3, B, L]. Extend by 64 text-like positions (same index on T/H/W axes):
start = pos.amax(dim=-1, keepdim=True) + 1                     # [3, B, 1]
qpos  = start + torch.arange(64, device=pos.device).view(1,1,64)
qpos  = qpos.expand(3, B, 64)
position_ids = torch.cat([pos, qpos], dim=-1)                  # [3, B, L+64]
```

**Forward + read-out (final hidden states at the 64 query positions):**

```python
out = model.model(                      # the inner Qwen3VLModel (LM stack)
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False)
H = out.last_hidden_state               # [B, L+64, D]
sem_h = H[:, -64:-32, :]                # SEM_QUERY  final hidden states  (= "conditions")
mot_h = H[:, -32:,  :]                  # MOTION_QUERY final hidden states
```

(If you prefer the high-level `Qwen3VLForConditionalGeneration.forward`, pass `inputs_embeds`/`attention_mask`/`position_ids`/`output_hidden_states=True` and read `hidden_states[-1]` — same result. Working at the inner `model.model` avoids the LM head.)

### 5.3 What to train (mirror AFUN's 32.21M)
- **Trainable:** `sem_q`, `mot_q`; a **2-layer MLP** per branch projecting `[2048] → downstream dim`; the **downstream heads**:
  - **SEM path:** either (A) MLP→**SAM2 prompt embedding** → mask (LISA/AFUN-style; SAM2 frozen, only fine-tune its mask decoder if desired), or (B) feed `sem_h` straight into `igsw/dynamics` **region cross-attention** (cheaper, no SAM in-loop) — the draft already wires `region_tokens [R,d]`.
  - **MOTION path:** feed `mot_h` into the **existing SC-GS dynamics** as `motion_tokens [Q,d]` (no separate Bézier decoder needed — our dynamics net *is* the motion decoder; this is the one deviation from AFUN and it is a simplification, not a loss of fidelity).
- **Frozen:** entire Qwen3-VL (ViT + LM + deepstack), and (recommended) SAM2 encoder.

### 5.4 What to copy for losses
- **Grounding (if SAM route):** BCE + Dice with **λ_bce=2.0, λ_dice=0.5** (LISA's verified weights); optional Stage-I MSE alignment of `sem_h` to SAM2 text-feature space (AFUN Stage I).
- **Motion:** keep our existing **direct 3D trajectory losses** (the draft's `L_obj_motion`, `L_role_rigid`, `L_bg_static`) — our analogue of AFUN's curve loss.
- **Staged schedule (AFUN-style):** (I) align SEM_QUERY→grounding; (II) train grounding; (III) joint with motion. Reduces the "image-domination" failure noted in the draft.

### 5.5 Qwen3-VL-specific GOTCHAS (do not skip)

(a) **Injection is `masked_scatter` on `image_token_id`, NOT a concat prefix.** If you hand-build `inputs_embeds`, you **must** reproduce the scatter (5.2 step a) or the image features won't land. Do **not** append image features as a prefix.

(b) **M-RoPE position_ids are `(3, B, L)` (interleaved, `mrope_section=[24,20,20]`).** If you append queries and **forget to extend `position_ids`**, the model will either error on shape or silently mis-position the queries. Always extend all **3 axes** (5.2). Putting the queries at **contiguous text-like positions after the max index** is the safe, defensible choice (the sources are silent on the "correct" query positions; this matches how text continues a prompt).

(c) **DeepStack injects ViT features into LLM layers `[5,11,17]` at *visual* positions only.** Our appended queries are **not** visual positions, so deepstack won't touch them directly — they only see deepstack-enriched image tokens **through attention**. This is fine and is exactly the intended path (queries attend to enriched image tokens). Just be aware `get_image_features` returns deepstack embeds too; route them through the normal forward (don't drop them) so the image tokens are properly enriched.

(d) **Causal mask is the native + sufficient choice (per MetaQueries).** Appending queries **after** image+text means the native **causal** mask already lets them attend to everything before them. **Do not** try to hack bidirectional attention into the frozen LM — MetaQueries explicitly keeps causal masking and it works. (AFUN is silent; MetaQueries is explicit → follow MetaQueries.)

(e) **bf16 for attention (our memory): SDPA in fp32 OOMs at 16k tokens.** Keep queries/projectors in bf16 to match; cast only the tiny downstream heads to fp32 if needed for stability. gsplat stays fp32 (separate module).

(f) **`use_cache=False` during training** (no incremental decoding; we read hidden states in one forward).

(g) **Tie the new tokens to no vocab head.** Unlike LISA/PixelLM (which *generate* `<SEG>` and need vocab/embedding entries + word-embedding training), the **MetaQuery pattern does not generate the queries** — they are injected in embedding space and only **read out**. So **no `lm_head`/vocab expansion is required**, and there is **no autoregressive CE on the queries**. This is simpler and is the correct pattern for "extract conditioning, don't generate." (Only adopt the `<SEG>`-generation route if you specifically want the LLM to *decide how many* entities via emitted tokens — cf. LISA++ bipartite — otherwise fixed query counts like AFUN's 32+32 are simpler.)

(h) **Number of queries:** AFUN uses **32 sem + 32 motion**; MetaQueries' generation task wanted **256**. For *conditioning a dynamics net* (not synthesizing an image), **start at AFUN's 32+32**; it's the closer precedent and cheaper. Ablate upward only if grounding is under-resolved.

### 5.6 Relationship to what we already have
Our memory notes an existing **"special-token spatial-aggregation head distilling all 28 layers."** That is a **complementary** mechanism (it pools *existing* layer features). **MetaQuery is different and arguably better for this goal:** the query tokens are **inputs** that the frozen transformer *computes over*, so they can **actively gather task-conditioned information via attention** (per-object, per-intent), rather than passively pooling a fixed forward pass. The draft's key diagnosis — *a single GLOBAL hidden vector cannot carry per-object localization* — is exactly why N learnable queries (each free to specialize to a role/intent) is the right fix. Consider: keep the 28-layer head for dense per-Gaussian features (Stream A side), use MetaQuery SEM/MOTION tokens for the task-conditioned grounding/intent (Stream B). They are not mutually exclusive.

---

## 6. Sources (all opened during verification)

- AFUN: https://arxiv.org/abs/2606.02551 · https://arxiv.org/html/2606.02551 · arXiv API `id_list=2606.02551`
- MetaQueries: https://arxiv.org/abs/2504.06256 · https://arxiv.org/html/2504.06256v1 · https://xichenpan.com/metaquery/ · https://openreview.net/pdf/0ed3644e169be9c9603fd6e341777f869a88d2a7.pdf
- LISA: https://arxiv.org/abs/2308.00692 · https://ar5iv.labs.arxiv.org/html/2308.00692
- GLaMM: https://arxiv.org/abs/2311.03356 · https://arxiv.org/html/2311.03356v3 · https://github.com/mbzuai-oryx/groundingLMM
- PixelLM: https://arxiv.org/abs/2312.02228 · https://arxiv.org/html/2312.02228v2 · https://pixellm.github.io/
- VideoGLaMM: https://arxiv.org/abs/2411.04923 · https://arxiv.org/html/2411.04923v1 · https://mbzuai-oryx.github.io/VideoGLaMM/
- u-LLaVA: https://arxiv.org/abs/2311.05348 · https://arxiv.org/html/2311.05348v2 · https://github.com/OPPOMKLab/u-LLaVA
- LISA++: https://arxiv.org/abs/2312.17240 · https://arxiv.org/html/2312.17240v3
- Groma: https://arxiv.org/abs/2404.13013 · https://groma-mllm.github.io/ · https://github.com/FoundationVision/Groma
- Ferret: https://arxiv.org/abs/2310.07704 · https://arxiv.org/html/2310.07704v1
- Qwen3-VL: HF docs https://huggingface.co/docs/transformers/model_doc/qwen3_vl · code https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py · config `Qwen/Qwen3-VL-2B-Instruct` · Tech report https://arxiv.org/abs/2511.21631

## 7. Honest gaps / UNVERIFIED specifics
- **AFUN attention mask & positional handling for queries:** [SOURCE SILENT] — recipe uses MetaQueries' explicit causal-mask design.
- **AFUN exact λ values** (λ_sam3, λ_curve) and exact K (Bézier control-point count): not pinned down beyond "T=16 sampled points." Not load-bearing for us (we replace the curve head).
- **LISA LoRA rank:** the paper does not state it (HTML and API both silent). Common community value is r=8 but that is **not** from the paper — treat as unverified.
- **MetaQueries connector inner dim / param count:** project page shows 24 layers and ~896 inner dim for the small model and ~316M connector params in text; exact per-config numbers vary by MLLM — verify against the specific table if you reproduce.
- I did **not** find any *separate* paper literally titled "AFUN" other than `2606.02551`; the `2504.06256` ID is unambiguously MetaQueries. No fabrication: both IDs resolve to real, distinct papers.
