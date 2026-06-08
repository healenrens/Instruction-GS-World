# Research E2: Training-Free Visual Grounding via Frozen VLM Attention

**Goal:** From (frame0 image + instruction), extract soft per-region relevance maps for role-phrase nouns ("apple", "the object to move", "destination") — without any fine-tuning — and project onto 3D Gaussians.

---

## 1. Paper: "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding"

**STATUS: VERIFIED**

- **arXiv:** https://arxiv.org/abs/2503.06287 (Kang, Kim, Kim, Hwang; March 2025)
- **Venue:** CVPR 2025 — https://openaccess.thecvf.com/content/CVPR2025/papers/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.pdf
- **HTML full text:** https://arxiv.org/html/2503.06287

### Key Finding
Only **3 out of thousands** of attention heads in a frozen LVLM are sufficient for competitive visual grounding. These are called **localization heads**. The method is entirely **training-free**.

### Head Discovery Algorithm (exact)

**Step 1 — Attention Sum Criterion.** For each layer `ℓ` and head `h`, compute the sum of attention weights directed toward image patches from the last text token's query vector, over a calibration set of 1,000 image-text pairs:

```
S_img^(ℓ,h) = Σ_{i=1}^{P²} a^(ℓ,h)[i]
```

Keep only heads with `S_img^(ℓ,h) ≥ τ`, where τ is chosen at the maximum curvature of the sorted score graph (example: τ=0.24 for LLaVA-1.5-7B).

**Step 2 — Spatial Entropy Criterion.** Among image-focused heads from Step 1:

1. Binarize the attention map: values above mean → 1, below → 0.
2. Find connected components C_i via 8-connectivity.
3. Compute entropy: `H(A^(ℓ,h)) = −Σ_i P(C_i) log P(C_i)`, where `P(C_i) = |C_i| / Σ_j |C_j|`.
4. Lower entropy = more localized attention.

**Step 3 — Selection by frequency.** Over the 1,000 calibration samples, for each sample rank all heads by spatial entropy; record how often each head appears in the top-10 lowest-entropy heads. Select the k heads with highest selection frequency. **k=3 is optimal across all tested models** (ablation shows both criteria jointly needed; using either alone drops accuracy from 67.4% to ~25%).

### Attention Map → Bounding Box Pipeline

Given a query image+instruction, at inference:

1. **Query token:** Use the query vector of the **last input text token** (encapsulates full context).
2. **Extract:** For each of the k=3 localization heads, read the text-to-image attention row `a^(ℓ,h)[last_text_tok, image_positions]`, reshape to P×P patch grid.
3. **Aggregate:** Element-wise sum of the k maps → combined map M.
4. **Smooth:** Apply Gaussian filter with kernel_size=7, σ=1.0.
5. **Binarize:** Values above mean → 1.
6. **Box:** Apply convex hull algorithm; take the smallest tight bounding box enclosing the largest hull region.
7. **Mask (optional):** Feed box as prompt to SAM for dense segmentation.

### Models Tested
LLaVA-1.5 (7B, 13B), LLaVA (7B, 13B), DeepSeek-VL (1.3B, 7B), InternVL (6B), Yi-VL (6B), ShareGPT4V (7B), Mini-Gemini (2B). Best: 87.2% REC / 76.1% RES on RefCOCO with LLaVA-1.5-13B. Note: **Qwen3-VL not directly tested**, but the architecture is analogous (decoder-only, image tokens in-sequence).

---

## 2. Paper: PnP-OVSS — "Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models"

**STATUS: VERIFIED**

- **arXiv:** https://arxiv.org/abs/2311.17095 (Nov 2023, updated v4)
- **HTML:** https://arxiv.org/html/2311.17095v4
- **Venue:** CVPR 2024

### Core Claim: Pooled Embeddings vs. Cross-Attention
The key insight is that VLMs using pooled embeddings (e.g., CLIP) produce a single vector encoding that "discards information about detailed positions of objects and words." Models with explicit **text-to-image cross-attention** (BLIP, BridgeTower) preserve **patch-word correspondence** at fine granularity — a P×P map per class name rather than a scalar.

### Algorithm

**Backbone models used:** BLIP (ViT-L/16, BERT), BridgeTower (ViT-L/14, RoBERTa, 6-layer cross-modal encoder with 16 heads).

**Step 1 — Cross-Attention Extraction.** For each class name k, extract the K×P×P cross-attention tensor from the cross-attention layers (text queries, image keys/values). For multi-token class names, average attention maps across tokens. Exclude the first three tokens ("A", "picture", "of").

**Step 2 — GradCAM Sharpening.** Raw attention maps over-segment (attend broadly). Apply GradCAM with the image-text matching (ITM) loss to sharpen:

```
M̃^(k) = max(0, ∂L_ITM/∂M^(k)) ⊗ M^(k)
```

Gradients are computed treating the pair as a match. This focuses on *discriminative* patches (e.g., the head of an elephant) but tends to under-segment.

**Step 3 — Salience Dropout (4 iterations).** To recover the full object extent:
1. Compute class-agnostic salience: `U^(t) = Σ_k M̃^(k,t)`
2. Drop 50% of remaining patches with highest salience (zero out above-median patches in current pass)
3. Recompute GradCAM on remaining patches
4. Repeat for t=1..4 (stops after ~93.75% patches touched cumulatively)
5. Combine: `M̂^(k) = Σ_{t=1}^{4} M̃^(k,t)`

Dropped patches remain zeroed in all subsequent iterations.

**Step 4 — Post-processing.** Threshold at T, Gaussian blur (σ), Dense CRF with color consistency.

**Performance gains:** +26.2% mIoU on Pascal VOC, +20.5% on MS COCO over comparable baselines.

### Relevance to Qwen3-VL
PnP-OVSS targets encoder-decoder VLMs (BLIP, BridgeTower) with explicit cross-attention. **Qwen3-VL is a decoder-only model** — there is no dedicated cross-attention block between text and image. Instead, text and image tokens share a single self-attention mechanism. The GradCAM + Salience Dropout ideas remain applicable to the text-to-image slices of self-attention, but require adaptation (see Section 5).

---

## 3. DAAM — "What the DAAM: Interpreting Stable Diffusion Using Cross Attention"

**STATUS: VERIFIED**

- **arXiv:** https://arxiv.org/abs/2210.04885 (Tang et al., 2022/2023)
- **ACL 2023:** https://aclanthology.org/2023.acl-long.310
- **Code:** https://github.com/castorini/daam

### Algorithm (from source code + paper)

DAAM targets **Stable Diffusion's U-Net cross-attention**, where text tokens are queries and image spatial locations are keys/values — structurally analogous to cross-attention in encoder-decoder VLMs, not decoder-only.

**Extraction:** A hook (`UNetCrossAttentionHooker`) captures `attention_probs` (post-softmax) from every U-Net cross-attention layer during each denoising timestep.

**Spatial Reshape:** Raw attention has shape `[tokens, heads, N]` where N = H_feat × W_feat (flattened spatial). Reshape to `[tokens, heads, H_feat, W_feat]` using `h = w = int(sqrt(x.size(1)))`.

**Upscaling:** Each layer's attention map is upsampled to a common resolution (64×64 or 96×96) via **bicubic interpolation**: `F.interpolate(heat_map, size=(x,x), mode='bicubic').clamp_(min=0)`.

**Aggregation across layers and heads:**
```python
maps = stack_of_all_maps  # shape [N_layers*N_heads, resolution, resolution, vocab_size]
maps = maps.mean(0)[:, 0]  # mean over layers/heads; take first token map
```
Aggregation across timesteps is implicit (maps are accumulated throughout the entire denoising trajectory).

**Normalization:** Optional probability-style normalization excluding special tokens:
```
maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)
```

### Relevance to VLMs
DAAM was designed for Stable Diffusion's **cross-attention**. For decoder-only LMMs, there is no separate cross-attention — the same principle (summing attention weights from text token positions to image token positions) must be applied to the **self-attention** of the LLM backbone. The DAAM insight (aggregate across layers, smooth/normalize) transfers directly; the upscaling step also applies since attention operates on merged patch tokens, not original pixels.

### "Attention-as-Grounding" Family for VLMs (2024–2025)

Several recent works confirm self-attention extraction works for decoder-only VLMs:

- **Entropy-Gradient Grounding** (arXiv:2604.08456) — extracts evidence from VLM self-attention training-free. **STATUS: UNVERIFIED** (found in search results but abstract not fetched in full).
- **FasterVLM** (https://theia4869.com/FasterVLM/) — uses CLS-to-image cross-attention in the *visual encoder* (ViT) for token pruning; notes that LLM cross-modal text-to-image attention is "inaccurate" for this purpose. This is a warning for our use case (see caveats in Section 5).
- **MDSAM** (arXiv:2506.17664) — training-free hallucination mitigation by tracking image-token attention dynamically. **STATUS: UNVERIFIED** (found in search, not fetched).

---

## 4. Qwen3-VL Architecture: Extracting Text→Image Attention

**Sources:**
- Qwen3-VL HuggingFace doc: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
- Qwen2-VL HuggingFace doc: https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl
- Modeling code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631

### Architecture Summary

Qwen3-VL (and Qwen2-VL) is a **decoder-only** causal LLM where:

- Image patches are encoded by a ViT visual encoder (patch_size=16 for Qwen3-VL; 14 for Qwen2-VL, temporal_patch_size=2).
- A **spatial merger** with `spatial_merge_size=2` compresses 2×2 patch neighborhoods → 1 LLM token, reducing the token count by 4×.
- The merged visual tokens replace `<|image_pad|>` (token_id=151655) placeholders in the sequence, bracketed by `<|vision_start|>` (151652) and `<|vision_end|>` (151653).
- The LLM then sees a flat sequence: `[system tokens ... <|vision_start|> [image_pad×N_vis] <|vision_end|> instruction_tokens ...]`
- **mRoPE (Multimodal RoPE):** Position IDs are 3D tensors `[3, batch, seq_len]` = (temporal, height, width). Text tokens get equal values in all three dims (effectively 1D). Image tokens get distinct (h, w) coordinates matching their spatial location in the merged patch grid. In Qwen3-VL this uses **interleaved MRoPE layout** (enhanced vs Qwen2-VL).
- **DeepStack (Qwen3-VL only):** ViT intermediate features at layers [8, 16, 24] are also injected into the LLM, tightening vision-language alignment. `deepstack_visual_indexes = [8, 16, 24]`.

### `image_grid_thw` Structure

`image_grid_thw` is a tensor of shape `[num_images, 3]` where each row is `[t, h, w]`:
- `t` = temporal grid size (=1 for static images)
- `h` = height grid size **after** spatial merging = `ceil(H_pixels/patch_size) // spatial_merge_size`
- `w` = width grid size **after** spatial merging = `ceil(W_pixels/patch_size) // spatial_merge_size`

Number of LLM image tokens per image = `t × h × w`.

**Example:** 448×448 image, patch_size=14, spatial_merge_size=2:
- ViT patches: 32×32 = 1024
- After merge: 16×16 = 256 LLM tokens
- `image_grid_thw[0] = [1, 16, 16]`

### Precise Extraction Algorithm for Qwen3-VL

Below is a complete, implementable procedure.

```python
import torch
import torch.nn.functional as F
import numpy as np

# ── Step 0: Setup ──────────────────────────────────────────────────────
# Must use attn_implementation="eager" or "sdpa" — NOT flash_attention_2
# (flash_attention does NOT return attention weights)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    attn_implementation="eager",   # CRITICAL: flash_attn drops attn weights
    torch_dtype=torch.float32,     # bfloat16 may lose precision in attention
    device_map="auto"
)
model.eval()

# ── Step 1: Forward pass with output_attentions=True ──────────────────
with torch.no_grad():
    outputs = model(
        **inputs,                       # processor output: input_ids, attention_mask,
                                        # pixel_values, image_grid_thw
        output_attentions=True,         # returns tuple of per-layer attn matrices
        return_dict=True
    )

# outputs.attentions: tuple of L tensors, each [batch, num_heads, seq_len, seq_len]
# L = num_hidden_layers (e.g., 32 for 8B)
# num_heads = 32 (8B model; GQA: num_key_value_heads may differ)
attentions = outputs.attentions  # len = L

# ── Step 2: Locate image token positions in input_ids ─────────────────
input_ids = inputs["input_ids"][0]  # [seq_len]
IMAGE_TOKEN_ID = 151655             # <|image_pad|>; same for Qwen2-VL and Qwen3-VL
image_mask = (input_ids == IMAGE_TOKEN_ID)  # bool tensor [seq_len]
image_positions = image_mask.nonzero(as_tuple=True)[0]  # 1D indices of image tokens

# Number of image tokens should equal t*h*w from image_grid_thw
t, h, w = inputs["image_grid_thw"][0]   # e.g., [1, 16, 16]
assert len(image_positions) == t * h * w

# ── Step 3: Locate text phrase token positions ─────────────────────────
# Tokenize the role phrase separately to find its token IDs
phrase = "apple"
phrase_ids = processor.tokenizer.encode(phrase, add_special_tokens=False)
# Find positions in input_ids where this sub-sequence occurs
# Simple approach: search for the first occurrence
def find_subseq(seq, subseq):
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m].tolist() == subseq:
            return list(range(i, i+m))
    return []

phrase_positions = find_subseq(input_ids, phrase_ids)
# phrase_positions: list of indices in input_ids

# ── Step 4: Select layers and heads ────────────────────────────────────
# Option A: Use ALL layers and ALL heads (baseline, mean-aggregation)
# Option B: Use only the last few layers (layers -4 to -1) — empirically richer
# Option C: Run the localization-head discovery from [arXiv:2503.06287]
#   (requires calibration set, see Section 1)
#
# For quick use without calibration, Option B is recommended:
layers_to_use = list(range(len(attentions) - 4, len(attentions)))  # last 4 layers

# ── Step 5: Extract text→image attention slice ─────────────────────────
# For each selected layer and head, extract attn[phrase_positions, image_positions]
# attn shape: [batch=1, heads, seq_len, seq_len]
# Row = query (which token is attending), Col = key (what it attends to)

phrase_to_image_maps = []  # will collect [H_patches, W_patches] maps

for layer_idx in layers_to_use:
    attn_layer = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
    for head_idx in range(attn_layer.shape[0]):
        head_attn = attn_layer[head_idx]   # [seq_len, seq_len]
        # Rows = phrase tokens, Cols = image tokens
        phrase_img_attn = head_attn[phrase_positions, :][:, image_positions]
        # shape: [num_phrase_tokens, num_image_tokens]
        # Aggregate over phrase tokens (mean)
        map_flat = phrase_img_attn.mean(dim=0)  # [num_image_tokens]
        # Reshape to 2D patch grid (for static image, t=1)
        map_2d = map_flat.reshape(h, w)         # [h_grid, w_grid]
        phrase_to_image_maps.append(map_2d)

# Stack all collected maps
all_maps = torch.stack(phrase_to_image_maps, dim=0)  # [N_maps, h, w]

# ── Step 6: Aggregate across layers/heads ──────────────────────────────
# Simple mean (DAAM-style):
agg_map = all_maps.mean(dim=0)  # [h, w]

# ── Step 7: Upsample to image pixel space ─────────────────────────────
# patch_size=16, spatial_merge_size=2 → each LLM token covers 32×32 pixels
patch_stride = model.config.vision_config.patch_size * model.config.vision_config.spatial_merge_size
H_px = h * patch_stride
W_px = w * patch_stride

agg_map_upsampled = F.interpolate(
    agg_map.unsqueeze(0).unsqueeze(0),  # [1, 1, h, w]
    size=(H_px, W_px),
    mode="bicubic",
    align_corners=False
).squeeze().clamp(min=0)  # [H_px, W_px]

# ── Step 8: Normalize to [0, 1] ────────────────────────────────────────
vmin, vmax = agg_map_upsampled.min(), agg_map_upsampled.max()
relevance_map = (agg_map_upsampled - vmin) / (vmax - vmin + 1e-8)
# relevance_map: float32 [H_px, W_px], values in [0,1]

# ── Step 9 (optional): Gaussian smoothing (DAAM / localization-head style)
from torchvision.transforms.functional import gaussian_blur
relevance_map = gaussian_blur(
    relevance_map.unsqueeze(0), kernel_size=7, sigma=1.0
).squeeze()
```

### Mapping from LLM Image Token Index → Pixel Region

Given image token index `i` in the flat image token sequence (0-indexed within the image block, not the full sequence):

```python
# token i corresponds to merged patch grid cell (row, col):
row = i // w
col = i % w

# Pixel region covered (top-left corner of the 32×32 pixel block):
px_y = row * patch_stride   # = row * (patch_size * spatial_merge_size)
px_x = col * patch_stride
# Gaussian anchor pixel for this token:
anchor_y = px_y + patch_stride // 2
anchor_x = px_x + patch_stride // 2
```

This is the inverse mapping needed to look up, for each 3D Gaussian whose anchor lies at pixel `(py, px)`, which image token it corresponds to:

```python
token_row = py // patch_stride
token_col = px // patch_stride
token_idx = token_row * w + token_col
```

---

## 5. Concrete Algorithm: Per-Role-Phrase Relevance Map from Frozen Qwen3-VL

**Full pipeline (training-free):**

```
INPUTS:
  frame0:  PIL image (H×W pixels)
  instruction: string (e.g. "Move the apple to the basket")
  role_phrases: list of strings (e.g. ["apple", "basket"])

OUTPUTS:
  relevance_maps: dict mapping phrase → float32 tensor [H_patch, W_patch]
  (can be upsampled to pixel space; each value ∈ [0,1])
```

**Procedure:**

1. **Process inputs.** Run `processor.apply_chat_template` with the image + instruction. Get `input_ids`, `pixel_values`, `image_grid_thw`, `attention_mask`. Read `(t, h, w) = image_grid_thw[0]`.

2. **Forward pass.** Call `model(**inputs, output_attentions=True)`. Collect `outputs.attentions` (L tensors of shape `[1, num_heads, seq_len, seq_len]`).
   - **CRITICAL:** Must use `attn_implementation="eager"` (not flash_attention_2). With flash attention, `output_attentions=True` is unsupported and will error or return None.

3. **Find image token positions.** `image_positions = (input_ids[0] == 151655).nonzero().squeeze()`. Length should equal `t*h*w`.

4. **For each role phrase:**
   a. Tokenize phrase → `phrase_ids`. Find first occurrence in `input_ids[0]` → `phrase_positions`.
   b. For layers in selected subset (recommendation: last 4–8 layers; OR use calibrated localization heads per §1):
      - For each head: extract `attn[phrase_positions, :][:, image_positions]`, mean over phrase tokens → `map_flat` shape `[h*w]`, reshape to `[h, w]`.
   c. Stack all per-head-per-layer maps, take mean → `agg_map [h, w]`.
   d. Bicubic upsample to pixel resolution `[h*patch_stride, w*patch_stride]`.
   e. Normalize to `[0,1]`. Optional: Gaussian smoothing (kernel=7, σ=1.0).
   f. Store as `relevance_maps[phrase]`.

5. **Sample at Gaussian anchors.** For each Gaussian with anchor pixel `(py, px)`:
   ```python
   val = relevance_map[py, px]  # or bilinear-sample from the upsampled map
   ```
   This is the soft relevance weight for the role phrase at that Gaussian.

---

## 6. Critical Caveats and Reliability Notes

### (A) Flash Attention incompatibility
`output_attentions=True` requires `attn_implementation="eager"` or `"sdpa"`. Flash Attention does not materialize attention weights in standard inference. This significantly increases memory usage (O(seq²) attention matrices per layer). For a 672-token sequence, each layer is [1, 32, 672, 672] × 4 bytes ≈ 55 MB; × 32 layers = ~1.8 GB extra. Plan accordingly.

### (B) Causal attention mask
Self-attention in Qwen3-VL is **causal** (lower-triangular mask). Image tokens appear before text tokens in the sequence. Therefore:
- Text-token queries CAN attend to image-token keys (image comes earlier in sequence). ✓
- **BUT:** image-token queries CANNOT attend to text tokens (future). This means `attn[image_pos, text_pos]` is masked out.
- We want text→image attention (rows=text phrase positions, cols=image positions), which is correctly accessible.

### (C) mRoPE and attention bias
mRoPE modifies query-key dot products by injecting position-dependent rotation. The rotation for image tokens encodes spatial (row, col) coordinates, while text tokens have uniform rotation. This may cause the model to systematically dampen attention to spatially distant image tokens regardless of semantic relevance. The localization-head selection procedure in §1 empirically filters for heads where this effect is semantic, not merely positional.

### (D) GQA (Grouped Query Attention)
Qwen3-VL uses GQA: `num_attention_heads` may differ from `num_key_value_heads`. When `output_attentions=True`, the returned attention matrices are for the full `num_attention_heads` dimension (keys/values are broadcast to all heads in the group). This is correct for our use case.

### (E) DeepStack multi-level features (Qwen3-VL)
Qwen3-VL injects intermediate ViT features at layer indices `[8, 16, 24]` of the ViT via the DeepStack mechanism. This adds additional image-related tokens interleaved with the LLM sequence. **These tokens will also have `image_token_id=151655`.** The total `N_vis` tokens includes DeepStack tokens. The effective spatial layout may be more complex than a simple `t×h×w` grid. **Recommendation:** Verify `len(image_positions) == t*h*w` and investigate if it differs before assuming the simple reshape is correct.

### (F) Signal quality
Empirical evidence from FasterVLM (https://theia4869.com/FasterVLM/) notes that "cross-modal text-to-image attention in LLMs is inaccurate" compared to ViT CLS-to-patch attention. This echoes findings in the localization-head paper that *only 3 of thousands of heads* are semantically grounded — most heads' attention is not spatially meaningful. The calibrated head selection procedure (§1) is therefore important for clean signal. Using raw mean-over-all-heads maps will be noisy.

### (G) Decoder-only vs. cross-attention architectures
PnP-OVSS (§2) is designed for encoder-decoder models with dedicated cross-attention. The GradCAM + Salience Dropout cannot be directly applied to Qwen3-VL without modifications. However, the concept of iterative masking of high-salience patches could be adapted by re-running inference with masked image regions.

---

## 7. Recommendation for Instruct-GS-World

**Practical starting point (no calibration set needed):**

1. Use Qwen3-VL-7B or 8B with `attn_implementation="eager"`.
2. Forward-pass with `output_attentions=True` on the (frame0, instruction) pair.
3. Use last 8 LLM layers, all heads, mean aggregation.
4. For each noun/role phrase in the instruction: extract text→image attention as described in §4.
5. Bicubic upsample, normalize, smooth.

**Better (requires ~1000-sample calibration set from your manipulation domain):**

Run the localization-head discovery from §1 (arXiv:2503.06287) on Qwen3-VL specifically. Select the top-3 heads. Use only those heads for map extraction. This dramatically reduces noise.

**Reliability:** Expect useful but imperfect localization. The maps will be coarse (16×16 to 32×32 patch grid) and may conflate similar-looking objects. For objects mentioned in the instruction that are visually prominent, signal quality is reasonable. For ambiguous role phrases ("destination"), the maps will be diffuse. Post-hoc filtering with object detection (e.g., Grounding DINO) can serve as a complementary fallback.

---

## References (all verified unless noted)

| Paper | arXiv / URL | Status |
|-------|------------|--------|
| Localization Heads (CVPR 2025) | https://arxiv.org/abs/2503.06287 | VERIFIED |
| PnP-OVSS (CVPR 2024) | https://arxiv.org/abs/2311.17095 | VERIFIED |
| DAAM (ACL 2023) | https://arxiv.org/abs/2210.04885 | VERIFIED |
| DAAM code | https://github.com/castorini/daam | VERIFIED |
| Qwen3-VL Technical Report | https://arxiv.org/abs/2511.21631 | VERIFIED |
| Qwen3-VL HF docs | https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl | VERIFIED |
| Qwen2-VL HF docs | https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl | VERIFIED |
| Qwen2-VL modeling code | https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py | VERIFIED |
| Entropy-Gradient Grounding | https://arxiv.org/abs/2604.08456 | UNVERIFIED (found in search, not fetched) |
| FasterVLM | https://theia4869.com/FasterVLM/ | VERIFIED (site fetched) |
| MDSAM | https://arxiv.org/abs/2506.17664 | UNVERIFIED (found in search, not fetched) |
