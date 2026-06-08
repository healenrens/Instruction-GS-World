# research_G — MetaQuery conditioning: precise implementation design

> Faithful, line-cited analysis of the Qwen3-VL dense forward in transformers **5.10.2**
> (file `.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`).
> Code draft in §3 is syntax-checked but not GPU-run (GPUs busy).
> Written 2026-06-06.

---

## 1. Qwen3-VL dense forward — line-cited anatomy

### 1.1 Class hierarchy

```
Qwen3VLForConditionalGeneration          # the full model (what QwenVLEncoder loads)
  .model : Qwen3VLModel                  # line 1290 — hosts visual + text submodels
    .visual : Qwen3VLVisionModel         # line 871 — 24-layer ViT + patch merger + deepstack mergers
    .language_model : Qwen3VLTextModel   # line 872 — 28-layer causal decoder
      .embed_tokens                      # word embedding table [151936, 2048]
      .layers[0..27] : Qwen3VLTextDecoderLayer   # line 529
      .norm                              # final RMSNorm
      .rotary_emb : Qwen3VLTextRotaryEmbedding   # M-RoPE, mrope_section=[24,20,20]
  .lm_head                               # line 1291 — NOT used by QwenVLEncoder
```

### 1.2 Qwen3VLModel.forward (lines 1165–1280)

Signature (lines 1165–1177):
```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> ...
```

**Step-by-step (image-only path, our use case):**

1. **Word embedding** (line 1188): `inputs_embeds = self.get_input_embeddings()(input_ids)` — shape `[1, L, 2048]`, where L = #text_tokens + #image_placeholder_tokens (image_token_id=151655).

2. **Vision tower** (lines 1190–1200): `get_image_features(pixel_values, image_grid_thw)` runs `self.visual(pixel_values, grid_thw=image_grid_thw)`. This:
   - Patch-embeds the image with Conv3d (`temporal_patch_size=2, patch_size=16`) → raw patches of dim 1024
   - Runs through 24 ViT blocks. At blocks `deepstack_visual_indexes=[5,11,17]`, saves the output through a separate merger (`deepstack_merger_list`, each a `Qwen3VLVisionPatchMerger`) → this produces `deepstack_features` = list of 3 tensors, each shape `[n_vis_merged, 2048]`.
   - Final merger (`self.merger`) reduces the patch sequence: output `pooler_output` = list of per-image feature tensors, each `[n_vis_merged, 2048]`. For one image of grid `(T=1, H, W)`, `n_vis_merged = (H*W) / (spatial_merge_size**2) = (H*W)/4`.
   
   Returns `BaseModelOutputWithDeepstackFeatures` with `.pooler_output` (list of tensors) and `.deepstack_features` (list of 3 tensors).

3. **Image injection via masked_scatter** (lines 1200–1206):
   ```python
   image_embeds = torch.cat(image_embeds, dim=0)          # [n_vis_merged, 2048]
   image_mask, _ = self.get_placeholder_mask(             # line 1073
       input_ids, inputs_embeds, image_features=image_embeds)
   inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
   ```
   `get_placeholder_mask` (lines 1073–1112): finds positions where `input_ids == image_token_id (151655)`, returns `[1, L, 2048]` bool mask expanded along hidden dim. `masked_scatter` writes `image_embeds` into those positions (it's a FLAT write — requires that `n_image_tokens * 2048 == image_embeds.numel()`).

4. **visual_pos_masks and deepstack assembly** (lines 1207–1240): The boolean mask over the image token positions `[1, L]` becomes `visual_pos_masks`. The 3-element `deepstack_image_embeds` list flows to `deepstack_visual_embeds` unchanged.

5. **Position IDs via M-RoPE** (lines 1241–1254): `compute_3d_position_ids` calls `get_rope_index` which produces `position_ids` of shape `[3, 1, L]`. Shape: 3 axes (temporal, height, width); for text tokens all 3 are identical (monotone); for image tokens they encode the (t, h, w) position in the image grid (lines 1014–1031 in `get_rope_index`). Returns also `mrope_position_deltas` (shape `[1, 1]` = `max_position + 1 - len(sequence)`).

6. **Text model call** (lines 1253–1261):
   ```python
   outputs = self.language_model(
       input_ids=None,   # <-- ALWAYS None here; uses inputs_embeds
       position_ids=position_ids,
       attention_mask=attention_mask,
       inputs_embeds=inputs_embeds,
       visual_pos_masks=visual_pos_masks,
       deepstack_visual_embeds=deepstack_visual_embeds,
       **kwargs,
   )
   ```

### 1.3 Qwen3VLTextModel.forward (lines 765–867)

Signature (lines 765–780):
```python
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    # deepstack args:
    visual_pos_masks: torch.Tensor | None = None,          # [B, L] bool
    deepstack_visual_embeds: list[torch.Tensor] | None = None,  # list of [n_vis, 2048]
    **kwargs: Unpack[FlashAttentionKwargs],
) -> ...
```

**Key internals:**

- **Position IDs reshaping** (lines 793–806): If `position_ids` is `None`, creates `[4, B, L]` (text+T+H+W). If `position_ids.ndim == 3` and `shape[0] == 4`, splits off `text_position_ids = position_ids[0]` (used for mask/cache) and passes `position_ids[1:]` (shape `[3, B, L]`) to the rotary embedding. **The M-RoPE therefore takes shape `[3, B, L]`.** The 4th axis is the text (absolute) position ID used only for causal mask creation.

  > NOTE: `get_rope_index` returns shape `[3, B, L]` (temporal/height/width only — no text axis). When this is passed as `position_ids` to `Qwen3VLTextModel`, the shape-3 check at line 802 triggers `text_position_ids = None` (since shape[0]==3, not 4). The causal mask is created without explicit position_ids. This is the current code path. The 4-dim path requires prepending a text axis.

- **Rotary embedding** (lines 817–819):
  ```python
  position_embeddings = self.rotary_emb(hidden_states, position_ids)
  ```
  `Qwen3VLTextRotaryEmbedding.forward` (lines 354–394): takes `position_ids` of shape `[3, B, L]`, applies `mrope_section=[24, 20, 20]` interleaving (the interleaved M-RoPE). Returns `(cos, sin)` each `[B, L, head_dim]`.

- **Decoder loop with deepstack** (lines 820–845):
  ```python
  for layer_idx, decoder_layer in enumerate(self.layers):   # 28 iterations
      hidden_states = decoder_layer(hidden_states, attention_mask, ...)
      if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
          # deepstack_visual_embeds has 3 entries (for layers [0,1,2] = idx 0,1,2)
          hidden_states = self._deepstack_process(
              hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])
  ```
  `_deepstack_process` (lines 850–857):
  ```python
  def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
      hidden_states = hidden_states.clone()
      local_this = hidden_states[visual_pos_masks, :] + visual_embeds
      hidden_states[visual_pos_masks, :] = local_this
      return hidden_states
  ```
  **Critical**: `visual_pos_masks` is boolean `[B, L]`; indexing `hidden_states[visual_pos_masks, :]` selects only the visual-token rows across the batch. Appended query tokens are OUTSIDE the visual span (not image_token_id in input_ids), so `visual_pos_masks` does NOT include them — deepstack does NOT touch query positions. ✓

- **output_hidden_states mechanism** (lines 596–598, output_capturing.py):
  ```python
  _can_record_outputs = {
      "hidden_states": Qwen3VLTextDecoderLayer,   # hook on every decoder layer
      "attentions": Qwen3VLTextAttention,
  }
  ```
  When `output_hidden_states=True` is passed, the `@capture_outputs` decorator installs hooks that collect the output of every `Qwen3VLTextDecoderLayer.forward` call (shape `[B, L, 2048]`) and the initial `inputs_embeds` as the 0th entry, resulting in `out.hidden_states` = tuple of 29 tensors (embeds + 28 layer outputs). The existing conditioning code does `hidden_all = torch.stack(hs[1:1+num_layers], dim=0)[:, 0]` → shape `[28, L, 2048]`.

### 1.4 get_rope_index (lines 936–1029)

Iterates over batch items. For each batch item:
- Splits the token sequence into groups by `mm_token_type_ids` (0=text, 1=image, 2=video).
- For text groups: assigns positions `[current_pos, current_pos+text_len)` on all 3 axes (monotone 3D → text tokens are "flat" on all axes).
- For image groups: calls `get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, device)` which assigns 3D positions in the merged (T//1, H//2, W//2) grid, offset by `current_pos`.
- After the image block, `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`.
- Returns `mrope_position_deltas[b] = llm_positions.max() + 1 - len(sequence)`.

**Max position**: at the end of the sequence, `llm_positions.max()` = the largest of the 3 axes = typically the larger of H/2 or W/2 (for image-dominated sequences) + the text prefix offset. For the query extension, we need `max_pos = llm_positions.max() + 1`.

### 1.5 Summary of what actually changes for MetaQuery

Current path: `input_ids=[BOS+<img_tok...>+text_tok+EOS]`, L tokens → `inputs_embeds=[1,L,2048]` (image positions overwritten) → 28 layers → `out.hidden_states[1..28]` each `[1,L,2048]` → slice off LAST N from each → `[28, N, 2048]`.

MetaQuery appends N learnable query embeddings AFTER the normal sequence:
- Total length `L' = L + N`.
- `inputs_embeds = cat([normal_embeds, query_embeds], dim=1)` — shape `[1, L', 2048]`.
- `attention_mask` extended with N ones.
- `position_ids` extended: query positions get "text-like" coords `[max_pos, max_pos+1, ..., max_pos+N-1]` on all 3 axes (same as how a text suffix would get positions).
- `deepstack_visual_embeds` and `visual_pos_masks` are UNCHANGED (query positions are not in visual span).
- Pass `output_hidden_states=True` → `out.hidden_states[j][:, -N:, :]` = query hidden states at layer `j`.

---

## 2. Step-by-step MetaQuery append mechanism

### Step (a) — Build inputs_embeds with vision injection WITHOUT modifying input_ids

**Problem**: `Qwen3VLModel.forward` normally handles both embedding and injection. We need to split this:
1. Prepare inputs exactly as the processor does (with `image_token_id` placeholders in `input_ids`).
2. Call the model's own `get_image_features` helper to run the ViT + deepstack merger.
3. Call `get_placeholder_mask` to get the injection mask.
4. Do `masked_scatter` ourselves to produce `inputs_embeds`.

This avoids reimplementing the vision tower. We call the model at `self.model.model` (= `Qwen3VLModel`) level:

```python
# Inside QwenVLEncoder.forward_metaquery
qwen_model = self.model.model   # Qwen3VLModel instance

# (a1) Word embed — produces placeholder embeddings for image tokens
inputs_embeds = qwen_model.get_input_embeddings()(input_ids)  # [1, L, 2048]

# (a2) Vision tower: ViT + deepstack mergers
image_out = qwen_model.get_image_features(
    pixel_values, image_grid_thw, return_dict=True)
image_embeds = torch.cat(image_out.pooler_output, dim=0)  # [n_vis, 2048]
deepstack_visual_embeds = image_out.deepstack_features    # list of 3, each [n_vis, 2048]

# (a3) Injection via masked_scatter
image_mask, _ = qwen_model.get_placeholder_mask(
    input_ids, inputs_embeds, image_features=image_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
# inputs_embeds: [1, L, 2048], image positions overwritten with ViT features

# (a4) visual_pos_masks for deepstack
visual_pos_masks = image_mask[..., 0]   # [1, L] bool
```

### Step (b) — Concatenate query embeddings

```python
# query_embeds: nn.Parameter [N, 2048], shared across layers, on same device/dtype
q = query_embeds[None].to(inputs_embeds.dtype)     # [1, N, 2048]
inputs_embeds_ext = torch.cat([inputs_embeds, q], dim=1)   # [1, L+N, 2048]
```

### Step (c) — Extend attention_mask

```python
# Original attention_mask: [1, L] with 1 for valid, 0 for padding
query_mask = torch.ones(1, N, dtype=attention_mask.dtype, device=attention_mask.device)
attention_mask_ext = torch.cat([attention_mask, query_mask], dim=1)   # [1, L+N]
```

### Step (d) — Extend position_ids with text-like M-RoPE from max+1

`get_rope_index` requires `input_ids` and `mm_token_type_ids` to build position_ids. We call it on the ORIGINAL `input_ids` (without the query tokens, since get_rope_index uses input_ids only to parse modality groups). This gives us the 3D positions for the original L tokens plus `mrope_position_deltas`. Then we extend manually:

```python
# Get positions for the original L-token sequence
pos_ids, rope_deltas = qwen_model.get_rope_index(
    input_ids=input_ids,
    mm_token_type_ids=mm_token_type_ids,
    image_grid_thw=image_grid_thw,
    attention_mask=attention_mask,
)
# pos_ids: [3, 1, L]; rope_deltas: [1, 1]

# Max position used by the sequence:
max_pos = pos_ids.max().item()   # scalar; = llm_positions.max() over all 3 axes
# Positions for N appended query tokens: text-like = same value on all 3 axes
query_pos = torch.arange(N, device=input_ids.device, dtype=pos_ids.dtype) + max_pos + 1
# query_pos: [N]; tile to [3, 1, N]
query_pos_3d = query_pos[None, None, :].expand(3, 1, N)   # [3, 1, N]
pos_ids_ext = torch.cat([pos_ids, query_pos_3d], dim=2)   # [3, 1, L+N]
```

> **IMPORTANT**: `get_rope_index` is called on `input_ids` (shape `[1, L]`) and parses `mm_token_type_ids` to decide which groups are image vs text. The EXTENDED `inputs_embeds_ext` is `[1, L+N, 2048]` — we do NOT need to pass the N extra tokens to `get_rope_index`; instead we manually append the positions above. The position of the queries is thus `max_pos+1 .. max_pos+N`, identical on all 3 axes, which is exactly "text-like" M-RoPE behavior per the code in `get_rope_index` line 1014: `torch.arange(text_len).expand(3,-1) + current_pos`.

### Step (e) — Deepstack unchanged

`deepstack_visual_embeds` (3 entries, one per `deepstack_visual_indexes=[5,11,17]`) and `visual_pos_masks` (shape `[1, L]` bool) are passed UNCHANGED to the text model. `_deepstack_process` indexes `hidden_states[visual_pos_masks, :]` which selects only rows 0..L-1 at the image positions — the query rows (indices L..L+N-1) are NOT selected because `visual_pos_masks` is `[1, L]` and we do NOT extend it (or we extend it with `False`s). Either way deepstack is safe.

```python
# Extend visual_pos_masks to match L+N (optional, for clarity)
# visual_pos_masks already [1, L]; but hidden_states is [1, L+N, 2048]
# _deepstack_process uses: hidden_states[visual_pos_masks, :]
# This will index into [1, L+N, 2048] using a [1, L] mask → shape mismatch!
# Must extend:
query_vmask = torch.zeros(1, N, dtype=torch.bool, device=visual_pos_masks.device)
visual_pos_masks_ext = torch.cat([visual_pos_masks, query_vmask], dim=1)  # [1, L+N] bool
```

### Step (f) — Run language_model directly, collect per-layer query hidden states

We bypass `Qwen3VLModel.forward` (which would re-inject images) and call `Qwen3VLTextModel.forward` directly with our pre-built `inputs_embeds_ext`:

```python
out = qwen_model.language_model(
    input_ids=None,             # using inputs_embeds
    inputs_embeds=inputs_embeds_ext,     # [1, L+N, 2048]
    attention_mask=attention_mask_ext,   # [1, L+N]
    position_ids=pos_ids_ext,            # [3, 1, L+N]
    visual_pos_masks=visual_pos_masks_ext,        # [1, L+N]
    deepstack_visual_embeds=deepstack_visual_embeds,  # list of 3, unchanged
    output_hidden_states=True,
    use_cache=False,
    return_dict=True,
)
# out.hidden_states: tuple of 29 tensors (embed + 28 layer outputs), each [1, L+N, 2048]
# Slice query positions from each layer:
hs = out.hidden_states   # tuple[29] of [1, L+N, 2048]
query_hidden = torch.stack(
    [hs[j+1][0, -N:, :] for j in range(28)], dim=0
)  # [28, N, 2048]
```

---

## 3. Code draft — `forward_metaquery` for `QwenVLEncoder`

This is a DRAFT method to be added to `QwenVLEncoder` in `code/igsw/dynamics/conditioning.py`. **NOT applied to the live file.**

```python
# NOTE: This is a draft code block. Do NOT apply to conditioning.py directly.
# Syntax-checked (python -m py_compile) in isolation.

@torch.no_grad()
def forward_metaquery(
    self,
    inputs: dict,
    query_embeds: torch.Tensor,   # [N, 2048] learnable parameter (on correct device/dtype)
) -> torch.Tensor:
    """
    True MetaQuery conditioning: append N learnable query tokens into the Qwen input
    sequence (AFTER image+text tokens), run them through Qwen's 28 FROZEN layers
    under the native causal mask with M-RoPE positions assigned text-like from max+1,
    then read the query positions' per-layer hidden states.

    Args:
        inputs: dict from QwenVLEncoder.build_inputs — must contain:
            input_ids [1, L], attention_mask [1, L],
            pixel_values [...], image_grid_thw [1, 3],
            mm_token_type_ids [1, L]  (from the processor in HF 5.x)
        query_embeds: [N, 2048] nn.Parameter, trainable, on device/dtype of inputs

    Returns:
        query_hidden: [28, N, 2048]  — per-layer hidden states of query positions.
            Stack as ctx_per_block for the dynamics cross-attention.
    """
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    qwen_model = self.model.model     # Qwen3VLModel
    device = query_embeds.device
    dtype = query_embeds.dtype
    N = query_embeds.shape[0]

    input_ids = inputs["input_ids"].to(device)            # [1, L]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)        # [1, L]
    pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device)  # [1, 3]
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(device)

    # ------------------------------------------------------------------
    # (a) Build inputs_embeds with vision injection via model's own helpers
    # ------------------------------------------------------------------
    # Word embed — gives placeholder embeddings at image_token_id positions
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)  # [1, L, 2048]

    # Vision tower: ViT(24 layers) + 3 deepstack mergers + final merger
    image_out = qwen_model.get_image_features(
        pixel_values, image_grid_thw, return_dict=True)
    image_embeds = torch.cat(image_out.pooler_output, dim=0).to(device, dtype)
    # image_embeds: [n_vis_merged, 2048] where n_vis_merged = H*W / 4 for this image
    deepstack_visual_embeds = image_out.deepstack_features   # list[3] of [n_vis, 2048]

    # Injection: find image_token_id positions and overwrite with vision features
    image_mask, _ = qwen_model.get_placeholder_mask(
        input_ids, inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    # inputs_embeds: [1, L, 2048], image slots now hold ViT features

    # visual_pos_masks for deepstack process (bool [1, L])
    visual_pos_masks_orig = image_mask[..., 0]   # [1, L]

    # ------------------------------------------------------------------
    # (b) Concatenate N learnable query embeddings after the sequence
    # ------------------------------------------------------------------
    q_ext = query_embeds.unsqueeze(0).to(dtype)   # [1, N, 2048]
    inputs_embeds_ext = torch.cat([inputs_embeds, q_ext], dim=1)   # [1, L+N, 2048]

    # ------------------------------------------------------------------
    # (c) Extend attention_mask: queries are valid (mask=1)
    # ------------------------------------------------------------------
    if attention_mask is not None:
        query_attn = torch.ones(1, N, dtype=attention_mask.dtype, device=device)
        attention_mask_ext = torch.cat([attention_mask, query_attn], dim=1)  # [1, L+N]
    else:
        attention_mask_ext = None

    # ------------------------------------------------------------------
    # (d) Build M-RoPE position_ids for L+N tokens
    #     get_rope_index is called on original input_ids (L tokens only);
    #     we manually append query positions as text-like (same value all 3 axes)
    #     starting from max_pos+1.
    # ------------------------------------------------------------------
    pos_ids, _rope_deltas = qwen_model.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    # pos_ids: [3, 1, L]; _rope_deltas: [1, 1]
    max_pos = int(pos_ids.max().item())
    query_pos = torch.arange(N, device=device, dtype=pos_ids.dtype) + max_pos + 1
    query_pos_3d = query_pos[None, None, :].expand(3, 1, N)   # [3, 1, N]
    pos_ids_ext = torch.cat([pos_ids, query_pos_3d], dim=2)   # [3, 1, L+N]

    # ------------------------------------------------------------------
    # (e) Extend visual_pos_masks (False for query positions)
    #     REQUIRED because _deepstack_process indexes hidden_states[visual_pos_masks]
    #     on the full [1, L+N, 2048] tensor.
    # ------------------------------------------------------------------
    query_vmask = torch.zeros(1, N, dtype=torch.bool, device=device)
    visual_pos_masks_ext = torch.cat([visual_pos_masks_orig, query_vmask], dim=1)  # [1, L+N]

    # ------------------------------------------------------------------
    # (f) Run Qwen3VLTextModel directly, collect all-layer hidden states
    # ------------------------------------------------------------------
    out = qwen_model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds_ext,         # [1, L+N, 2048]
        attention_mask=attention_mask_ext,       # [1, L+N]
        position_ids=pos_ids_ext,               # [3, 1, L+N]
        visual_pos_masks=visual_pos_masks_ext,   # [1, L+N]
        deepstack_visual_embeds=deepstack_visual_embeds,   # list[3], unchanged
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # out.hidden_states: tuple[29] of [1, L+N, 2048]
    # entry 0 = embed (before layer 0), entries 1..28 = after each decoder layer.
    hs = out.hidden_states
    # Slice the LAST N positions from each of the 28 layer outputs:
    query_hidden = torch.stack(
        [hs[j + 1][0, -N:, :] for j in range(self.num_layers)], dim=0
    )   # [28, N, 2048]

    return query_hidden
```

### 3b — How `model_full.encode` would use it under `cond_mode='metaquery'`

Below is a sketch of what would change in `InstructGSWorldModel`. The `ctx_per_block` that dynamics cross-attends to would come from `query_hidden[28,N,2048]` instead of the aggregator. The `layer_proj` projections are reused.

```python
# In InstructGSWorldModel.__init__, add when cond_mode='metaquery':
#   self.meta_query = nn.Parameter(torch.randn(n_query, 2048) * 0.02)  # [N, 2048]
# (replaces self.query [N,d] + self.layer_id_emb + self.aggregator + self.agg_norm)

def encode(self, vlm_inputs: dict, cond_mode: str = 'implicit'):
    if cond_mode == 'metaquery':
        # True MetaQuery: Qwen's own layers process the queries
        # Returns [28, N, 2048]
        query_hidden = self.encoder.forward_metaquery(
            vlm_inputs,
            query_embeds=self.meta_query.to(
                device=next(self.encoder.parameters()).device,
                dtype=torch.bfloat16))
        n_l = query_hidden.shape[0]
        # Project each layer's query hidden states to dynamics width d
        ctx_per_block = torch.stack(
            [self.layer_proj[j](query_hidden[j]) for j in range(n_l)], dim=0
        ).unsqueeze(0)   # [1, 28, N, d]
        ctx_mask = torch.ones(1, self.n_query, dtype=torch.bool,
                              device=query_hidden.device)
        # Global cond: pool query_hidden at last layer over N query positions
        # (instruction intent distilled by Qwen into the queries)
        pooled_query = query_hidden[-1].mean(0)         # [2048]
        cond_global = self.cond_proj(pooled_query)[None]  # [1, d]
        return ctx_per_block, ctx_mask, cond_global, pooled_query
    else:
        # existing 'implicit' aggregator path (unchanged)
        enc = self.encoder(vlm_inputs)
        hidden_all, valid_mask, text_mask = enc
        n_l = hidden_all.shape[0]
        ctx_full = torch.stack([self.layer_proj[j](hidden_all[j]) for j in range(n_l)], 0)
        tw = text_mask.float()[:, None]
        pooled = (hidden_all[-1] * tw).sum(0) / tw.sum().clamp_min(1e-6)
        cond_global = self.cond_proj(pooled)[None]
        if self.n_query > 0:
            agg = []
            for j in range(n_l):
                q = (self.query + self.layer_id_emb[j])[None]
                a = self.aggregator(q, ctx_full[j][None], valid_mask[None])[0]
                agg.append(self.agg_norm(a))
            ctx_per_block = torch.stack(agg, 0)[None]
            ctx_mask = torch.ones(1, self.n_query, dtype=torch.bool, device=hidden_all.device)
        else:
            ctx_per_block = ctx_full[None]
            ctx_mask = valid_mask[None]
        return ctx_per_block, ctx_mask, cond_global, pooled
```

**Design notes for the metaquery path:**
- `self.meta_query` (`nn.Parameter[N, 2048]`, initialized `randn * 0.02`) is the ONLY new trainable parameter for the query mechanism (vs. the implicit aggregator which has `self.query [N,d]` + `self.layer_id_emb [28,d]` + `CrossAttention` weights).
- `self.layer_proj` (28 × `Linear(2048, d)`) is shared with the implicit path → no new linear layers.
- `self.cond_proj` is reused to project pooled query hidden to global cond.
- `self.lang_proj` and `self.motion_head` (InfoNCE heads) are unchanged; pass `pooled_query` in place of `pooled_text`.
- The flag `cond_mode` can be set in the config so `forward` calls `encode(vlm_inputs, cond_mode=self.cond_mode)`.
- Switch between A/B by setting `self.cond_mode = 'metaquery'` or `'implicit'` at init time (or via a config field); the dynamics code (cross-attention) sees only `ctx_per_block [1, 28, N, d]` in both cases.

---

## 4. Validation checklist (for next GPU window)

```python
# ── 0. Load model and encoder
from code.igsw.dynamics.conditioning import QwenVLEncoder
import torch
enc = QwenVLEncoder().to("cuda").eval()
N = 64
meta_query = torch.nn.Parameter(torch.randn(N, 2048, device="cuda", dtype=torch.bfloat16) * 0.02)

# ── 1. Build inputs for a real episode frame
# (use the existing build_inputs + a sample image/instruction)
inputs = enc.build_inputs("Pick up the red block.", some_numpy_image)
inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
assert "mm_token_type_ids" in inputs, "processor must return mm_token_type_ids (HF 5.x)"

# ── 2. Shape assertion
with torch.no_grad():
    qh = enc.forward_metaquery(inputs, meta_query)
assert qh.shape == (28, N, 2048), f"Expected [28, {N}, 2048], got {qh.shape}"
print("shape OK:", qh.shape)

# ── 3. Gradient flows to meta_query but NOT to Qwen weights
# (with_grad block outside no_grad since forward_metaquery uses no_grad internally)
# Actually: forward_metaquery is @no_grad internally. To check meta_query grad,
# we need a thin wrapper that DOES allow grad w.r.t. query_embeds:
def forward_metaquery_grad(enc, inputs, query_embeds):
    # Same logic but without @no_grad — implement a separate version for training
    # OR: inside the @no_grad we detach hs, but query_embeds needs grad
    # RESOLUTION: forward_metaquery should NOT be @no_grad — it must allow
    # gradient to flow through qwen_model (as inputs_embeds, NOT as params).
    # Qwen weights are .requires_grad_(False), so no grad flows to them.
    # The gradient chain: loss -> query_hidden -> hs[j][0,-N:] -> inputs_embeds_ext[-N:]
    # -> q_ext -> query_embeds. This is correct ONLY if we do NOT use @no_grad.
    # See RISK §5.1.
    pass

# Temporary check: dummy loss on qh (requires removing @no_grad from forward_metaquery
# for training — keep no_grad only at inference):
loss = qh.sum()
loss.backward()
assert meta_query.grad is not None and meta_query.grad.abs().max() > 0, "meta_query has no grad"
# Verify no Qwen weight has grad:
for n, p in enc.model.named_parameters():
    assert p.grad is None, f"Qwen param {n} has gradient — should not"
print("grad check OK")

# ── 4. Memory estimate
# N=64 extra tokens at L+64 tokens (typical L ~ 256-512 for one image + instruction):
# Memory cost of N extra tokens per Qwen3VL-2B forward:
# Attention: O((L+N)^2 * layers) extra flops — tiny for N=64
# KV cache not used (use_cache=False) — no incremental overhead
# Extra activations: 28 * 64 * 2048 * 2 bytes = 28 * 64 * 2048 * 2 = ~7.3 MB extra
# (vs. existing 28 * L * 2048 * 2 bytes for L~400 → ~46 MB). Negligible. ✓
print("Memory: ~7 MB extra per forward for N=64")

# ── 5. Instruction sensitivity: swap instruction → query_hidden should change
inputs2 = enc.build_inputs("Open the drawer slowly.", some_numpy_image)
inputs2 = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs2.items()}
with torch.no_grad():
    qh2 = enc.forward_metaquery(inputs2, meta_query)
diff = (qh - qh2).abs().mean().item()
print(f"Mean |query_hidden diff| across instructions: {diff:.4f}")
assert diff > 0.001, "query_hidden does NOT change when instruction changes — check position/mask bug"
```

**Expected results:**
- Shape `[28, 64, 2048]`. ✓
- `meta_query.grad != None`, all Qwen param grads None. ✓
- `diff > 0.001` (empirically expect >> 0.01 since Qwen is instruction-sensitive). ✓

---

## 5. Risks and uncertainties

### 5.1 CRITICAL: `@no_grad` in forward_metaquery

The draft has `@torch.no_grad()`. This is correct for pure inference but **wrong for training**: we need the gradient to flow from the loss through `query_hidden[j] = hs[j+1][0, -N:, :]` back to `meta_query`. Inside `@no_grad`, no autograd graph is built, so `meta_query.grad` will be `None`.

**Resolution**: remove `@torch.no_grad()` from `forward_metaquery`. Qwen weights are `requires_grad=False`, so calling the frozen Qwen forward without `no_grad` is safe — PyTorch will not allocate grad buffers for those params. Only the graph from `inputs_embeds_ext[-N:]` (= `q_ext` = `query_embeds`) is tracked. The existing `QwenVLEncoder.forward` (for the implicit path) is also `@no_grad`, and gradients to the aggregator flow because the aggregator is called OUTSIDE `forward` in `encode`. For MetaQuery, the gradient path goes THROUGH the Qwen forward, so we cannot use `no_grad`.

**Memory implication**: removing `no_grad` means the Qwen forward will build a partial autograd graph for `query_embeds` through the 28 transformer blocks. Activation memory will scale with L+N (but still tiny for N=64). Estimate: ~8x overhead vs. no-grad for the query-token activations only, since Qwen weight activations are still detached.

### 5.2 `mm_token_type_ids` may not always be present

`get_rope_index` requires `mm_token_type_ids` (per the check at line 1120–1124 of `Qwen3VLModel.compute_3d_position_ids`). The processor in HF 5.10.2 returns it, but `QwenVLEncoder.build_inputs` should be verified to preserve it. The current `build_inputs` calls `self.processor(text=[prompt], images=images, return_tensors="pt")` — the processor does return `mm_token_type_ids` in 5.10.2 for Qwen3-VL (check by printing `list(inputs.keys())`). If missing, `get_rope_index` will raise a `ValueError` at line 1120.

**Mitigation**: after calling `build_inputs`, assert `"mm_token_type_ids" in inputs`. Alternatively, call `compute_3d_position_ids` which also handles the `can_compute_mrope=False` fallback gracefully.

### 5.3 TextModel.forward position_ids ndim

`Qwen3VLTextModel.forward` (line 802–806):
```python
if position_ids.ndim == 3 and position_ids.shape[0] == 4:
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]
else:
    text_position_ids = None
```
We pass `pos_ids_ext` of shape `[3, 1, L+N]` (ndim=3, shape[0]=3, not 4). So `text_position_ids = None`. This means the causal mask is created without explicit `position_ids`, which is correct for a simple left-to-right causal mask (no position-based masking). ✓

If we ever need to pass the 4-dim version (text + T + H + W) to be safe, prepend a text axis: the text axis should be the cumulative count of valid tokens (simple 0..L+N-1). But from the code this is only needed for hybrid cache scenarios, not for our use case.

### 5.4 `_deepstack_process` indexing on extended sequence

`_deepstack_process` (line 850): `hidden_states[visual_pos_masks, :]` where `visual_pos_masks` is `[B, L+N]`. When `B=1`, this fancy-indexes a 2D tensor `[1, L+N, 2048]` using a 2D bool mask — PyTorch treats this as `hidden_states.view(-1, 2048)[(visual_pos_masks.view(-1)), :]`. If `visual_pos_masks_ext` has the N trailing False entries, only the visual rows are selected. However, on line 856–857 the assignment `hidden_states[visual_pos_masks, :] = local_this` writes back via fancy indexing — this is fine. ✓

**Risk**: if for any reason `visual_pos_masks_ext` is NOT extended to `[1, L+N]` (e.g., the model caches the original mask), indexing will fail with a shape mismatch. The draft explicitly extends it with `cat(...zeros...)`. Verify this at GPU time.

### 5.5 HF version-specific method names

The methods `get_image_features`, `get_placeholder_mask`, `get_rope_index` are all confirmed present in `modeling_qwen3_vl.py` (transformers 5.10.2, file line 1050, 1073, 936). The `get_vision_position_ids` is a method of `Qwen3VLModel` at line 878. These names should be stable.

**Risk**: `get_image_features` on `Qwen3VLModel` (not on `Qwen3VLForConditionalGeneration`) returns a `BaseModelOutputWithDeepstackFeatures` (confirmed line 1080). The `.pooler_output` is a LIST of per-image tensors (line 1067: `image_embeds = torch.split(image_embeds, split_sizes)`). For a single image, this list has 1 element. `torch.cat(image_out.pooler_output, dim=0)` is therefore `image_embeds_list[0]`. ✓

### 5.6 Query position collision with image positions

The query M-RoPE positions are assigned `max_pos+1 .. max_pos+N`. The image token positions range over `[start_pos, start_pos + max(H//2, W//2))`. If the text comes AFTER the image (standard layout: `[BOS, vision_start, img_tokens..., vision_end, text_tokens..., EOS]`), then `max_pos` includes both the image spatial positions AND the text positions. The text tokens use monotone positions starting at `current_pos_after_image`. The actual max is the last text token position = `L_text_tokens - 1 + current_pos_after_image`. The query positions start 1 above this, so there is no collision.

**Verification**: print `pos_ids[:, 0, :]` for a sample input and check the text, image, and query ranges.

### 5.7 Causal mask and query-to-image attention

The `create_causal_mask` (masking_utils.py line 893) creates a standard lower-triangular causal mask. Appended query tokens at positions L..L+N-1 can attend to ALL preceding positions (image + text) under the causal mask because all image+text tokens have LOWER indices. The queries can also attend to each other causally (left-to-right within the N-block). This is the intended behavior for MetaQuery. ✓

Note: the query tokens are NOT a bidirectional block; they attend causally (later queries can see earlier queries). If full bidirectional attention among queries is desired, a custom `and_mask_function` could be passed to `create_causal_mask`. For simplicity, and to match the MetaQuery paper's design (which also uses a causal context), stick with the standard causal mask.

### 5.8 Whether `get_rope_index` needs the query tokens in input_ids

As described in §2(d), we call `get_rope_index(input_ids=input_ids, ...)` on the original L-token `input_ids`. The function iterates over token groups by `mm_token_type_ids` and builds positions for each group, resulting in `llm_positions` of shape `[3, L]`. We then manually extend to `[3, L+N]` by appending `max_pos+1..max_pos+N`.

**Alternative**: pass a padded `input_ids` with N dummy text tokens appended (`0` or `pad_token_id`) and extend `mm_token_type_ids` with N zeros (=text). Then `get_rope_index` will naturally assign positions `max_pos+1..max_pos+N` on all 3 axes for the N dummy text tokens. This would be equivalent and slightly simpler. However, it risks the `n_image_tokens` count check in `get_placeholder_mask` failing if the extended `input_ids` are reused elsewhere. The manual extension is safer.

---

## 6. Summary and recommended approach

**Recommended**: implement `forward_metaquery` as drafted in §3 (minus the `@no_grad` decorator for training) on a `_dev` branch copy of `conditioning.py`. The mechanism is:

1. Call `qwen_model.get_image_features(pixel_values, image_grid_thw)` to run the ViT and get vision features + deepstack features.
2. Call `qwen_model.get_input_embeddings()(input_ids)` for word embeddings.
3. Inject vision features via `get_placeholder_mask` + `masked_scatter`.
4. Concatenate `meta_query` (nn.Parameter `[N, 2048]`) as `[1, N, 2048]` after the sequence.
5. Build position IDs via `get_rope_index` on the original input, then manually extend with text-like positions from `max_pos+1`.
6. Extend `visual_pos_masks` with N False entries; pass `deepstack_visual_embeds` unchanged.
7. Call `qwen_model.language_model(inputs_embeds=..., position_ids=..., visual_pos_masks=..., deepstack_visual_embeds=..., output_hidden_states=True, use_cache=False)`.
8. Slice `out.hidden_states[j+1][0, -N:, :]` for j=0..27 → `[28, N, 2048]`.

This `[28, N, 2048]` replaces the aggregator's `ctx_per_block [28, Q, d]` after projecting each layer through `layer_proj[j]` (shared with implicit path). The trainable parameters are only `meta_query [N, 2048]` and the existing `layer_proj` weights — no new attention modules are needed.

**Blocking unknowns for GPU window:**
- Confirm `mm_token_type_ids` is returned by the processor for this specific model/processor version (print `list(inputs.keys())`).
- Confirm `out.hidden_states` has 29 entries (embed + 28 layers) for `Qwen3VLTextModel` with `output_hidden_states=True` — the `@capture_outputs` decorator may behave differently from the standard `output_hidden_states` in older HF. Print `len(out.hidden_states)`.
- Run the gradient check (remove `@no_grad` for training, confirm `meta_query.grad` non-None).
- Measure `diff` between two different instructions on the same image (instruction sensitivity of query_hidden).
- Confirm `visual_pos_masks_ext` shape handling in `_deepstack_process` does not raise.

**Estimated trainable parameters with MetaQuery (N=64):**
- `meta_query`: 64 × 2048 = 131,072 (~0.1M)
- `layer_proj` (28 × Linear(2048, d)): already present, ~= 28 × 2048 × d (d=512 → ~29M)
- `cond_proj`, `motion_enc`, `motion_head`, `lang_proj`: already present
- MetaQuery adds ~0.1M vs. implicit aggregator's ~CrossAttention(d, 8, d) ≈ 1.6M + layer_id_emb 28d ≈ ~2M
- Net: MetaQuery is LEANER, with the expressiveness coming from the 28-layer Qwen forward.

