"""Qwen3-VL (Cosmos-Reason2-2B) conditioning — FROZEN, image+text, with self-grounding.

The VLM is fully frozen (no LoRA). It is given the frame-0 IMAGE + the fine-grained
sub-task instruction, and we read:
  - hidden_all  [n_layers, L, 2048]  per-layer token features (image+text), NOT a
                pooled vector -> the dynamics cross-attends to these (avoids the
                frame-0-image-domination that pooling caused).
  - text_mask   [L]  which positions are instruction text (for a text-only global cond).
  - rel_grid    [gh, gw]  a TRAINING-FREE relevance map from the instruction tokens'
                attention to the image tokens ("few attention heads" grounding,
                arXiv:2503.06287) -> sampled per-Gaussian at the anchor pixel. This
                replaces the external SAM/GroundingDINO teacher (grounding from Qwen itself).
Runs once per clip, no_grad, eager attention (needed for output_attentions).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEFAULT_QWEN_PATH = "/mnt/pfs/public/xuhaoming/model_zoo/Cosmos-Reason2-2B"


def build_qwen_inputs(processor, text: str, image):
    """Build the shared Qwen image+text processor payload without running the model."""
    from PIL import Image as PILImage
    content, images = [], None
    if image is not None:
        imgs = list(image) if isinstance(image, (list, tuple)) else [image]
        images = [PILImage.fromarray(im).convert("RGB") for im in imgs]
        content += [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=images, return_tensors="pt")


class QwenInputProcessor:
    """Processor-only cache-prep path; output is identical to QwenVLEncoder.build_inputs."""
    def __init__(self, model_path: str = DEFAULT_QWEN_PATH):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def build_inputs(self, text: str, image):
        return build_qwen_inputs(self.processor, text, image)


class QwenVLEncoder(nn.Module):
    def __init__(self, model_path: str = DEFAULT_QWEN_PATH, dtype: torch.dtype = torch.bfloat16,
                 attn_layers: int = 8, attn_impl: str | None = None):
        super().__init__()
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as _Model
        except Exception:
            from transformers import AutoModelForImageTextToText as _Model
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # attn_impl="eager" is REQUIRED for encode_grounded (output_attentions returns None under sdpa/flash);
        # default None keeps the fast sdpa/flash path for the training forward.
        kw = dict(trust_remote_code=True)
        if attn_impl is not None:
            kw["attn_implementation"] = attn_impl
        try:
            model = _Model.from_pretrained(model_path, dtype=dtype, **kw)
        except TypeError:
            model = _Model.from_pretrained(model_path, torch_dtype=dtype, **kw)
        model.eval()
        model.config.use_cache = False
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        cfg = model.config
        self.num_layers = int(cfg.text_config.num_hidden_layers)
        self.hidden_size = int(cfg.text_config.hidden_size)
        self.image_token_id = int(getattr(cfg, "image_token_id", 151655))
        self.merge = int(getattr(cfg.vision_config, "spatial_merge_size", 2))
        self.attn_layers = attn_layers
        # special tokens to exclude from the "instruction text" rows
        self._special = set(int(x) for x in [
            getattr(cfg, "image_token_id", 151655), getattr(cfg, "video_token_id", 151656),
            getattr(cfg, "vision_start_token_id", 151652), getattr(cfg, "vision_end_token_id", 151653),
            getattr(cfg.text_config, "bos_token_id", 151643), getattr(cfg.text_config, "eos_token_id", 151645),
        ])

    def build_inputs(self, text: str, image):
        """image: a single HxWx3 np.ndarray, OR a list/tuple of them (multi-view: [head, left, right]).
        Each becomes its own <image> token for Qwen3-VL. The FIRST image is the HEAD view — the per-token
        GPSToken grid is sliced from image #0 (see _grid_from_*), so the 3D head stays head-only while the
        extra wrist views only enrich the cross-attention context (hidden_all)."""
        return build_qwen_inputs(self.processor, text, image)

    @torch.no_grad()
    def encode_grounded(self, text: str, image: np.ndarray, device, dtype=torch.bfloat16, grounding=True):
        """Returns dict: hidden_all[n_layers,L,H], text_mask[L] bool, rel_grid[gh,gw] (float in
        [0,1]) or None, grid_hw (gh,gw). image is REQUIRED (frame0)."""
        inputs = self.build_inputs(text, image)
        inputs = {k: (v.to(device, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
                      else v.to(device)) for k, v in inputs.items() if torch.is_tensor(v)} | \
                 {k: v for k, v in inputs.items() if not torch.is_tensor(v)}
        out = self.model(**inputs, output_hidden_states=True,
                         output_attentions=grounding, use_cache=False)
        hs = out.hidden_states
        hidden_all = torch.stack(hs[1:1 + self.num_layers], dim=0)[:, 0]      # [n_layers,L,H]
        ids = inputs["input_ids"][0]
        L = ids.shape[0]
        image_pos = (ids == self.image_token_id)
        text_mask = torch.ones(L, dtype=torch.bool, device=device)
        for sp in self._special:
            text_mask &= (ids != sp)
        if "attention_mask" in inputs:
            text_mask &= inputs["attention_mask"][0].bool()

        rel_grid, grid_hw = None, None
        if grounding and out.attentions is not None and image_pos.any() and text_mask.any():
            try:
                img_idx = image_pos.nonzero(as_tuple=True)[0]
                txt_idx = text_mask.nonzero(as_tuple=True)[0]
                n_img = img_idx.numel()
                acc = torch.zeros(n_img, device=device, dtype=torch.float32)
                layers = out.attentions[-self.attn_layers:]
                for A in layers:                                              # A [1,heads,L,L]
                    a = A[0].float()[:, txt_idx][:, :, img_idx]               # [heads, n_txt, n_img]
                    acc += a.mean(dim=(0, 1))                                 # average heads+text
                acc /= max(1, len(layers))
                # reshape n_img -> (gh, gw) using the merged image grid
                thw = inputs.get("image_grid_thw")
                if thw is not None:
                    t, h, w = [int(x) for x in thw[0].tolist()]
                    gh, gw = h // self.merge, w // self.merge
                    if t * gh * gw == n_img:
                        g = acc.reshape(t, gh, gw)[0]
                        g = (g - g.min()) / (g.max() - g.min()).clamp_min(1e-6)
                        rel_grid, grid_hw = g, (gh, gw)
            except Exception:
                rel_grid = None
        return {"hidden_all": hidden_all, "text_mask": text_mask, "rel_grid": rel_grid, "grid_hw": grid_hw}

    @torch.no_grad()
    def forward(self, inputs: dict):
        """image+text encode (fast) -> (hidden_all[n_layers,L,H], valid_mask[L], text_mask[L]).
        text_mask excludes image + special tokens so the global cond can be pooled from the
        INSTRUCTION only (avoids the frame-0-image domination that pooling image+text caused)."""
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        hidden_all = torch.stack(hs[1:1 + self.num_layers], dim=0)[:, 0]
        ids = inputs["input_ids"][0]
        valid = inputs["attention_mask"][0].bool() if "attention_mask" in inputs \
            else torch.ones_like(ids, dtype=torch.bool)
        text_mask = valid.clone()
        for sp in self._special:
            text_mask &= (ids != sp)
        return hidden_all.detach(), valid, text_mask

    @torch.no_grad()
    def forward_batch(self, vlm_list):
        """BATCHED image+text encode: collate B per-clip processor outputs into ONE padded forward (vs B
        sequential forwards). RIGHT-pad input_ids (valid tokens at the front), concat pixel_values, stack
        image_grid_thw. The model's get_rope_index derives M-RoPE position ids from input_ids + grid_thw +
        attention_mask, so each clip's valid-token hidden states match the single-clip forward (verified).
        Returns hidden_all [n_l,B,Lmax,H], valid [B,Lmax], text_mask [B,Lmax], input_ids [B,Lmax], thw [B,3]."""
        p = next(self.model.parameters())
        dev, dtype = p.device, p.dtype
        ids_list = [v["input_ids"][0] for v in vlm_list]
        B = len(vlm_list)
        Lmax = max(int(x.shape[0]) for x in ids_list)
        pad_id = self.processor.tokenizer.pad_token_id
        pad_id = int(pad_id) if pad_id is not None else 0
        input_ids = torch.full((B, Lmax), pad_id, dtype=ids_list[0].dtype, device=dev)
        attn = torch.zeros((B, Lmax), dtype=torch.long, device=dev)
        # Qwen3-VL needs mm_token_type_ids (marks image vs text tokens) for batched M-RoPE; pad with 0 (text).
        has_mm = "mm_token_type_ids" in vlm_list[0]
        mm = torch.zeros((B, Lmax), dtype=(vlm_list[0]["mm_token_type_ids"].dtype if has_mm else torch.long),
                         device=dev) if has_mm else None
        for i, v in enumerate(vlm_list):
            ids = v["input_ids"][0]; Li = int(ids.shape[0])
            input_ids[i, :Li] = ids.to(dev)                          # RIGHT padding
            attn[i, :Li] = 1
            if has_mm:
                mm[i, :Li] = v["mm_token_type_ids"][0].to(dev)
        pixel_values = torch.cat([v["pixel_values"].to(dev, dtype) for v in vlm_list], 0)
        image_grid_thw = torch.cat([v["image_grid_thw"].to(dev) for v in vlm_list], 0)   # [B,3]
        kw = dict(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values,
                  image_grid_thw=image_grid_thw, output_hidden_states=True, use_cache=False)
        if has_mm:
            kw["mm_token_type_ids"] = mm
        out = self.model(**kw)
        hs = out.hidden_states
        hidden_all = torch.stack(hs[1:1 + self.num_layers], dim=0)    # [n_l, B, Lmax, H]
        valid = attn.bool()
        text_mask = valid.clone()
        for sp in self._special:
            text_mask &= (input_ids != sp)
        return hidden_all.detach(), valid, text_mask, input_ids, image_grid_thw

    @torch.no_grad()
    def image_grid_features(self, inputs: dict):
        """Per-control SPATIAL grounding (research_E / agent.md §37): expose the frozen-Qwen
        IMAGE patch tokens of the LAST layer in SPATIAL GRID form so the dynamics can sample a
        per-control visual feature at each control's frame-0 (u,v).

        The Qwen3-VL processor lays the image patches out as a single contiguous block of
        `input_ids == image_token_id` positions, in row-major (merged) order matching
        `image_grid_thw = [t, h, w]` with merged grid (gh,gw) = (h//merge, w//merge). We take the
        last-layer hidden state at those positions and reshape -> [gh, gw, hidden].

        Returns:
            grid  [gh, gw, hidden]  (float, last-layer image-token features), or None if the
                  inputs carry no image / the token count does not match the grid.
            (gh, gw)  the merged grid dims (or None).
        """
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            return None, None
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_last = out.hidden_states[self.num_layers][0]            # [L, H] (post-final layer)
        ids = inputs["input_ids"][0]
        image_pos = (ids == self.image_token_id)
        if not bool(image_pos.any()):
            return None, None
        img_tokens = hidden_last[image_pos]                           # [n_img_total, H] head(+wrist), row-major
        thw = inputs["image_grid_thw"]
        t, h, w = [int(x) for x in thw[0].tolist()]                   # image #0 = HEAD (3D stays head-only)
        gh, gw = h // self.merge, w // self.merge
        n_head = t * gh * gw                                          # head tokens FIRST; wrist views (if any) follow
        if img_tokens.shape[0] < n_head:
            return None, None
        # head frame (t=0); row-major reshape -> [gh, gw, H]
        grid = img_tokens[:n_head].reshape(t, gh, gw, -1)[0].contiguous()   # [gh, gw, H]
        return grid.float().detach(), (gh, gw)

    @torch.no_grad()
    def relevance_grid(self, inputs: dict):
        """INFERENCE-AVAILABLE token-placement saliency (NO GT future): cosine similarity between each
        image patch's last-layer hidden state and the POOLED instruction-text hidden -> [gh,gw] map in
        [0,1]. Patches whose visual content matches the instruction score high. Uses ONLY the fast forward
        (hidden states; NO attention rollout / NO eager attention needed -> works on the SDPA/flash path
        used in training, unlike encode_grounded which needs output_attentions=eager). This is the
        deployment-time replacement for the GT-motion mover_saliency. Returns (rel[gh,gw], (gh,gw)) or
        (None,None) if there is no image/text or the grid count mismatches."""
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            return None, None
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[self.num_layers][0]                   # [L,H] last layer
        ids = inputs["input_ids"][0]
        image_pos = (ids == self.image_token_id)
        valid = inputs["attention_mask"][0].bool() if "attention_mask" in inputs \
            else torch.ones_like(ids, dtype=torch.bool)
        text_mask = valid.clone()
        for sp in self._special:
            text_mask &= (ids != sp)
        if not bool(image_pos.any()) or not bool(text_mask.any()):
            return None, None
        img = hs[image_pos].float()                                 # [n_img,H] (causal: patches encode visual content)
        txt = hs[text_mask].float().mean(0, keepdim=True)           # [1,H] pooled instruction text
        rel = torch.nn.functional.cosine_similarity(img, txt, dim=-1)   # [n_img]
        thw = inputs["image_grid_thw"]; t, h, w = [int(x) for x in thw[0].tolist()]
        gh, gw = h // self.merge, w // self.merge
        if t * gh * gw != rel.shape[0]:
            return None, None
        rel = rel.reshape(t, gh, gw)[0]                             # [gh,gw]
        rel = (rel - rel.min()) / (rel.max() - rel.min()).clamp_min(1e-6)
        return rel.detach(), (gh, gw)

    # ------------------------------------------------------------------
    # MetaQuery conditioning (arXiv:2504.06256, research_G): append N learnable
    # query tokens INTO the frozen Qwen forward (after image+text) so Qwen's own
    # 28 layers process them under the native causal mask + M-RoPE; read the query
    # positions' per-layer hidden states. The A/B alternative to the aggregator.
    #
    # NOTE: deliberately NOT @torch.no_grad — for TRAINING the gradient must flow
    # loss -> query_hidden -> inputs_embeds_ext[-N:] -> query_embeds (the meta_query
    # parameter). Qwen params are requires_grad=False (frozen) so NO grad reaches
    # them; only meta_query (+ downstream trainable heads) receive gradient. The
    # CALLER controls the grad context (torch.no_grad() at inference). See
    # research_G §5.1 — this is the load-bearing detail.
    # ------------------------------------------------------------------
    def forward_metaquery(self, inputs: dict, query_embeds: torch.Tensor) -> torch.Tensor:
        """True MetaQuery: append N learnable query tokens into the Qwen input
        sequence and read their per-layer hidden states.

        Args:
            inputs: processor dict (from build_inputs) — must contain input_ids[1,L],
                attention_mask[1,L], pixel_values[...], image_grid_thw[1,3], and
                mm_token_type_ids[1,L] (returned by the HF 5.x Qwen3-VL processor;
                required by get_rope_index for M-RoPE).
            query_embeds: nn.Parameter [N, 2048], trainable, placed on the encoder's
                device/dtype by the caller.

        Returns:
            query_hidden [n_layers, N, 2048] — per-layer hidden states of the N query
                positions (layer j = after Qwen decoder layer j; last is post-final-norm,
                matching the aggregator path's hs[1:1+num_layers] convention).
        """
        qwen_model = self.model.model           # Qwen3VLModel (hosts visual + language_model)
        device = next(qwen_model.parameters()).device
        dtype = query_embeds.dtype
        N = int(query_embeds.shape[0])

        input_ids = inputs["input_ids"].to(device)                       # [1, L]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)                  # [1, L]
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        image_grid_thw = inputs["image_grid_thw"].to(device)            # [1, 3]
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device)

        # ---- (a) inputs_embeds with vision injection via the model's OWN helpers.
        # The vision tower + word-embed produce constants w.r.t. query_embeds (Qwen is
        # frozen), so run them under no_grad to avoid retaining 24 ViT-block activations
        # (memory optimization; does NOT change the math — image_embeds carry no grad).
        with torch.no_grad():
            inputs_embeds = qwen_model.get_input_embeddings()(input_ids)         # [1, L, 2048]
            image_out = qwen_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_out.pooler_output, dim=0).to(device=device, dtype=dtype)  # [n_vis, 2048]
            deepstack_visual_embeds = image_out.deepstack_features              # list[3] of [n_vis, 2048]
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # image slots <- ViT feats
            visual_pos_masks = image_mask[..., 0]                               # [1, L] bool

            # ---- (d) M-RoPE position_ids for the L real tokens, then extend for N queries.
            pos_ids, _rope_deltas = qwen_model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )                                                                  # pos_ids [3,1,L]
            max_pos = int(pos_ids.max().item())
            query_pos = torch.arange(N, device=device, dtype=pos_ids.dtype) + max_pos + 1
            query_pos_3d = query_pos[None, None, :].expand(3, 1, N)            # [3,1,N] text-like (same all axes)
            pos_ids_ext = torch.cat([pos_ids, query_pos_3d], dim=2)           # [3,1,L+N]

            # ---- (e) extend visual_pos_masks with N False (deepstack must NOT touch queries;
            # _deepstack_process indexes hidden_states[visual_pos_masks] on the full [1,L+N] seq).
            query_vmask = torch.zeros(1, N, dtype=torch.bool, device=device)
            visual_pos_masks_ext = torch.cat([visual_pos_masks, query_vmask], dim=1)  # [1, L+N]

            # ---- (c) extend attention_mask with N ones (queries are valid).
            if attention_mask is not None:
                query_attn = torch.ones(1, N, dtype=attention_mask.dtype, device=device)
                attention_mask_ext = torch.cat([attention_mask, query_attn], dim=1)   # [1, L+N]
            else:
                attention_mask_ext = None

        # ---- (b) concat the N learnable query embeddings AFTER the sequence. This is the
        # ONLY tensor on the autograd path to query_embeds, so it stays OUTSIDE no_grad.
        q_ext = query_embeds.unsqueeze(0).to(dtype)                            # [1, N, 2048]
        inputs_embeds_ext = torch.cat([inputs_embeds, q_ext], dim=1)          # [1, L+N, 2048]

        # ---- (f) run the text model directly (bypass Qwen3VLModel.forward which would
        # re-inject images), collecting per-layer hidden states.
        out = qwen_model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds_ext,            # [1, L+N, 2048]
            attention_mask=attention_mask_ext,          # [1, L+N]
            position_ids=pos_ids_ext,                   # [3, 1, L+N] -> ndim3/shape0==3 -> text_position_ids=None
            visual_pos_masks=visual_pos_masks_ext,      # [1, L+N]
            deepstack_visual_embeds=deepstack_visual_embeds,   # list[3], unchanged
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states                          # tuple[n_layers+1] of [1, L+N, 2048]
        query_hidden = torch.stack(
            [hs[j + 1][0, -N:, :] for j in range(self.num_layers)], dim=0
        )                                               # [n_layers, N, 2048]
        return query_hidden
