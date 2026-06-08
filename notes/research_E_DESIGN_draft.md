# Module E (DESIGN DRAFT) — Language-conditioned Visual Grounding for 3DGS Dynamics

> This is the SYSTEM-DESIGN half (mapped to our actual codebase). The literature
> survey + verified citations are produced by research agents → `research_E1..E4_*.md`
> and will be merged into the final `research_E_instruction_visual.md`.
> Status: DRAFT (citations pending verification — do NOT cite unverified work).

## 0. Corrected diagnosis (why text-only is also wrong)
Measured facts from our logs (decisive eval, stream5/6):
- Feeding Qwen's **global pooled image+text hidden state** as the dynamics condition →
  swapping the instruction moved the feature only **0.026** vs **0.334** for text-only
  (≈12.9×). ⇒ image+text pooled feature is **frame-0-image-dominated**; the model
  learned "see frame0 → predict the one future", ignoring the instruction.
- Text-only fixed image-domination (lang-divergence ↑ ~2.3×) BUT removed visual
  grounding: Qwen now gives an abstract sentence with **no pointer into the current
  scene** ("which pixels/Gaussians are the apple/basket/plate").
- Root cause is NOT "use Qwen vision = bad". It is: **a single GLOBAL hidden vector
  cannot carry per-object localization** (this is exactly what the grounding
  literature says about pooled vs cross-attention features — to be cited from E2).

**Correct target:** read Qwen as a *language-conditioned visual-grounding compiler*:
given (frame0 + instruction) → output, per task role, **which image regions / which
Gaussians** are involved + the **motion intent** + **relations** — then let the 3DGS
dynamics do the continuous 3D motion. This keeps the original goal (language as the
condition) but routes it through grounding instead of a global vector.

## 1. Three information streams (mapped to our code)
**Stream A — geometry (exists):**
`frame0/video → Pi3 (igsw/lifting) → dense 3DGS + control set`; each Gaussian already
carries its **frame-0 anchor pixel (u,v)** via `points_to_gaussians(return_uv=True)`
and `SCGSRollout(ctrl_idx)`. ⇒ any 2D map at frame0 can be sampled per-Gaussian. ✓

**Stream B — language-visual grounding (NEW = Module E):**
`instruction + frame0 → frozen Qwen3-VL (+ small trained interface) → `
- `role_masks_2d`: object / source / target / hand / background (soft maps)
- `role_relevance_3d`: per-Gaussian relevance (sample role_masks_2d at each anchor uv)
- `region_tokens [R,d]`: per-role visual-language features
- `motion_tokens [Q,d]`: task/motion-intent (pick/place/pour/push/open/close + relations)

**Stream C — dynamics (exists, to extend):** `igsw/dynamics` DiTBlock currently does
self-attn(Gaussians) + cross-attn(language) + AdaLN. Extend to:
- Gaussian **self-attention** (geometry/physics interaction) — keep.
- **Region cross-attention** — Gaussians attend to `region_tokens`.
- **Motion cross-attention** — Gaussians attend to `motion_tokens`.
- **Mask-conditioned delta gate**: `delta_i *= task_gate_i`, where
  `task_gate_i = sigmoid(MLP([gaussian_i, role_relevance_i, motion_token]))`.

## 2. Per-Gaussian token (extended)
Current: `[FourierPE(μ), q, log s, logit σ, rgb]`.
Module E adds: `[ semantic_latent, role_rel(object,source,target,hand), role_id_emb,
pooled_region_feature, motion_intent_feature ]`. (Tokenizer `feature_dim` already
supports extra per-Gaussian features — wire role-relevance + semantic latent there.)

## 3. Loss design (turns "background shouldn't move" into structure)
- `L_bg_static = Σ_i (1 - task_relevance_i) · ||Δμ_i||²`  (freeze background)
- `L_obj_motion`: trajectory loss **weighted toward** object/source/target/hand Gaussians
  (concentrate the existing direct 3D supervision on the task-relevant subset).
- `L_role_rigid`: rigidity within each role group (objects move ~rigidly).
- Keep existing direct 3D losses (pos+vel+rot via CoTracker+Pi3) + render(aux).
Rationale (key insight): even WITHOUT counterfactual instructions, concentrating
motion supervision on the **correct Gaussian subset** (known from grounding) makes the
instruction *matter* — different instructions → different relevant subsets → different
predicted motion. This is the structural substitute for counterfactual data.

## 4. Implementation phases (ablation ladder; metrics ≠ just PSNR)
- **Baseline:** current text-only (stream6/7).
- **v7-min:** external teacher (GroundingDINO/SAM2 — see E3) → role masks → per-Gaussian
  relevance feature + `L_bg_static`. No Qwen training. Fast, async grounding worker.
- **v7-attn:** training-free text→image attention from frozen Qwen (see E2) → soft
  relevance → Gaussian attention bias. Continuous, differentiable-friendly.
- **v7-metaquery (target):** frozen Qwen3-VL + learnable **SEM_QUERY/MOTION_QUERY**
  tokens (see E1/AFUN) → region/motion tokens + (via mask decoder) role masks;
  train only queries + projections + dynamics. Region/Motion cross-attn in dynamics.
**Metrics (per E):** (1) language divergence (same G0, diff instruction → rollout Δ),
(2) background motion energy (low-relevance Gaussian displacement ↓), (3) object motion
accuracy (object-relevant trajectory loss ↓), (4) role consistency (apple moves;
plate/basket/background don't get dragged).

## 4b. v7-metaquery — VERIFIED implementation recipe (from E1; AFUN 2606.02551 + MetaQueries 2504.06256)
Precedent = **AFUN** (VERIFIED): frozen Qwen3-VL (8B headline; **2B ablated → our 2B OK**) +
**MetaQuery**: 64 learnable queries (32 SEM + 32 MOTION), **only 32.21M trainable**
(queries + proj MLP + decoder); frozen {Qwen3-VL, SAM3, Sonata}. SEM→SAM3 mask;
MOTION→3D motion decoder. **We replace AFUN's motion decoder with our SC-GS dynamics.**
MetaQuery mechanism (VERIFIED): queries appended to frozen MLLM input under its **native
causal mask**; read **final-layer hidden states** at query positions; train only queries +
connector. → our `Z_sem=H[SEM_QUERY]`, `Z_motion=H[MOTION_QUERY]`.
**Qwen3-VL-2B impl gotchas (VERIFIED, load-bearing):**
1. Image features injected by **`masked_scatter` on `image_token_id=151655`** (NOT a concat
   prefix) — if hand-building `inputs_embeds` to append queries, reproduce this.
2. **M-RoPE `position_ids` = (3,B,L)** (mrope_section [24,20,20] interleaved) — must extend
   ALL 3 axes for appended query tokens (use `get_rope_index`).
3. Keep the **causal mask** (do NOT hack bidirectional into the frozen LM).
4. **No vocab/lm_head expansion** needed — queries are *read*, not generated (unlike the
   `<SEG>`-token family). Simpler than LISA/GLaMM.
5. deepstack enriches only visual positions; queries see it via attention. bf16 attention.
Seg-token alternatives (VERIFIED, if we later want emitted masks): LISA (`<SEG>`→MLP→SAM,
λ_bce 2.0/λ_dice 0.5), PixelLM (no SAM, own decoder), VideoGLaMM (→SAM2, temporal).

## 4c. v7-min — VERIFIED external teacher (from E3)
**GroundingDINO-tiny + SAM2** (both Apache-2.0; `pip install transformers`; ~1.6 GB; ~8 GB VRAM):
GroundingDINO on frame0 with phrase list `["apple","basket","plate","robot hand"]` → boxes;
**SAM2 VideoPredictor** turns boxes→masks and **propagates across the clip frames** (amortizes
cost). Per-Gaussian role relevance = sample each role mask at the Gaussian's anchor uv.
Run as an **async worker / cache** (frame0 only per clip; not in the train GPU step).
Proxy `HTTP(S)_PROXY=http://10.66.65.186:18000`, `HF_HUB_ENABLE_HF_TRANSFER=0`.
Avoid: SAM3 (gated + custom license), Qwen-native boxes (single-instance bias for dup objects).
Fallbacks: GroundingDINO thresh 0.35→0.20 → Florence-2 (MIT, polygons) / OWLv2.

## 4d. v7-attn — VERIFIED training-free attention grounding (from E2; "Few Attention Heads" 2503.06287 CVPR'25)
Frozen Qwen3-VL, NO training: only ~3 "localization heads" carry grounding.
Algorithm (per role phrase): forward with **`attn_implementation="eager"` (mandatory)** +
`output_attentions=True` → slice `attn[phrase_token_rows, image_token_cols]`, mean over phrase
tokens, reshape to the patch grid via `image_grid_thw`, mean over selected heads/last-layers,
bicubic-upsample (each token = patch16×spatial_merge2 = **32×32 px**), smooth, normalize →
soft relevance map. Per-Gaussian: token for anchor pixel (py,px) = `(py//32, px//32)`.
Caveats: eager-mode +memory; **DeepStack adds image tokens — verify `len(image_pos)` before
reshape**; maps coarse (good for prominent objects); calibrated head-selection improves cleanliness.
PnP-OVSS (cross-attn) is encoder-decoder-only → N/A for decoder-only Qwen.

## 5. Trainable-interface policy (avoid small-data LoRA damage)
- Layer 1 (default): **Qwen3-VL fully frozen**; train only MetaQuery tokens +
  Qwen→mask projection + Qwen→dynamics projection + dynamics additions. (AFUN-like, tiny.)
- Layer 2 (only if grounding weak): LoRA Qwen's top/cross-modal/projector layers ONLY,
  with a **distillation constraint** to preserve original grounding (no big 3D-traj
  gradient into the whole VLM).
- Layer 3 (strong control): action-conditioning (already prototyped, stream7) /
  counterfactual data. Final path: `language + grounded objects → motion intent →
  Gaussian dynamics` (VLA-style, see E4).

## 6. OOM / scale (user point 1): DeepSpeed ZeRO-2
Module E adds memory (grounding interface, region/motion cross-attn, possibly an
in-loop grounding model). stream7 already ~70GB/80GB. Plan:
- Integrate **DeepSpeed ZeRO-2** (shard optimizer states across the 4 GPUs → frees
  ~14GB of the fp32 Adam states) into `train_stream.py` (engine.backward/step replacing
  manual opt; keep the rollout-inside-forward pattern). Fallback levers already present:
  `checkpoint_every`, `max_gaussians`, `M`, `render_steps`.
- Run external grounding (SAM2/GroundingDINO) as an **async worker / cached** (not in the
  train step) to keep the train GPUs for dynamics — masks are frame0-only per clip.

## 7. Engineering hooks already in place (low-friction)
- `points_to_gaussians(return_uv=True)` → per-Gaussian anchor uv (for mask projection). ✓
- `SCGSRollout(ctrl_idx=...)` → control Gaussians are a known pixel subset (track/ground them). ✓
- `GaussianTokenizer(feature_dim=...)` → extra per-Gaussian features (role relevance/semantic). ✓
- `DiTBlock` cross-attn (now always-on) → add a second cross-attn for region/motion tokens. ✓
- `streaming.py` already loads frame0 + instruction (+ actions) → add an async grounding field. ✓
