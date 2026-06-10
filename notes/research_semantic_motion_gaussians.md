# Research — Semantic + Motion Gaussians: fixing instruction→object grounding in the 3DGS dynamics model

> Written 2026-06-08. Targets the failures diagnosed in `agent.md` §37–43: on sim test
> clips the model (a) predicts motion for STATIC background — the table "sinks/collapses"
> (false positives, §2.1), and (b) does NOT move the RED CUBE during a "pick" (false
> negatives on the instruction target, §2.2). Held-out motion-corr ≈ 0.5–0.64.
>
> Every paper below was found via web search (June 2026) and is cited with its arXiv id.
> Codebase claims are cited as `file:line`. This note is implementation-focused — it ends
> with 3 concrete experiments; **Experiment 1 is the top recommendation to build next.**

---

## 0. TL;DR — the diagnosis and the fix in one paragraph

Our "spatial grounding" (`igsw/model_full.py:168` `_control_visual` → `vis_tok/vis_film/vis_vhead`,
fed into `igsw/dynamics/model.py:105–125`) gives every control Gaussian a per-pixel Qwen image
feature at its frame-0 `control_uv`. That is necessary (the §39 overfit corr 0.95 proved it) but
**not sufficient and not instruction-grounded**: (i) the feature is the raw last-layer Qwen image
patch — it carries appearance, not "am I the object the *instruction* names"; (ii) the whole
"which controls move vs stay" decision is supervised ONLY by the regression `trajectory_loss`,
which is dominated by the static majority (the §35–37 median-collapse) so background leaks and the
target is timid; (iii) at **inference there is no GT-motion `rel`** — the only thing that tells the
net "cube vs table" is that raw patch feature, with no explicit object/static signal. The literature
on object-centric flow world models (3DFlowAction, FOCUS), dynamic-static Gaussian decomposition
(DynaSplat/DeGauss), and feed-forward semantic Gaussians (SemanticSplat, GaussianGrasper) all
converge on the same answer: **localize first, move second** — derive an explicit, instruction-
conditioned per-Gaussian *role* (mover / static, and which-object) from the frozen VLM + an
auxiliary loss, then let the dynamics move only the addressed Gaussians. We have the **perfect free
labels** for this in sim — `seg_per_g` (exact per-Gaussian entity id, `scripts/maniskill_gt.py:511`)
gives both a static/mover label and an object-identity label at zero cost. The plan below adds (1) an
explicit per-Gaussian **mover/static + semantic head** distilled from `seg_per_g` and from Qwen, and
(2) a **static gate** on the dynamics output, so background can't move and the instruction's object
must. These are the two changes that directly attack §2.1 and §2.2.

---

## 1. What the literature does (4 research questions, with transferable mechanisms)

### Q1 — Semantic / feature 3DGS, and what transfers to a FEED-FORWARD frozen-VLM setting

The classic per-scene methods are **not** directly usable (they optimize a field per scene):
- **LangSplat** (arXiv:2312.16084, CVPR'24): SAM 3-scale masks → OpenCLIP 512-d → a **scene-wise
  autoencoder** 512→3 per Gaussian. The AE is *per-scene* → the latent does not transfer → unusable
  for a generalizable model (already noted in `notes/research_B_semantic_gaussians.md`).
- **Feature-3DGS** (arXiv:2312.03203): per-Gaussian N-d feature + N-channel rasterizer, distilled
  against a 2D teacher (SAM/LSeg). The transferable idea = **differentiable feature rasterization +
  low-dim-then-upsample**.
- **Gaussian Grouping** (arXiv:2312.00732, ECCV'24): per-Gaussian **Identity Encoding** (16-d) +
  **cross-entropy to SAM masks** + an **unsupervised 3D regularization** that pulls the k-NN
  Gaussians' identities together. This CE-to-mask + 3D-NN-consistency recipe is exactly what we
  need for an auxiliary per-Gaussian object head from `seg_per_g`.
- **OpenGaussian** (arXiv:2406.02058, NeurIPS'24): single-frame **intra-mask smoothing** + **inter-
  mask contrastive** instance features (no cross-frame tracking) — directly reusable since our sim
  has one canonical frame.

The transferable, **feed-forward** pattern (no per-scene optimization) comes from the 2025 wave:
- **SemanticSplat** (arXiv:2506.09565): augments each Gaussian with a **latent semantic attribute
  f_j** predicted by a network **head**; two sub-heads branch f_j → segmentation latent f_j^S and
  language latent f_j^L; trained by **cosine-similarity distillation** of the *rasterized* latents
  against SAM and CLIP-LSeg 2D feature maps. Shared low-dim latent → both granularities. This is the
  template: *a head predicts a per-Gaussian semantic latent; distill it by cosine against a frozen
  2D teacher.*
- **GSemSplat** (arXiv:2412.16932), **SceneSplat++** (arXiv:2506.08710), **SemGS**
  (arXiv:2603.02548): all confirm the **generalizable feed-forward** paradigm beats per-scene
  optimization for fast inference and that a network can regress aligned per-Gaussian semantics.
- **GaussianGrasper** (arXiv:2403.09637): for *robot grasping*, stores a **low-dim latent per
  Gaussian** (CLIP is >500-d → too big), trained via **contrastive learning on sampled pixel pairs
  within SAM masks** (their "Efficient Feature Distillation", avoids the ~70 GB cost of raw-CLIP-per-
  Gaussian), then a 2-layer MLP decoder recovers to CLIP space *only at distillation time*. At query
  time: text → CLIP embedding → **cosine relevance heatmap → threshold (0.85) → object mask**. This
  is the cleanest "text → which-Gaussians-are-the-target" recipe and it is the inference-time signal
  we currently lack.

> **Transfer to us:** we do NOT need CLIP rasterization or per-scene AE. We need a small **per-
> Gaussian semantic head** (regressed from the Qwen patch feature we already sample) trained by an
> auxiliary loss against our **free `seg_per_g` labels** (object identity) — Gaussian-Grouping CE +
> OpenGaussian contrastive — plus a **text↔semantic cosine** so the instruction picks the target
> object's Gaussians at inference (GaussianGrasper-style). This is feed-forward and frozen-VLM-only.

### Q2 — Frozen VLM → per-region grounding (which pixels are the instruction's target)

- **Qwen2.5-VL / Qwen3-VL natively output boxes AND points** for referring expressions (Qwen2.5-VL
  tech report arXiv:2502.13923; PyImageSearch grounding tutorial). So the *cleanest, no-GT* "which
  pixels = the cube" signal is to **prompt the frozen Qwen for a point/box on the instruction's
  object** and rasterize that to a per-Gaussian relevance via `control_uv`. This requires no training
  and no SAM/Grounding-DINO.
- **Qwen3-VL-Seg** (arXiv:2605.07141): turns a (mostly) frozen Qwen3-VL into referring segmentation
  by using the **MLLM-predicted box as a "semantically grounded structural prior"** + a tiny 17M
  (0.4%) decoder. Confirms: the *box/point output is the load-bearing grounding signal*, not the raw
  attention.
- **"Few attention heads" grounding** (arXiv:2503.06287, CVPR'25): training-free attention grounding
  works only after **calibrated head-selection** (3 localization heads chosen by an attention-sum +
  spatial-entropy criterion over 1000 samples). Our own §28 test (`notes/research_E2…`) found naive
  text→image attention is an **attention sink** (peak pinned at a corner, true-vs-other corr 0.88) →
  do NOT use raw attention; if used, do the calibrated 3-head selection.
- **PnP-OVSS / the grounding literature consensus** (already in `notes/research_E_instruction_visual.md`):
  a single **pooled** global hidden vector cannot carry per-object localization; **cross-attention /
  per-patch features preserve patch-word correspondence**. We pool the instruction tokens for AdaLN
  (`igsw/model_full.py:127–129`) — fine for "what action", useless for "which object".

> **Transfer to us:** the strongest no-GT object signal is **Qwen's own referring point/box** on the
> instruction's noun, rasterized per-control. As a learned alternative that needs no extra Qwen call
> at inference, train a **text↔per-Gaussian-semantic cosine** (Q1) so the instruction embedding
> selects the object's Gaussians. Either way, the signal must be **explicit and instruction-
> conditioned**, not an implicit hope that the regression loss teaches grounding (it provably
> doesn't — §37).

### Q3 — Object-centric / which-part-moves motion (avoid moving background, ensure target moves)

This is the most on-point body of work and it unanimously does **localize-then-move**:
- **3DFlowAction** (arXiv:2506.06199, "Learning Cross-Embodiment Manipulation from 3D Flow World
  Model"): given **language + RGB-D**, predicts the future **3D flow of the manipulated object
  only**. An **auto-detect pipeline localizes the moving object first** (segments out the gripper,
  tracks point sets) → flow is predicted on **object-centric query points**, NOT the whole scene →
  background is structurally excluded. A video-diffusion world model conditions on (I₀, language).
  This is the canonical answer to §2.1+§2.2: *select the object, predict motion only there.*
- **FOCUS** (arXiv:2307.02427, object-centric world model): per-object latents with a **per-object
  mask auxiliary loss**; "objects compete to occupy their correct space … through the mask loss."
  Mask-guided **background suppression** is a recurring theme (also the segmentation-tracker
  literature).
- **Dynamic-static Gaussian decomposition**: **DynaSplat** (arXiv:2506.09836) and the dynamic-3DGS
  line estimate a **per-Gaussian dynamics mask BEFORE deformation** and "categorize all primitives
  into dynamic and static groups," then apply a **complex deformation MLP to the dynamic group and a
  near-identity one to the static group." **DeGauss** (ICCV'25) decomposes dynamic vs static
  Gaussians for distractor-free reconstruction. The motion mask is often seeded from
  YOLO/optical-flow; **we get it for free from `seg_per_g`.** This is the precedent for a **mover/
  static head + static gate**.
- **Articulated which-part-moves** (FlowBot3D arXiv:2205.04382; GAMMA arXiv:2309.16264; survey
  Wiley CGF'25): predict **dense per-point motion / a movable-part segmentation** with a head atop
  point features; "additional heads predict motion parameters along with the part segmentation."
  Confirms: **mover-classification as an auxiliary head** is standard and effective.
- **VLA cross-attention to bind language→object**: VIMA (arXiv:2210.03094), Vid2Robot
  (arXiv:2403.12943): "state encoding as queries, prompt as keys/values … learns which object to
  attend to based on the prompt." This is the precedent for an explicit **instruction↔Gaussian
  cross-attention** that gates motion to the referred object (we have language cross-attn at
  `igsw/dynamics/transformer.py:108`, but it attends to 16 *distilled/global* tokens, not to the
  per-control object identity — see §2).

### Q4 — Why OUR per-control grounding is too weak (specific to our files/layers)

Reading the code, four concrete reasons, each fixable:

1. **The conditioning that varies per-control is too weak vs the global path.** The dynamics output is
   `head(final_norm(x))` where `x` is driven by: (a) the token features (per-control), (b) **AdaLN
   from a GLOBAL `cond_global`** pooled over instruction tokens (`igsw/model_full.py:127–129`,
   applied in `igsw/dynamics/model.py:107`), and (c) **cross-attention to 16 query-aggregated tokens
   that are global** (`igsw/model_full.py:130–138`). Two of the three conditioning sources are
   **uniform across controls**, so the easy minimum is a near-uniform output — exactly the §35–37
   "every control gets the same displacement" pathology. AdaLN is known to dominate (this is why
   DiT-style models put *content* in tokens and *global style* in AdaLN); here the per-object decision
   is content, but it is starved.
2. **The per-control visual feature is appearance, not instruction-grounded identity.** `_control_visual`
   (`igsw/model_full.py:173`) samples the **raw last-layer Qwen image patch** at `control_uv`. That
   patch encodes "redness/texture here," not "this is the object the *instruction* says to pick." The
   instruction only enters globally. So the feature cannot, by itself, separate "red cube I must move"
   from "red thing on the table I must not." There is **no explicit instruction→object binding** at
   the per-control level.
3. **No static/dynamic prior; the mover decision is left entirely to the regression loss, which
   collapses to the static median.** `trajectory_loss` (`igsw/training/losses.py:58`) normalizes over
   **all visible controls** (line 76); with ~½ the controls static (mover-biased sampling still
   leaves a large static set, `train_sim.py:61`) the L1 minimum is "predict ≈0 for everyone" → table
   leaks (small but nonzero) and cube is timid. `obj_focus`/`background_static_loss` exist
   (`losses.py:71,87`) but **`obj_focus` defaults to 0 in `train_sim.py:109` and
   `background_static_loss` is NOT imported/used in `train_sim.py` at all** — so neither the foreground
   boost nor the background freeze is active in the sim regime. And both rely on `rel`, which is
   **GT-motion-derived (`train_sim.py:242–243`) → unavailable at inference.**
4. **`vis_vhead` is a per-control velocity vote, but it is unconditioned on the instruction and
   trained only by the same collapsing L1.** `vis_vhead` (`model_full.py:95`) → `v += v_logit_local`
   (`model.py:125`) is the "load-bearing" localizer, but its input is the same raw patch (point 2) and
   its only gradient is `trajectory_loss` (point 3). It can fit one clip (overfit) but **does not
   generalize the *which-Gaussian* decision** because nothing teaches it "mover vs static" or "this is
   the instruction's object" explicitly — hence held-out corr 0.5–0.64.

**One-line root cause:** *grounding is implicit and appearance-based; the move/stay and which-object
decisions have no explicit, instruction-conditioned supervision or architectural gate — so the global
AdaLN path wins and the output regresses to a near-uniform translation.*

---

## 2. Prioritized plan — 3 concrete experiments

All three **reuse the frozen Qwen3-VL and the free sim labels** (`seg_per_g`, exact `traj`). They are
additive and can be stacked; **build Exp-1 first** (it most directly fixes both §2.1 and §2.2 and
unblocks the others). Baseline to beat: held-out (heldseed/heldtask) **corr ≈ 0.65–0.76**, with
visible table-leak and timid cube (`agent.md` §39, §43).

---

### ★ Experiment 1 (TOP) — Explicit per-Gaussian MOVER/STATIC + OBJECT head with a static GATE on motion

**Idea.** Add a small per-Gaussian classifier head on top of the dynamics token (and the Qwen patch
feature) that predicts, per control: **(a) `p_dyn` = probability this Gaussian moves**, and **(b) a
semantic embedding `e_sem`** aligned to object identity. Supervise both with the **free sim labels**:
`p_dyn` against `(seg_per_g ∈ movers)` (a binary label computable from `traj` exactly as
`_moving_seg_ids` does, `scripts/maniskill_gt.py:583`), and `e_sem` against `seg_per_g` object
identity (Gaussian-Grouping CE + OpenGaussian intra/inter-mask contrastive). Then **gate the predicted
motion**: `v ← p_dyn · v`, `ω ← p_dyn · ω` (and optionally `Δscale, Δopacity` likewise), so a
Gaussian the head calls "static" **cannot move** regardless of what the regression head emits. Bind the
instruction by computing `p_dyn` as a function of **both** the Qwen patch feature **and** the
instruction–semantic match (so "pick the red cube" raises `p_dyn` on the cube's Gaussians and lowers
it on the green cube / table).

**Why it fixes §2.1 (table sinks) and §2.2 (cube doesn't move).**
- §2.1: the static gate `v ← p_dyn·v` with `p_dyn→0` on the table **structurally forbids** background
  motion — this is precisely the dynamic-static decomposition of DynaSplat/DeGauss ("estimate a per-
  Gaussian dynamics mask before deformation; static group gets a near-identity deformation"). The
  `p_dyn` supervision against `seg_per_g` movers is the explicit signal the regression loss never
  provided (Q4-3).
- §2.2: making `p_dyn` depend on the **instruction↔object semantic match** is the object-centric
  "select-the-object-then-move" of 3DFlowAction/FOCUS. The cube's Gaussians get high `p_dyn` *because
  the instruction names them*, so motion is routed to them instead of drowned by the static median.
- Generalization: `p_dyn` is a *classification* target (mover/static), which does **not** suffer the
  L1 median-collapse of regressing heavy-tailed displacements (Q4-3); classification of "does this
  move" generalizes far better than regressing "how much," and it is the established
  articulated/which-part-moves recipe (FlowBot3D, GAMMA).

**Exact files/functions to change.**
- `igsw/model_full.py`: in `__init__` (near the `vis_*` block, lines 79–103) add
  `self.mover_head = MLP(H + d → d → 1)` (sigmoid → `p_dyn`) and `self.sem_head = MLP(H + d → d → S)`
  (e.g. S=16, the Gaussian-Grouping identity dim). Inputs = the Qwen patch feat (already computed in
  `_control_visual`, `model_full.py:184–187`) concatenated with the post-transformer token. **Zero-
  init the mover_head's last layer with a +bias so `p_dyn≈1` at start** → identity warm-start (the
  model still moves everything initially, then learns to zero `p_dyn` on static). Return `p_dyn`,
  `e_sem` in the `forward` dict (extend `model_full.py:218–247`).
- `igsw/dynamics/model.py`: thread a `gate` argument into `predict_deltas` and apply `v = gate * v;
  omega = gate * omega` right after the `v_logit_local` add (after line 125, before the tanh bounds
  126–129). Keep the gate a no-op (`gate=1`) when not provided → legacy behavior preserved.
- `igsw/training/losses.py`: add `mover_bce_loss(p_dyn, mover_label, vis)` (masked BCE) and
  `semantic_id_loss(e_sem, seg_per_g, knn_idx)` = CE-to-prototype (or supervised contrastive) + the
  **3D-NN consistency** term (pull k-NN identities together, Gaussian-Grouping arXiv:2312.00732) using
  the existing `knn_idx` (`train_sim.py:244`). Optionally a `text_object_cosine` term aligning the
  instruction embedding (`out["lang_emb"]`, `model_full.py:236`) to the mover Gaussians' `e_sem`
  (GaussianGrasper-style cosine).
- `scripts/maniskill_gt.py`: nothing structural needed — `seg_per_g` is already saved
  (`maniskill_gt.py:511, 703`). Add a derived **`mover_label`** to the saved clip (per-Gaussian bool,
  `seg_per_g ∈ _moving_seg_ids(clip)`) so the trainer doesn't recompute it. (One line in the `save`
  dict at `maniskill_gt.py:699–710`.)
- `scripts/train_sim.py`: compute `mover_label` from `seg_per_g`/`traj` (or load it), pass it +
  `knn_idx` + `seg_per_g[ctrl_idx]` to the new losses, add `--w_mover --w_sem --w_text_obj`, and pass
  the gate path through `model(...)` (`train_sim.py:266`). Keep `obj_focus`-style `rel` for logging
  only. **At inference (`eval_sim_generalization.py`) the gate uses the model's own `p_dyn`** — no GT
  needed, which is the whole point.

**New data fields / features.** `mover_label[N]` (bool, free from sim) added to the clip; the per-
Gaussian `e_sem` is a model output. No new external model, no new Qwen call.

**Expected metric movement.**
- **static-leakage** (mean predicted displacement of `seg_per_g`∈static Gaussians, normalized by
  workspace radius): should drop ~5–10× toward 0 (the gate forbids it). This is a new metric worth
  logging explicitly (currently masked by corr).
- **mover-precision / recall of `p_dyn`** (vs the `seg_per_g` mover label) on held-out: target >0.9.
- **corr(GT_disp, PRED_disp)** held-out: 0.65→**0.8+** (motion concentrated on the right Gaussians).
- **top-mover ratio** (`train_sim.py:314`): 0.6–0.8 → **>0.85** (cube moves the right amount).

**Risk.** (i) The gate can **kill all motion** if `p_dyn` collapses to 0 (the inverse failure) — guard
with the +bias warm-start (`p_dyn≈1` initially) and a modest `w_mover` so it learns to zero only
confident statics; monitor mover-recall. (ii) `e_sem`/object loss may overfit the 3 sim shapes —
mitigate with the 3D-NN consistency + contrastive (encourages instance separation, not memorization)
and validate on heldtask StackCube. (iii) The instruction→`p_dyn` coupling needs the instruction to
*name* the object (sim instructions do: "Pick up the **red cube**", `maniskill_gt.py:252`) — fine in
sim; for real data the sub-task text also names objects.

---

### Experiment 2 — Instruction↔Gaussian CROSS-ATTENTION on per-control object tokens (replace/augment the global-token cross-attn)

**Idea.** Today each DiT block cross-attends to **16 global query-aggregated tokens**
(`igsw/dynamics/transformer.py:108`, ctx from `model_full.py:130–138`). Add a second, **per-control**
cross-attention where the **query is each Gaussian's token** and the **keys/values are the instruction
text tokens** (Qwen text-token hidden states, already available as `ctx_full` over `text_mask`,
`model_full.py:125–127`). This lets each Gaussian *individually* ask "does the instruction refer to
me?" — the VIMA/Vid2Robot "learn which object to attend to from the prompt" mechanism
(arXiv:2210.03094, arXiv:2403.12943), but at Gaussian granularity. Optionally bias the attention with
the Qwen patch feature so position+appearance+text jointly decide relevance.

**Why it helps §2.2 (and §2.1).** It is the missing **explicit instruction→object binding at the per-
control level** (Q4-2). A Gaussian on the named object gets strong text-conditioned features (→ moves);
a background Gaussian gets weak/irrelevant text match (→ stays). It complements Exp-1: Exp-1 gives an
explicit *label*-supervised gate; Exp-2 gives a *learned* attention route that also works when no clean
mover label exists (e.g., the eventual real-data regime).

**Exact files/functions.**
- `igsw/dynamics/transformer.py::DiTBlock`: add `self.cross_text = CrossAttention(dim, n_heads,
  ctx_dim=lang_dim)` and an always-on residual `x = x + norm_ct(self.cross_text(modulate(...),
  text_tokens, text_mask))` (mirror the existing un-gated `norm_ca` branch at line 108, with the same
  tanh-bounded AdaLN discipline from §31 to avoid the LN-backward NaN). Keep it identity-safe via the
  zero delta-head.
- `igsw/dynamics/model.py::predict_deltas`: accept `text_ctx, text_mask` and pass to each block
  (alongside the existing `ctx_per_block`).
- `igsw/model_full.py`: pass the raw **text-token** hidden states (`ctx_full[j][:, text_positions]` or
  the last-layer text tokens) as `text_ctx` (do NOT pre-distill them through the 16-query aggregator —
  the per-control attention needs the *full* text tokens to localize).

**New data fields.** None — text tokens + `text_mask` already exist (`conditioning.py:126–128`).

**Expected metric.** lang-sensitivity **Δ_null/Δ_wrong** (`eval_lang_sensitivity.py`,
`eval_sim_generalization.py`): the current ~0.2 should rise markedly (motion now genuinely re-routes
when the instruction's object changes — testable by swapping "red cube"→"green cube" on StackCube);
held-out corr +; static-leakage ↓ (text doesn't address background).

**Risk.** Adds an always-on residual → re-introduces the AdaLN/LN-backward instability risk (§30–31);
**must** reuse the tanh-bounded modulation + `norm_ct` param-free LayerNorm. Memory/compute: per-
control attention over ~tens of text tokens for M=2048 controls is cheap. Could over-attend to a sink
text token — mitigate by masking special tokens (already done via `text_mask`). **Weaker than Exp-1 if
used alone** (still no explicit static prior) — best stacked on Exp-1.

---

### Experiment 3 — Qwen REFERRING-POINT grounding (no-GT object prior) + feed-forward semantic distillation

**Idea.** Two parts, both leveraging the frozen Qwen with zero new trained model:
1. **Inference-time object prior from Qwen's native grounding.** Prompt the frozen Qwen3-VL for a
   **point or box on the instruction's object** ("Point to the red cube") — Qwen2.5/3-VL output
   absolute-coordinate points/boxes (arXiv:2502.13923, Qwen3-VL-Seg arXiv:2605.07141). Rasterize that
   point/box to a soft 2D map, sample it per-control at `control_uv` → a **per-Gaussian object-prior
   `r_obj ∈ [0,1]`**, available at inference with **no GT**. Feed `r_obj` as a token feature
   (`feature_dim` path, `tokenizer.py:51`) and/or as the `p_dyn` prior in Exp-1.
2. **Train-time semantic distillation** (SemanticSplat/GaussianGrasper template): a per-Gaussian
   semantic head distilled by **cosine** against a frozen 2D teacher feature at `control_uv`. In sim
   the teacher can be the **`seg_per_g` one-hot / a CLIP-text-of-the-object map**; for transfer to real
   data swap in CLIP-LSeg or SAM features. Use GaussianGrasper's **contrastive-within-mask**
   distillation (sample pixel pairs within an entity's mask, pull their latents together) to avoid the
   raw-feature memory cost.

**Why it helps.** Part 1 gives the cleanest *no-GT* "which pixels are the instruction's object" signal
(the §2.2 false-negative is exactly a failure to know that), and it is the only one that works at
inference without any learned grounding having generalized. Part 2 is the principled, literature-backed
way to make per-Gaussian semantics that *transfer* (the eventual real-data path), and it gives the
text↔Gaussian cosine for object selection (GaussianGrasper threshold-mask).

**Exact files/functions.**
- `igsw/dynamics/conditioning.py`: add `refer_point(text, image)` that runs Qwen generation with a
  grounding prompt and parses the point/box (the processor + `Qwen3VLForConditionalGeneration` are
  already loaded, `conditioning.py:31–44`). Cache per clip (one extra Qwen forward, like
  `encode_grounded`).
- `igsw/grounding/relevance.py`: add `point_to_relevance(point/box, uv, H, W, sigma)` → `r_obj[M]`
  (a Gaussian bump around the referred point), reusing `sample_mask_at_uv` (`relevance.py:32`).
- `scripts/train_sim.py` / `eval_sim_generalization.py`: pass `r_obj` as `feature_dim=1` input
  (set `g0.features[ctrl_idx]=r_obj`, exactly the §36 "relevance-as-input" plumbing) and/or into
  Exp-1's mover head. For Part 2, add the cosine-distillation loss against the teacher feature.

**New data fields.** `r_obj[M]` (computed online from Qwen, no storage) and (Part 2) a per-Gaussian
teacher feature for distillation (in sim derivable from `seg_per_g`; for real data from CLIP-LSeg/SAM).

**Expected metric.** With `r_obj` as a clean object prior, **mover-precision** and **corr** should jump
even without Exp-1's label (it directly tells the net the object); lang-sensitivity high by construction.
This is the **most robust to the real-data transition** (no reliance on clean `seg_per_g`).

**Risk.** (i) Qwen's referring point may be **wrong/unstable** on cluttered or novel scenes — our §28
naive-attention test failed; **the generation-based point/box (not raw attention) is more reliable**
(Qwen3-VL-Seg confirms the box is the load-bearing prior), but verify on held-out before trusting it;
fall back to Exp-1's learned `p_dyn`. (ii) One extra Qwen *generation* per clip is slower than a forward
(parse cost) — cache it. (iii) Part-2 distillation adds a teacher dependency for real data — acceptable,
it is the standard semantic-3DGS path.

---

## 3. Recommended build order and a clean acceptance gate

1. **Exp-1 first** (mover/static head + static gate, supervised by free `seg_per_g`). It is the single
   change that *structurally* fixes §2.1 (gate forbids background motion) and *explicitly* fixes §2.2
   (instruction-conditioned `p_dyn` routes motion to the named object), reuses only existing models +
   free labels, and is low-risk (identity warm-start via `p_dyn≈1`). 
2. **Stack Exp-2** (per-control instruction↔Gaussian cross-attn) to strengthen the learned binding and
   raise language-sensitivity.
3. **Add Exp-3** when moving toward real data (Qwen referring-point as the no-GT object prior +
   feed-forward semantic distillation for transfer).

**Acceptance gate (mirror §37's discipline, now with the right metrics):** on held-out sim clips —
**static-leakage → near 0**, **mover-precision/recall(p_dyn) > 0.9**, **corr → 0.8+**, **top-mover
ratio > 0.85**, and a qualitative check that the **table is frozen** and the **named cube moves** in
the rollout video (`eval_sim_generalization.py`). Only then scale / port to real data.

---

## 4. Sources

**Feed-forward / semantic 3DGS:**
- SemanticSplat — arXiv:2506.09565 — https://arxiv.org/abs/2506.09565
- GSemSplat — arXiv:2412.16932 — https://arxiv.org/abs/2412.16932
- SceneSplat++ — arXiv:2506.08710 — https://arxiv.org/abs/2506.08710
- SemGS — arXiv:2603.02548 — https://arxiv.org/abs/2603.02548
- LangSplat — arXiv:2312.16084 ; Feature-3DGS — arXiv:2312.03203
- Gaussian Grouping — arXiv:2312.00732 — https://arxiv.org/abs/2312.00732
- OpenGaussian — arXiv:2406.02058
- GaussianGrasper — arXiv:2403.09637 — https://arxiv.org/abs/2403.09637

**Frozen-VLM grounding:**
- Qwen2.5-VL Technical Report — arXiv:2502.13923 — https://arxiv.org/abs/2502.13923
- Qwen3-VL-Seg — arXiv:2605.07141 — https://arxiv.org/html/2605.07141v1
- "Few attention heads for visual grounding" — arXiv:2503.06287 (CVPR'25)

**Object-centric / which-part-moves motion & dynamic-static decomposition:**
- 3DFlowAction — arXiv:2506.06199 — https://arxiv.org/abs/2506.06199
- FOCUS (object-centric world model) — arXiv:2307.02427
- DynaSplat — arXiv:2506.09836 ; DeGauss (ICCV'25, dynamic-static decomposition)
- FlowBot3D — arXiv:2205.04382 ; GAMMA — arXiv:2309.16264 (articulated which-part-moves)
- VIMA — arXiv:2210.03094 ; Vid2Robot — arXiv:2403.12943 (prompt→object cross-attention)
- ManiGaussian — arXiv:2403.08321 (per-Gaussian delta + semantic-feature propagation, language-cond.)

**Our codebase (key file:line):**
- `igsw/model_full.py:168` `_control_visual` (Qwen patch sampling); `:127–138` global cond + 16-query
  aggregator; `:95,125` `vis_vhead`/velocity vote.
- `igsw/dynamics/model.py:105–129` per-control injection + tanh bounds (insert the **static gate** here).
- `igsw/dynamics/transformer.py:108` always-on language cross-attn (add **per-control text cross-attn**).
- `igsw/training/losses.py:58` `trajectory_loss` (median-collapse), `:71` `obj_focus`, `:87`
  `background_static_loss` (exists, **unused in sim**).
- `scripts/train_sim.py:109` `obj_focus=0` default; `:242–243` `rel` is **GT-only**; `:266` model call.
- `scripts/maniskill_gt.py:511,703` `seg_per_g` saved (the **free mover/object label**); `:583`
  `_moving_seg_ids` (the mover-label computation); `:252` instructions name the object.
- Prior notes: `notes/research_E_instruction_visual.md`, `research_E1..E4`, `research_F`, `research_G`.
