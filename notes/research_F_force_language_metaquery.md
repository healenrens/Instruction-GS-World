# Research F — Forcing a (redundant-ish) language condition to be USED + MetaQuery for our frozen Qwen3-VL-2B

**Purpose.** Two deliverables for Instruct-GS-World.
- **PART 1** — how to FORCE a conditional world-model to actually *use* a language instruction that is (in single-action robot clips) statistically redundant with the static scene. The contrastive hinge we ship today (`contrastive_lang_loss`, margin 0.003) is pinned exactly at its margin → the model predicts the same future regardless of the instruction = **condition-ignoring / posterior collapse on the language condition.**
- **PART 2** — implementation-grade MetaQuery recipe for our **frozen Qwen3-VL-2B** (Cosmos-Reason2-2B, hidden 2048): append N learnable query tokens, handle `masked_scatter` image injection, M-RoPE `(3,B,L)` position extension via `get_rope_index`, native causal mask, read query hidden states, no `lm_head`/vocab change.

**Verification protocol.** Every paper checked against arXiv (API/abs/HTML) and/or primary code. Each carries **VERIFIED / UNVERIFIED**. Where a source is silent on a load-bearing detail it is tagged **[SOURCE SILENT]** and the choice we adopt is stated as *our* engineering decision, not the paper's. Where the brief's framing matches a paper exactly it is flagged. Date of verification: **2026-06-06**.

**Relation to prior notes.** This sharpens and *operationalizes* `research_E1_metaquery_segtokens.md` (which already verified MetaQuery + AFUN and gave a Qwen3-VL injection sketch). PART 2 here re-verifies and pins the M-RoPE math against the live `transformers` source; PART 1 is new (language-forcing losses). Cross-ref `research_E_DESIGN_draft.md` (region/motion tokens) and the current code in `igsw/model_full.py`, `igsw/dynamics/transformer.py`, `igsw/training/losses.py`, `scripts/train_stream.py`.

---

## 0. Current state of OUR system (what is actually wired today)

Read directly from the repo so the recommendation is faithful:

- **Conditioning** (`igsw/dynamics/conditioning.py`, `igsw/model_full.py::encode`): Qwen3-VL frozen; we read **all 28 layers'** token hidden states, project per-layer, and a **learnable query aggregator** (16 queries + per-layer id-embed + shared `CrossAttention`) distills each layer → `ctx_per_block [1,28,Q,d]`. A **global** cond vector is pooled from **instruction tokens only** (`text_mask`) → drives AdaLN.
- **Dynamics block** (`igsw/dynamics/transformer.py::DiTBlock`): self-attn (AdaLN-Zero gated) → **cross-attn to language (now ALWAYS-ON, un-gated)** → MLP (AdaLN-Zero gated). Comment in file confirms the diagnosis: *"the model was provably instruction-insensitive: contrastive loss pinned at the margin"*, so cross-attn was un-gated to force language to carry gradient.
- **Language-forcing loss today** (`igsw/training/losses.py::contrastive_lang_loss`, called in `scripts/train_stream.py`): a **single-negative hinge on step-0 control velocity**:
  ```python
  e_correct = ||v_correct0 - gt_v0||^2 ; e_wrong = ||v_wrong0 - gt_v0||^2
  L = mean_vis( relu(margin + e_correct - e_wrong) ),  margin = 0.003, weight w_lang_contrast = 0.5
  ```
  The wrong instruction is one string sampled from a 128-deque of recent instructions (`instr_buffer`), fallback `"Do nothing; keep the scene completely static."` Only **step 0**, only **velocity**, one negative, fixed margin.
- **Sampling**: boundary-biased clip sampling (`igsw/data/streaming.py`, `margin=30` frames around segment boundaries) → already emphasizes clips where the action onset is ambiguous (good: maximizes the conditional-MI signal; see §1.2/§1.8).
- **No counterfactual data**: we have fine-grained sub-task instructions but NO same-scene/different-instruction videos.

**Why the hinge saturates (the core failure).** A hinge `relu(m + e_correct − e_wrong)` has **zero gradient as soon as `e_wrong − e_correct ≥ m`**. With a *fixed, tiny* margin (0.003 in squared-metre velocity units) the model can satisfy it by making the correct prediction *infinitesimally* better than the wrong one and then **stop** — it never has to make the two predictions *meaningfully* different. Worse, in our setup the cheapest way to get `e_wrong` slightly above `e_correct` is to let the *scene* (frame-0 image, which both branches share) dominate and add a microscopic language-dependent perturbation. The loss reads as "solved" (pinned at margin) while the model is functionally instruction-agnostic. The fix is a loss whose **gradient never vanishes while the two predictions remain similar**, and that **scales with how much information the instruction adds** — that is exactly InfoNCE / conditional-MI (§1.3, §1.8).

---

# PART 1 — Forcing a redundant-ish language condition to be USED

## 1.0 The unifying principle (one equation to rule them all)

Every technique below is, at root, **maximizing the conditional mutual information between the instruction and the predicted motion, given the observation**:

> **I(motion ; language | observation)** — *"the model should attend to language exactly when it adds information about the future that cannot be deduced from the observation alone."*

This is **literally the framing in CAST** (Glossop, Chen, Bhorkar, Shah, Levine — Sergey Levine's group), **arXiv:2508.13446**, **VERIFIED**, quoting the paper:
- *"the future action distribution typically collapses given any single observation (e.g., given an observation of a chest of drawers, the only probable task for a robot is 'open the drawer'). Thus, even powerful models have little incentive to pay attention to the language command, **suffering from posterior collapse**."*
- *"we expect a model to attend to language input when it adds additional information about the action that cannot be deduced from the observation. This can be captured by the **conditional mutual information between the label and the action, conditioned on the observation I(a; ℓ | o)**."*

**This paper is our problem stated verbatim** (VLA / world-model, single-observation → narrow action posterior → language ignored). It is the strongest external justification for the whole brief and should be cited in the paper. The methods below are different *estimators / surrogates / data-augmentations* for raising `I(a; ℓ | o)`.

Two families:
1. **Make the condition causally necessary in the LOSS** without new data: InfoNCE over (instruction, motion) within a batch (§1.3, §1.8 — **our top pick**), CFG-style condition-dropout to expose an unconditional branch (§1.1), auxiliary "predict the instruction from the motion" cycle (§1.4), affordance/action auxiliaries (§1.5).
2. **Make the condition causally necessary in the DATA**: counterfactual same-scene/different-instruction pairs (CAST, §1.7). We *cannot* film these, but CAST *synthesizes* them with a VLM + the policy itself — partially applicable.

---

## 1.1 Classifier-free guidance (CFG) TRAINING — Ho & Salimans

**Paper:** *Classifier-Free Diffusion Guidance*, Jonathan Ho & Tim Salimans, **arXiv:2207.12598** (26 Jul 2022, NeurIPS 2021 workshop). **VERIFIED** (abs + ar5iv HTML). https://arxiv.org/abs/2207.12598 · https://ar5iv.labs.arxiv.org/html/2207.12598

### 1.1.1 The mechanism (exact)
- **Training:** jointly train a single network as **conditional and unconditional** by **randomly dropping the condition** to a fixed null token `∅` with probability `p_uncond`. One network, two modes.
- **Sampling (guided):** linearly extrapolate the conditional away from the unconditional:
  > **ε̃(z_t, c) = (1 + w)·ε(z_t, c) − w·ε(z_t, ∅)**   (Ho & Salimans's parameterization; equivalently `ε̃ = ε(∅) + s·(ε(c) − ε(∅))` with guidance scale `s = w+1`).
  - `w = 0` (`s=1`): ordinary conditional model. `w > 0` (`s>1`): **over-emphasize** the part of the output that *depends on the condition*, suppressing the condition-independent (unconditional) part → "fidelity↑, diversity↓."
- **Verified dropout rates in practice:** GLIDE replaces **20%** of captions with the empty token; Imagen zeros the text embedding for **10%** of instances; Ho & Salimans found `p_uncond ∈ {10%, 20%}` works and that quality is **fairly insensitive** in that range. (GLIDE/Imagen: VERIFIED via primary write-ups; see Sources.)

### 1.1.2 Why CFG forces condition-USE — the mechanism that matters for us
CFG's training half (**condition-dropout**) is what creates an explicit, well-defined **unconditional branch** `f(x, ∅)`. The guided output is `f(∅) + s·(f(c) − f(∅))`. The quantity `f(c) − f(∅)` is *literally the part of the prediction caused by the condition*; amplifying it by `s>1` at inference **mechanically forces the condition to matter** even when the model would otherwise lean on the unconditional prior. Crucially, condition-dropout during training also **prevents the model from baking the "typical" motion into the conditional pathway** — because for `p_uncond` of the steps the conditional pathway is *absent*, the network is pushed to route scene-prior / "do the obvious thing" through `f(∅)` and reserve the conditional delta for genuinely language-specific content. This directly attacks our failure mode (scene determines the future → language redundant).

### 1.1.3 Applicability to a REGRESSION world-model (not diffusion) — **the key question**
CFG was derived for the **score/noise** parameterization of diffusion, but the training trick (condition-dropout) and the guidance extrapolation are **architecture-agnostic** and apply to any conditional predictor, including our deterministic regression dynamics:

- **Training (drop-in, no new data):** with probability `p_uncond ≈ 0.1–0.2`, replace the instruction with a **null condition** (a learned `∅` token / empty string / zeroed language features) so the dynamics learns both `f_θ(G0, image, ℓ)` and `f_θ(G0, image, ∅)`. Our code already *constructs* an unconditional-ish negative ("Do nothing…") — formalize it as a **single learned `∅` embedding** appended in place of the instruction, and drop to it 15% of the time.
- **Guided rollout (inference):** extrapolate in **output (motion/velocity) space**:
  > **v̂ = v(∅) + s·( v(ℓ) − v(∅) )**, `s ∈ [1.5, 4]` (tune on held-out).
  This amplifies the language-specific motion at rollout time. (For us, guide the **per-control velocity / delta** field, the natural analogue of the score.)
- **Precedent that CFG-style dropout+guidance transfers beyond diffusion to control/RL:** *Policy Gradient Guidance Enables Test-Time Control*, Qi, Tang, Zhu, **arXiv:2510.02148** (Oct 2025), **VERIFIED** — extends CFG's "train with conditioning dropout, interpolate conditional/unconditional at test time" to **RL policy models** (non-diffusion), confirming the recipe is not diffusion-specific. [SOURCE SILENT on a closed-form guidance eq for deterministic regressors] — the extrapolation in output space above is the standard, defensible adaptation.

**Verdict for us:** CFG-training is **cheap, data-free, and complementary** to a contrastive loss. It does *not by itself* fix posterior collapse during *training* (the conditional branch can still ignore language), but it (a) gives a principled unconditional baseline, (b) gives a **free inference-time knob (`s`) to dial up instruction adherence**, and (c) the dropout regularizes the conditional pathway toward language-specific content. **Adopt the dropout + guided-rollout; pair it with InfoNCE (§1.8) which provides the training-time gradient that dropout alone lacks.**

---

## 1.2 Posterior-collapse remedies (conditional VAE / conditional generation)

These come from the cVAE literature where "the decoder ignores the latent/condition" is the canonical pathology. Even though we are **not** a VAE (no KL term, deterministic regression), the *diagnoses* and a few *mechanisms* transfer.

- **KL annealing (β-warmup).** Multiply the KL term by `β` ramped `0→1` over N steps so the model first learns to use the latent before regularization closes it off. **VERIFIED** as standard (Bowman et al. 2016; Fu et al. *cyclical annealing*). **Caveat (verified):** annealing only **delays** collapse — *"if encoder variance is not learned simultaneously, the collapsed solution is recovered"* (linear-VAE analysis, **arXiv:1911.02469**, "Don't Blame the ELBO"). **For us:** there is no KL term, so KL-annealing is **N/A directly**; the transferable idea is **schedule the *condition* in, not out** — i.e. *anneal the language-forcing weight UP* (warm-start without the contrastive term so the dynamics first fits geometry, then ramp the InfoNCE weight). See the schedule in §1.9.
- **Free bits.** Reserve a per-dimension KL floor `γ` so each latent dim is *forced* to carry ≥ γ nats: `Σ_d max(γ, KL_d)`. **VERIFIED** (Kingma et al. IAF, 2016). **For us (analogue):** impose a **floor on how different the conditional and unconditional predictions must be** — this is exactly what a *margin in InfoNCE logits* or a **"free-bits" floor on `I(motion;language|obs)`** gives. Concretely: don't just push `e_correct < e_wrong`; require the *language-attributable motion energy* `||v(ℓ) − v(∅)||²` to exceed a floor on language-relevant Gaussians (a free-bits-style anti-collapse term). This is strictly better than the current fixed hinge because the floor is on a *quantity that is zero under collapse*, so its gradient is alive precisely when the model is collapsing.
- **Mutual-information maximization (InfoVAE / the principled fix).** Replace/augment the bound so that `I(x; z)` is *maximized* rather than incidentally minimized. **The MI view is the correct lens** and routes straight to InfoNCE (§1.3): maximizing `I(motion; language | obs)` via an InfoNCE lower bound is the **non-saturating, principled successor to our hinge.**
- **BN-VAE / architectural fixes.** "A Batch-Normalized Inference Network Keeps the KL Vanishing Away" (**arXiv:2004.12585**, VERIFIED) — constrain the posterior-mean statistics so KL can't vanish. **For us:** the architectural analogue we already did is **un-gating the language cross-attention** (force a non-zero language pathway), and we can add **BN/LayerNorm on the language-conditioning vector** so the model can't trivially shrink it to zero. Keep the un-gating; it is the right move.

**Net:** the cVAE literature confirms *(a)* this is a known, named failure (posterior/condition collapse), *(b)* annealing alone is a band-aid, *(c)* **the durable fix is MI maximization (→ InfoNCE) + an anti-collapse floor (free-bits analogue)**, and *(d)* architectural guarantees of a non-zero condition path (un-gating, normalization) help.

---

## 1.3 Contrastive / alignment losses done RIGHT — InfoNCE

**Paper:** *Representation Learning with Contrastive Predictive Coding*, van den Oord, Li, Vinyals, **arXiv:1807.03748** (2018). **VERIFIED** (abs + HTML). InfoNCE is introduced here as a **variational lower bound on mutual information** (their Appendix A). https://arxiv.org/abs/1807.03748

### 1.3.1 Exact InfoNCE
For an anchor `x` with positive `x⁺` and a set of negatives `{x⁻}` (the rest of the batch), with a similarity/critic `sim(·,·)` and temperature `τ`:

> **L_NCE = − log [ exp(sim(x, x⁺)/τ) / ( exp(sim(x, x⁺)/τ) + Σ_{x⁻} exp(sim(x, x⁻)/τ) ) ]**

- It is the **cross-entropy of an N-way classification**: "which of the N candidates is the true match?" Minimizing it **maximizes a lower bound on I(x; x⁺)**: `I ≥ log(N) − L_NCE`. More negatives `N` ⇒ tighter bound ⇒ stronger pressure.
- **Temperature `τ`** controls hardness: low `τ` (≈0.07) focuses gradient on **hard negatives** (fine-grained); high `τ` (≈0.5) spreads gradient (smoother). Standard range **0.05–0.2**; **start `τ = 0.07`** (CPC/SimCLR/CLIP default region), raise if training is unstable.

### 1.3.2 Why InfoNCE does NOT saturate (vs. our fixed-margin hinge)
- **Hinge** `relu(m + e⁺ − e⁻)` → **gradient = 0** once one negative is `m` better. It only ever cares about **one** margin, against **one** negative, and it is **satisfied by an infinitesimal gap**. Pinned-at-margin ⇒ exactly this.
- **InfoNCE** is a **softmax over all negatives**; the loss is `−log p(correct)` and its gradient is `(p − 1_correct)` — **nonzero unless the model already assigns ~all probability to the correct instruction over the entire negative set.** There is **no fixed margin to "reach and stop"**; it keeps pushing the positive's score *above the log-sum-exp of all negatives*. It also uses **many negatives at once**, so the model must make its motion prediction **specifically distinguishable for the right instruction among many**, not merely ≠ one wrong instruction. This is precisely the non-saturating, multi-negative pressure our hinge lacks.

### 1.3.3 The right contrastive formulation for US — InfoNCE over (instruction, predicted-motion) within a batch
We have **no same-scene counterfactuals**, but a **batch contains different clips with different instructions and different motions**. Use **other clips' instructions as negatives** (CLIP-style symmetric InfoNCE), with the **critic = alignment between a clip's predicted motion and an instruction embedding**:

- Per batch of `B` clips, for clip `i`: positive = its own instruction `ℓ_i`; negatives = `{ℓ_j}_{j≠i}`.
- **Motion embedding** `g_i = MotionEnc(predicted per-control deltas / trajectory of clip i)` — a tiny trainable pooling head over `out["v"]` / `out["ctrl"]` (mean+attention pool → d).
- **Language embedding** `t_j = LangEnc(ℓ_j)` — pool the **frozen Qwen** instruction-token hidden states (we already compute these) → same d. (Frozen-side; only a small projector trains.)
- **Symmetric loss** (CLIP):
  > `s_ij = cos(g_i, t_j)/τ`  
  > `L = ½·CE_row(softmax_j s_ij, label=i) + ½·CE_col(softmax_i s_ij, label=j)`

This **maximizes I(predicted-motion ; instruction)** *across the batch*. Because the **predicted motion** (not GT) is the thing being aligned, gradient flows into the dynamics to make its output **instruction-discriminative**. With boundary-biased sampling (action-onset ambiguous), the positive vs. negative instructions correspond to genuinely different near-future motions → strong, clean signal. **This is the single best non-saturating replacement for the hinge.** (Full recipe + schedule in §1.8–1.9.)

[SOURCE-grounded but adapted] — InfoNCE/CLIP formulation is verified; the *application to (predicted-motion, instruction)* is our construction, justified by the CMI principle (§1.0) and directly analogous to CAST's `I(a; ℓ | o)`.

---

## 1.4 Auxiliary "predict the instruction from the motion" (cycle) loss

**Idea.** Add a head that, from the **predicted motion**, reconstructs/recovers the **instruction** — a cycle `instruction → motion → instruction`. If the motion does not encode the instruction, this head cannot succeed → backprop forces the motion to carry instruction-identifying information. This is the **inverse-dynamics / captioning-cycle** trick and it provably raises `I(motion; language)`.

**Verified precedents (cycle-consistency for text↔output alignment):**
- *Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences*, **arXiv:2506.02095** (ICCV 2025), **VERIFIED** — *"given an image and generated text, the text is mapped back to image space … the text-to-image cycle measures textual similarity between an input caption and its reconstruction; fine-grained captions ⇒ faithful reconstruction."* Demonstrates the cycle as a **supervisory/alignment signal** that rewards condition-faithfulness.
- *Leveraging Unpaired Data for Vision-Language Generative Models via Cycle Consistency*, **arXiv:2310.03734**, **VERIFIED** — image↔text cycles to improve grounding without paired data.

**Two concrete forms for us (pick the cheap one):**
1. **Discriminative cycle (recommended, cheap):** the **MotionEnc** of §1.3.3 is reused; an InfoNCE/classification head predicts *which instruction* (from the batch's instruction set) produced this motion. This is **the same as the batch-InfoNCE** above viewed from the motion side — so the symmetric CLIP loss in §1.3.3 *already implements the cycle*. (No extra module.)
2. **Generative cycle (heavier):** a tiny captioner that, from `g_i`, predicts the instruction token sequence (CE). Faithful but needs a decoder and risks the captioner shortcutting via the static scene; only add if the discriminative cycle underperforms.

**Recommendation:** the discriminative cycle = §1.3.3 InfoNCE. **Don't build a separate captioner** unless needed.

---

## 1.5 Action / affordance prediction auxiliaries

Force the language-conditioned features to be **action-predictive**, which (since actions cause the motion) makes language causally load-bearing.

- **AFUN**, **arXiv:2606.02551**, **VERIFIED** (re-checked via arXiv API 2026-06-06; title/authors/abstract confirmed — Zhaoning Wang, Yi Zhong, Jiawei Fu, Henrik I. Christensen, Jun Gao; 1 Jun 2026). From **single RGB-D + language** it predicts a **task-conditional functional mask** ("where to interact") + a **3D post-contact motion curve** ("how to interact"), with **frozen Qwen3-VL** + 64 MetaQuery tokens (32 semantic + 32 motion). The **affordance/contact auxiliary** is exactly an "action-grounding" head that makes language matter. (Full mechanism in `research_E1`.) **For us:** an **affordance head** off the SEM/MOTION query tokens (predict the contact region / first-contact Gaussian from language) is a strong auxiliary because the contact location *is* language-determined even when bulk motion is scene-determined.
- **VLA action auxiliary:** predict the **EEF action** (we already have `action_dim` plumbing + `action_embed` in `model_full.py`!) from the language-conditioned features, or **inverse-dynamics** (predict the action that explains the GT motion). Since the AgiBot data carries actions, a small **`(language, frame0) → action` CE/regression auxiliary** directly increases `I(language; action)` and is essentially free.

**Recommendation:** add an **inverse-dynamics / action-prediction auxiliary** from the conditioned features (low weight). It is the most *causally aligned* auxiliary (language→action→motion) and reuses existing action plumbing. Affordance/contact-region prediction is the §1.5 upgrade if we route through MetaQuery (PART 2).

---

## 1.6 VLA lessons (RT-2, π0, OpenVLA) — how language transfers to control

- **RT-2** (Brohan et al., **arXiv:2307.15818**, VERIFIED): represents **robot actions as text tokens** and **co-trains on web VQA + robot demos**. Co-training on internet vision-language data is *what keeps the action policy language-grounded* — the model can't drop language because the same weights must answer language-heavy web tasks. **Lesson:** **co-train an auxiliary language task** on the frozen-VLM features so the conditioning pathway stays language-sensitive. (For us the VLM is frozen, so this is less critical — but it motivates the auxiliaries in §1.4–1.5.)
- **OpenVLA** (Kim et al., **arXiv:2406.09246**, VERIFIED): 7B, Llama-2 + DINOv2/SigLIP, **discrete action tokens** autoregressively decoded; OpenX co-training. **Lesson:** action-tokenization makes actions share the *language* output space → language and action interfere/transfer in the same softmax. We don't tokenize actions, but the principle (put action and language in a shared, mutually-competing objective) ↔ our batch-InfoNCE.
- **π0** (Physical Intelligence, Black et al., **arXiv:2410.24164**, VERIFIED): VLM backbone + a **flow-matching action head** producing continuous trajectories; **heterogeneous co-training** across many robots. **Lesson directly useful to us:** π0 conditions a **continuous-output head** (flow matching) on a VLM — the **closest architecture to our continuous Gaussian-velocity head** — and it *works* with language conditioning. The flow-matching head is trained with a **conditional** velocity target; the same head supports **CFG** (drop the language prefix). This validates **CFG-on-a-continuous-policy-head** (our §1.1.3 plan) as a real, deployed pattern.
- **Language-dropout in VLAs:** the explicit, named technique for "stop ignoring language" in VLAs is **CAST's counterfactual relabeling** (§1.7), not a dropout trick. CFG-style action-conditioning dropout exists in diffusion/flow policies (π0-class) and is the transfer of §1.1.

**Net VLA lesson:** language transfers to control via **(a) co-training a language objective** (RT-2), **(b) shared action/language output space** (OpenVLA), **(c) conditional continuous head + guidance** (π0), and most pointedly **(d) counterfactual relabeling to break posterior collapse** (CAST).

---

## 1.7 CAST — counterfactual relabeling (the data-side fix), and how to use it WITHOUT filming counterfactuals

**Paper:** *CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models*, Glossop, Chen, Bhorkar, Shah, Levine, **arXiv:2508.13446** (19 Aug 2025). **VERIFIED**.

**Mechanism (verified quotes):**
- Objective: maximize **`I(a; ℓ | o)`** (§1.0).
- **Counterfactual *labels*:** *"The relabeling prompt is constructed using image and atomic label tuples corresponding to each decision point … This prompt instructs the **VLM to select a decision point, describe the counterfactual behavior** that could be executed, and indicate the atomic instruction that would correspond to this behavior."* → a VLM invents a *different plausible instruction* at a branch point in the SAME observation.
- **Counterfactual *actions*:** *"sampling the action label `a_cf ∼ π_a(a | ℓ_cf, o)` for each `(o, ℓ_cf)` … this action label acts as the counterfactual ending that branches from the original trajectory."* → the **policy itself** rolls out the alternate action for the alternate instruction, giving a synthetic (same-scene, different-instruction, different-future) pair.
- **Result:** **+27%** instruction-following success (53% vs 26% baseline) on navigation (their Fig. 3). **[SOURCE SILENT]** on exact loss form / counterfactual:real mixing ratio (paper reports it qualitatively; not pinned numerically in HTML).

**Applicability to us (we can't film counterfactuals, but we CAN synthesize them):**
- **We have a VLM (Qwen3-VL) AND fine-grained sub-task instructions** for each clip. CAST's relabeling is **directly runnable**: at an action-onset frame (our boundary-biased samples!), prompt Qwen for **a different plausible sub-task** that the scene *could* afford ("the counterfactual instruction"). That gives the *negative instruction* with a **guarantee that it is plausible-but-wrong for this future** — strictly better negatives than random batch instructions.
- The *counterfactual action/motion* (CAST's `a_cf`) is harder for us (we'd need our own model to hallucinate the alternate Gaussian motion, which is what we're trying to learn). So **adopt CAST's counterfactual-instruction generation to source HARD negatives for InfoNCE/hinge**, but **skip the counterfactual-action synthesis** initially (use it only as an InfoNCE negative, where we *don't* need a target future — we only need the model to predict a *different* future for it).

**Verdict:** CAST is the closest published treatment of our exact disease. **Use its VLM-relabeling to mine hard negative instructions** (plug into §1.8). This is a high-value, medium-effort add once the InfoNCE loss is in.

---

## 1.8 RECOMMENDED loss to force language use — exact formulation

Given our constraints (no counterfactual videos; boundary-biased sampling already emphasizes ambiguous-onset clips; deterministic regression head; frozen VLM), the best loss is a **batch InfoNCE between predicted motion and instruction**, replacing the saturating hinge, optionally hardened by CAST negatives, and paired with CFG-style condition-dropout for an inference knob.

### 1.8.1 The loss (drop-in for `contrastive_lang_loss`)
Per training micro-batch of `B` clips (accumulate across `grad_accum` if `B=1` per step — see 1.8.4):

```
# trainable heads (tiny):
#   MotionEnc: (per-control deltas v[K,M,3] or ctrl traj) --maskpool--> g_i in R^d, L2-normalized
#   LangProj : (frozen Qwen instruction-token hidden states) --meanpool+Linear--> t_i in R^d, L2-normalized
g_i = normalize(MotionEnc(motion_i))            # uses out["v"]/out["ctrl"], visibility-masked pooling
t_i = normalize(LangProj(qwen_instr_hidden_i))  # frozen side; only Linear trains
S   = (g @ t.T) / tau                            # [B,B] logits, tau=0.07
L_infonce = 0.5*CE(S,  arange(B)) + 0.5*CE(S.T, arange(B))   # symmetric CLIP loss
```

- **Why predicted motion (not GT):** gradient must enter the **dynamics** to make *its output* instruction-specific. (Aligning GT-motion to instruction would train only the projectors.)
- **Visibility mask** the motion pooling (reuse `vis_traj`).
- **`τ = 0.07`** start; **anneal up to 0.1** if unstable.
- **Negatives:** the `B−1` other clips' instructions (free). **Hard-negative upgrade (CAST):** additionally include, for each clip, a **Qwen-generated counterfactual instruction** for the same scene (§1.7) as an extra column → forces the motion to differ for a *plausible-but-wrong* instruction, not just an unrelated one.

### 1.8.2 Keep a (small) anti-collapse FLOOR (free-bits analogue)
Add a light term that is **zero only when the model collapses**, so its gradient is alive exactly when needed:

```
# language-attributable motion energy on task-relevant (high-relevance) Gaussians:
delta = v_correct0 - v_uncond0                     # v_uncond0 = motion under the learned ∅ condition
L_floor = relu( c0 - mean_relweighted( ||delta||^2 ) )   # push language to MOVE the prediction by >= c0
```
This replaces the brittle fixed-margin hinge with a **floor on how much language must change the prediction**, restricted to language-relevant Gaussians (use our `rel_control`). `c0` ~ a small fraction of typical per-step displacement (e.g. set so `~1–2 cm` of velocity difference is required on task Gaussians). Unlike the old hinge, the quantity inside is **0 under collapse** ⇒ never silently "solved."

### 1.8.3 CFG condition-dropout (the `∅` branch + inference knob)
- Replace the instruction with a **single learned `∅` embedding** with prob **`p_uncond = 0.15`** during training (formalizes today's "Do nothing…" fallback). This *defines* `v_uncond0` used in §1.8.2 and lets the conditional pathway specialize to language.
- **Guided rollout:** `v̂ = v(∅) + s·(v(ℓ) − v(∅))`, `s∈[1.5,4]`. Free, and gives a dial to *prove* (and tune) language dependence at eval.

### 1.8.4 Batch-size note (important for our streaming trainer)
Our trainer runs **B=1 clip/step** with grad-accum. InfoNCE needs **multiple instructions simultaneously**. Two fixes:
- **(preferred)** raise the **per-step micro-batch to `B≥8` clips** (the streaming JIT pipeline can yield several clips; cross-attn ctx is small) so negatives are in-graph; or
- **(fallback)** maintain a **momentum queue of recent `(g, t)`** (MoCo-style, queue ≥256) and contrast against the queue. The current `instr_buffer` deque (128) is the seed of this; extend it to also store **motion embeddings** `g`. Use a **stop-grad/EMA** copy for queued entries.

### 1.8.5 Why this beats the current hinge (summary)
| Property | current hinge (margin 0.003) | recommended InfoNCE (+floor) |
|---|---|---|
| Negatives | 1 random | B−1 (+CAST hard neg) |
| Saturates? | **yes, at fixed margin** | **no** (softmax over all negatives) |
| Quantity that is 0 under collapse drives gradient? | no | **yes** (CMI / free-bits floor) |
| Estimates I(motion;lang|obs)? | no | **yes** (InfoNCE = MI lower bound) |
| Inference knob for adherence? | no | **yes** (CFG scale `s`) |

---

## 1.9 Weight schedule (concrete)

Warm-start geometry first (cVAE annealing lesson: introduce the forcing term *after* the model can fit motion), then ramp:

- **Steps 0–2k (warm-up):** `w_infonce = 0`, `w_floor = 0`, `p_uncond = 0` (or 0.05). Train trajectory/rotation/render only — let the dynamics learn to move at all. Keep cross-attn un-gated (already done).
- **Steps 2k–8k (ramp):** linearly ramp `w_infonce: 0 → 0.5` and `w_floor: 0 → 0.05`; turn on `p_uncond = 0.15`. (Mirror today's `w_lang_contrast=0.5` magnitude for `w_infonce`.)
- **Steps >8k (full):** `w_infonce = 0.5`, `w_floor = 0.05`, `τ = 0.07`, CAST hard negatives on. Monitor **language-sensitivity metric** below.
- **Retire the hinge** (`w_lang_contrast → 0`) once InfoNCE is on; they target the same thing and the hinge only adds a saturating, weaker signal.

**Total loss:** `L = w_pos·L_pos + w_vel·L_vel + w_rot·L_rot + w_render·L_render + w_bg·L_bgstatic + w_reg·L_reg + w_infonce·L_infonce + w_floor·L_floor (+ w_act·L_action_aux)`.

**Diagnostic to watch (define & log):** the **language-sensitivity gap** `Δ = mean_relweighted ||v(ℓ) − v(∅)||` and the **counterfactual gap** `||v(ℓ) − v(ℓ_cf)||` (using a CAST/random negative). If `Δ→0`, language is still ignored regardless of what the loss reads. Today's "pinned at margin" symptom should be replaced by **`Δ` rising and the InfoNCE batch-accuracy climbing toward 1**.

---

# PART 2 — MetaQuery for our FROZEN Qwen3-VL-2B (implementation-grade)

## 2.0 Papers (re-verified 2026-06-06)
- **MetaQueries** — *Transfer between Modalities with MetaQueries*, Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, Saining Xie. **arXiv:2504.06256** (8 Apr 2025). **VERIFIED** (abs/HTML/project page/**official code facebookresearch/metaquery**). https://arxiv.org/abs/2504.06256 · https://xichenpan.com/metaquery/ · https://github.com/facebookresearch/metaquery
- **AFUN** — **arXiv:2606.02551** (1 Jun 2026). **VERIFIED** (arXiv API re-confirmed today). MetaQuery applied to affordance with **frozen Qwen3-VL** (8B main, **2B ablated**) + SAM3 + Sonata; 64 queries (32 sem + 32 motion); last-hidden-states read out; ~32.21M trainable. https://arxiv.org/abs/2606.02551

**Confirmed core facts (all TRUE, from MetaQueries primary):** (1) learnable queries `Q∈R^{N×D}`, `D = MLLM hidden`; main `N=256` (ablated 1…1024); (2) **appended to the MLLM input sequence**, whole thing run through the **frozen** MLLM; (3) **causal mask kept for the entire sequence** — *"we continue to use causal masking for the entire sequence rather than specifically enabling full attention for Q"*; (4) read **output hidden states at query positions** = conditions `C`; (5) trainable = {queries, connector (Enc-Proj: 24-layer bidirectional transformer in MLLM dim → projection), downstream decoder}; **MLLM frozen**, **no lm_head/vocab change**; loss = plain diffusion denoising on 25M image-caption pairs.

## 2.1 Verified Qwen3-VL-2B facts that drive the code (from the live `transformers` source + config)
Re-checked against `huggingface/transformers` `main` `modeling_qwen3_vl.py` and `Qwen/Qwen3-VL-2B-Instruct/config.json`:
- `hidden_size = 2048` ⇒ **query dim D = 2048**.
- `image_token_id = 151655`, `video_token_id = 151656`, `vision_start = 151652`, `vision_end = 151653`.
- `deepstack_visual_indexes = [5, 11, 17]` (ViT features injected into LLM layers 5/11/17 at **visual** positions only).
- M-RoPE: `mrope_section = [24,20,20]`, interleaved, `rope_theta = 5e6`.
- **`forward` accepts `inputs_embeds`** (`if (input_ids is None) ^ (inputs_embeds is not None): raise …`) ⇒ we can build our own embedding sequence.
- **Image merge = `inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)`** where `image_mask` comes from `get_placeholder_mask(input_ids, inputs_embeds, image_features)` ≈ `(input_ids == image_token_id)`. **Not a concat prefix.**

### 2.1.1 `get_rope_index` — EXACT behavior (verified from source, the load-bearing gotcha)
Returns **`position_ids` of shape `(3, batch, seq_len)`** (axis-0 = temporal/height/width) + `mrope_position_deltas`.
- **Text (non-vision) tokens:** all **3 axes get the SAME incrementing value**:
  ```python
  llm_pos_ids_list.append(torch.arange(text_len).view(1,-1).expand(3,-1) + current_pos)
  ```
- **After a vision block** the running counter advances by the merged spatial extent:
  ```python
  current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
  ```
  i.e. the next block continues from where the previous ended (effectively `max position so far + 1`).
- **Delta:** `mrope_position_deltas = llm_positions.max() + 1 − len(current_input_ids)` (offset for cached incremental decoding).

**Consequence for appended queries:** our query tokens are **text-like** (not vision) ⇒ assign them the **same monotonic index on all 3 axes**, starting at `position_ids.max()+1`, contiguous. This matches exactly how `get_rope_index` would continue the sequence with text. (MetaQueries/AFUN are **[SOURCE SILENT]** on query positions; this is the faithful, defensible choice and is *consistent with the model's own text-continuation rule* — not an arbitrary hack.)

## 2.2 EXACT PyTorch recipe (subclass-free where possible)

```python
import torch, torch.nn as nn
from transformers import AutoProcessor
try:    from transformers import Qwen3VLForConditionalGeneration as M
except: from transformers import AutoModelForImageTextToText as M

name = "/mnt/pfs/public/xuhaoming/model_zoo/Cosmos-Reason2-2B"
proc  = AutoProcessor.from_pretrained(name, trust_remote_code=True)
model = M.from_pretrained(name, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
for p in model.parameters(): p.requires_grad_(False)              # FREEZE everything

D   = model.config.text_config.hidden_size                        # 2048
IMG = model.config.image_token_id                                 # 151655
N_SEM, N_MOT = 32, 32                                             # AFUN counts (closer precedent than 256)
sem_q = nn.Parameter(torch.randn(N_SEM, D)*0.02)                  # trainable
mot_q = nn.Parameter(torch.randn(N_MOT, D)*0.02)                  # trainable
Nq = N_SEM + N_MOT

# ---- 1) normal multimodal batch (frame0 image + instruction) ----
msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":instruction}]}]
prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = proc(text=[prompt], images=[frame0_pil], return_tensors="pt").to(model.device)
# enc: input_ids[B,L], attention_mask[B,L], pixel_values, image_grid_thw

# ---- 2) build inputs_embeds the way the model does, THEN append queries ----
inputs_embeds = model.get_input_embeddings()(enc.input_ids)                       # [B,L,D]
image_embeds  = model.get_image_features(enc.pixel_values, enc.image_grid_thw)    # frozen ViT (+deepstack)
img_mask = (enc.input_ids == IMG).unsqueeze(-1)                                   # [B,L,1]
inputs_embeds = inputs_embeds.masked_scatter(img_mask.to(inputs_embeds.device),
                                             image_embeds.to(inputs_embeds.dtype))
B = inputs_embeds.size(0)
q = torch.cat([sem_q, mot_q], 0)[None].expand(B,-1,-1).to(inputs_embeds.dtype)    # [B,Nq,D]
inputs_embeds = torch.cat([inputs_embeds, q], dim=1)                              # [B,L+Nq,D]
attn = torch.cat([enc.attention_mask, torch.ones(B, Nq, device=q.device, dtype=enc.attention_mask.dtype)], 1)

# ---- 3) position_ids: compute for real tokens, then continue text-like for queries on ALL 3 AXES ----
pos, _ = model.model.get_rope_index(enc.input_ids, enc.image_grid_thw, attention_mask=enc.attention_mask)  # [3,B,L]
start  = pos.amax(dim=-1, keepdim=True) + 1                                       # [3,B,1]
qpos   = start + torch.arange(Nq, device=pos.device).view(1,1,Nq)                 # [3,B,Nq] (same idx all axes)
position_ids = torch.cat([pos, qpos.expand(3,B,Nq)], dim=-1)                      # [3,B,L+Nq]

# ---- 4) forward the INNER LM stack (no lm_head), read query hidden states ----
out = model.model(inputs_embeds=inputs_embeds, attention_mask=attn,
                  position_ids=position_ids, output_hidden_states=True, use_cache=False)
H = out.last_hidden_state                                                         # [B,L+Nq,D]
sem_h = H[:, -Nq:-N_MOT, :]        # [B,N_SEM,D]  -> grounding / region tokens
mot_h = H[:, -N_MOT:,   :]         # [B,N_MOT,D]  -> motion-intent tokens -> dynamics cross-attn
# (all-layer read-out: stack out.hidden_states[l][:, -Nq:, :] for per-layer query features)
```

**Notes baked into the code above:** (a) image features via `masked_scatter`, **not** prefix; (b) `position_ids` extended on **all 3 M-RoPE axes** with **text-like contiguous indices** from `max+1`; (c) `attention_mask` extended by ones; (d) **native causal mask kept** (we pass nothing special → the model builds its own causal mask; queries at the end see all image+text, per MetaQueries §2.0(3)); (e) read **inner `model.model`** to avoid the `lm_head`; (f) `use_cache=False`; (g) **no vocab/embedding expansion** — queries live only in embedding space and are **read, not generated**.

## 2.3 Where you must subclass / patch (and where you don't)

- **No subclass needed for the forward** if you (i) build `inputs_embeds` yourself and (ii) precompute `position_ids` yourself, then call `model.model(...)`. This is the clean path and avoids touching internals. **[Gotcha]** If you instead pass `input_ids` (not `inputs_embeds`) **and** appended-query placeholder tokens, you'd have to extend the tokenizer/embedding and patch `get_rope_index` — **avoid that**; build embeddings directly.
- **`get_image_features` + deepstack:** `get_image_features` returns the ViT features that deepstack later injects at layers [5,11,17]. **Route them through the normal scatter** (step 2) so image tokens get enriched; our appended queries are **non-visual** ⇒ deepstack does **not** touch them directly — they receive deepstack-enriched info **through attention** (intended). **Do not drop the deepstack embeds.** **[Gotcha]** Some `transformers` versions return image_embeds already including deepstack channels; verify the `masked_scatter` shape matches the number of image placeholder tokens (`(input_ids==IMG).sum()`), else you'll silently misalign — assert it.
- **`get_rope_index` signature drift:** the kwarg names (`image_grid_thw`, `video_grid_thw`, `attention_mask`) have shifted across `transformers` minor versions. **[Gotcha]** Pin the version (memory: `transformers 5.10.2`) and unit-test that `pos.shape == (3, B, L)` and `position_ids[:, :, L:]` is contiguous from `max+1`. Our `conditioning.py` already uses `image_grid_thw`/`spatial_merge_size`; reuse those accessors.
- **Attention mask under SDPA/flash:** Qwen3-VL builds the causal mask internally when none is passed. **[Gotcha]** If you pass a 2D `attention_mask` it is treated as the **padding** mask and combined with causal — correct for us. **Do not** pass a 4D bidirectional mask hoping to make queries bidirectional; MetaQueries explicitly keeps **causal** and it works (§2.0(3)). bf16 mandatory (our memory: fp32 SDPA OOMs at 16k tokens).
- **Trainable set (mirror AFUN ~32M):** `{sem_q, mot_q}` + a **2-layer MLP per branch** `[2048→d]` + the downstream head (our **dynamics**, which *is* the motion decoder — we skip AFUN's separate Bézier head). **Frozen:** all of Qwen3-VL.

## 2.4 MetaQuery vs. our current implicit cross-attention — the real decision

**What we have now** (`model_full.py::encode`): a **learnable query aggregator** that *pools a single frozen forward pass'* per-layer features (28 layers) into special tokens, then the dynamics cross-attends to those. The queries here are **read-side poolers** over a forward pass that the queries **did not participate in**.

**What MetaQuery adds:** the queries are **inputs the frozen transformer computes over** — they can **actively gather task-conditioned info via the LLM's own attention + FFN** (28 layers of computation *on the query*), specializing per-role/per-intent. This is a **strictly richer** conditioning signal than post-hoc pooling, and is the exact thing AFUN found necessary for affordance.

**However** — and this is the crucial honest caveat — **MetaQuery does NOT, by itself, fix the language-ignoring problem.** MetaQuery changes *how* we extract conditioning; the posterior-collapse failure is about *whether the downstream model is forced to use it*. If the future is scene-determined, even perfect MetaQuery motion tokens will be ignored unless a **language-forcing loss (PART 1)** makes them load-bearing. **PART 1 is the fix; PART 2 is an upgrade to the conditioning interface.** Do them in that order.

---

## 2.5 RANKED RECOMMENDATION

### (a) Which language-forcing loss to add NOW — **batch InfoNCE(predicted-motion, instruction) + free-bits floor + CFG dropout**, replacing the saturating hinge.
1. **Replace** `contrastive_lang_loss` (saturating hinge) with **symmetric batch InfoNCE** between a pooled **predicted-motion** embedding and the **frozen-Qwen instruction** embedding (§1.8.1), `τ=0.07`, negatives = other clips in batch. *Non-saturating, estimates `I(motion;lang|obs)`.*
2. **Add the free-bits-style floor** on language-attributable motion energy `||v(ℓ)−v(∅)||²` on task-relevant Gaussians (§1.8.2). *Gives gradient exactly when collapsing — fixes the "pinned at margin" pathology directly.*
3. **Add CFG condition-dropout** (`p_uncond=0.15`, learned `∅`) + **guided rollout** `v(∅)+s(v(ℓ)−v(∅))`, `s∈[1.5,4]` (§1.8.3). *Defines the `∅` branch the floor needs; free inference knob to prove/tune adherence.*
4. **Schedule:** geometry warm-up 0–2k, ramp InfoNCE/floor 2k–8k, full >8k; retire the hinge (§1.9).
5. **(medium-effort, high-value) CAST hard negatives:** use Qwen to generate a **plausible counterfactual sub-task** for the same scene as an extra InfoNCE negative (§1.7).
6. **(cheap) inverse-dynamics auxiliary:** predict the EEF action from conditioned features (reuse existing `action_dim` plumbing) (§1.5).

This requires **B≥8 clips/step** (or a MoCo queue extending the existing `instr_buffer`) — the one nontrivial plumbing change (§1.8.4).

### (b) Is MetaQuery worth implementing vs. the current implicit cross-attn?
**Yes, but second.** MetaQuery is the **right long-term conditioning interface** (it is *exactly* AFUN, our nearest precedent, and gives the LLM 28 layers of computation *on* the queries instead of post-hoc pooling). It is **~32M trainable, frozen-VLM, no vocab change**, and §2.2 is a faithful, ready recipe. **But it will not fix language-ignoring on its own** — that is a *loss/data* problem (PART 1), not a *feature-extraction* problem. Spend the first cycle on PART 1.

### (c) The experiment that decides MetaQuery vs. current cross-attn
With PART-1 losses **fixed and on**, run an **A/B on the conditioning interface only**, holding dynamics + losses + data identical:
- **Arm A (current):** query-aggregator pooling of the frozen forward pass (today's `encode`).
- **Arm B (MetaQuery):** appended 32+32 queries computed *through* frozen Qwen (§2.2); dynamics cross-attends to `mot_h` (+ `sem_h` for grounding).

**Decision metrics** (both measured at equal trainable budget / steps):
1. **Language-sensitivity gap** `Δ = mean_relweighted ||v(ℓ) − v(∅)||` and **counterfactual gap** `||v(ℓ) − v(ℓ_cf)||` — *the direct test of "does it use language."* MetaQuery wins iff it **raises these** at equal motion accuracy.
2. **Instruction-conditioned rollout accuracy** on **held-out boundary clips** (where onset is ambiguous and language is maximally informative): position/PSNR under the *correct* vs *swapped* instruction. MetaQuery wins iff the **correct−swapped gap** is larger.
3. **InfoNCE batch retrieval accuracy** (motion→instruction) as a cheap proxy.

If Arm B does not beat Arm A on (1)+(2) by a clear margin, **keep the simpler current pooling** — the implicit cross-attn is cheaper and already wired. (Prediction: MetaQuery helps most on the **grounding/where** signal — the `sem_h`/affordance side — and less on bulk motion; if so, adopt MetaQuery **only for the semantic/region tokens** and keep pooling for the rest, per `research_E1` §5.6.)

---

## 3. Sources (all opened during verification, 2026-06-06)

**PART 1**
- Classifier-Free Diffusion Guidance — Ho & Salimans, **arXiv:2207.12598** · https://arxiv.org/abs/2207.12598 · https://ar5iv.labs.arxiv.org/html/2207.12598
- CFG dropout rates (GLIDE 20%, Imagen 10%): eugeneyan.com/writing/text-to-image · sander.ai/2022/05/26/guidance.html (secondary, corroborated)
- CPC / InfoNCE — van den Oord, Li, Vinyals, **arXiv:1807.03748** · https://arxiv.org/abs/1807.03748
- Posterior collapse: "Don't Blame the ELBO!" **arXiv:1911.02469**; BN-VAE **arXiv:2004.12585**; cyclical KL-annealing (Fu et al.) — emergentmind.com/topics/posterior-collapse (overview, corroborated)
- CAST — Glossop, Chen, Bhorkar, Shah, Levine, **arXiv:2508.13446** · https://arxiv.org/abs/2508.13446 · https://arxiv.org/html/2508.13446v1
- CFG-in-RL — Qi, Tang, Zhu, *Policy Gradient Guidance Enables Test Time Control*, **arXiv:2510.02148** · https://arxiv.org/pdf/2510.02148
- Cycle consistency — *Cycle Consistency as Reward* (ICCV'25) **arXiv:2506.02095**; *Unpaired VLM via Cycle Consistency* **arXiv:2310.03734**
- VLAs — RT-2 **arXiv:2307.15818**; OpenVLA **arXiv:2406.09246**; π0 **arXiv:2410.24164**

**PART 2**
- MetaQueries — Pan et al., **arXiv:2504.06256** · https://arxiv.org/abs/2504.06256 · https://xichenpan.com/metaquery/ · https://github.com/facebookresearch/metaquery
- AFUN — **arXiv:2606.02551** · https://arxiv.org/abs/2606.02551 (re-verified via arXiv API)
- Qwen3-VL — modeling source: github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py (`get_rope_index`, `masked_scatter`, `inputs_embeds` verified); config `Qwen/Qwen3-VL-2B-Instruct`; tech report **arXiv:2511.21631**
- Local code reviewed: `igsw/model_full.py`, `igsw/dynamics/{conditioning,transformer}.py`, `igsw/training/losses.py`, `scripts/train_stream.py`, `igsw/data/streaming.py`

## 4. Honest gaps / UNVERIFIED
- **CAST exact loss form & counterfactual:real mixing ratio:** **[SOURCE SILENT]** — paper states the CMI objective and the VLM-relabel + policy-rollout procedure and reports +27% (Fig.3), but does not pin a closed-form loss or mixing ratio in the HTML. Our §1.8 loss is *inspired by* CAST's objective, not copied from it.
- **CFG guidance for a deterministic regressor:** no paper gives a closed-form guidance eq for non-probabilistic regression heads; the output-space extrapolation `v(∅)+s(v(ℓ)−v(∅))` is the standard, defensible adaptation (π0/2510.02148 support the dropout+interpolation pattern for continuous/policy heads, not literally our Gaussian-velocity field).
- **AFUN exact λ values / Bézier K / query attention-mask:** **[SOURCE SILENT]** (not load-bearing — we replace the curve head with our dynamics and adopt MetaQueries' explicit causal mask).
- **MetaQueries connector size per-config** (24-layer Enc-Proj; ~316M reported): varies by MLLM; verify against the table if reproducing the *generation* connector (we don't — our dynamics is the decoder).
- **Qwen3-VL `get_rope_index` kwarg names** drift across `transformers` minors; pin 5.10.2 and unit-test the shape/contiguity assertions in §2.3.
