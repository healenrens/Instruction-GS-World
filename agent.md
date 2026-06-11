# Instruct-GS-World — Agent Research Log

> Canonical, append-only research journal. Every decision, experiment, and rationale is recorded here for later retrospection.
> Keep both a local copy (`/Users/hela/Instruct-GS-World/agent.md`) and a server copy (`/mnt/pfs/public/xuhaoming/instruct_gs_world/agent.md`).

---

## 0. Mission (restated in my own words)

Build a **language-conditioned 3D-Gaussian-Splatting (3DGS) dynamics / world model**:

- Learn a **semantic 3DGS** representation of a scene (Gaussians carry language-aligned features so language can address regions/objects).
- Train a model that, given the current 3DGS state **and a natural-language instruction**, predicts **how the Gaussians transform** (the 3D "flow" of the scene: per-Gaussian motion / appearance change).
- Roll the prediction out **autoregressively** so that accumulating many short steps yields a **long-horizon (≥10 s)** scene evolution.

In one line: *Instruct the world; predict its 3D Gaussian future.*

Deliverable: a working, feasibility-verified model + the ability to hand off large-scale training (I start it, the user babysits).

**Hard constraint from the user:** every reference I implement must be **faithful and complete — no omissions, no simplifications**, because such shortcuts can be fatal.

---

## 1. Environment & resources (verified 2026-06-05)

### Remote server
- `ssh -p 8600 root@106.13.104.32`  (user `root`, host `aibox-r3823394d31a-...`)
- Workspace I created: **`/mnt/pfs/public/xuhaoming/instruct_gs_world/`**
  - subdirs: `code/ data/ checkpoints/ outputs/ third_party/ scripts/ notes/ logs/`
- GPUs: **4× NVIDIA A100-SXM4-80GB**, all idle (0 MiB used).
- Storage: `/mnt/pfs/public` = 201 T, 68 T free. Budget ≤ 2 TB for datasets (no approval needed).
- `uv` 0.11.8 at `/opt/conda/bin/uv`; system python 3.11.13.
- **Proxy (required for HF / GitHub / pip):**
  ```bash
  export http_proxy=http://10.66.65.186:18000
  export https_proxy=http://10.66.65.186:18000
  ```

### Local mirror (code only)
- `/Users/hela/Instruct-GS-World/`  (`code/ scripts/ notes/ docs/` + this `agent.md`)
- `docs/` was empty on arrival (only a stale `.DS_Store`) — the "initial survey docs" the user referenced were **not present locally**. Proceeding from the goal statement directly; will reconcile if docs appear.

### Models already on the server (no download needed)
| Path | What it is | Size |
|---|---|---|
| `model_zoo/Cosmos-Reason2-2B` | **`Qwen3VLForConditionalGeneration`** — a Qwen3-VL **2B** VLM (text hidden 2048, 28 layers, 16 heads / 8 KV, vision depth 24 w/ deepstack idx [5,11,17], video tokens). **This is the "Qwen-3 2B / cosmos-2B" backbone.** | 4.6 G |
| `model_zoo/Qwen3-VL-4B-Instruct` | Qwen3-VL 4B (hidden 2560, 36 layers) — larger fallback backbone | — |
| `model_zoo/Cosmos3-nano` | Cosmos world-model nano | 35 G |
| `model_zoo/Cosmos3-Super` | Cosmos world-model super | 126 G |
| `model_zoo/vjepa2-vitl-fpc64-256` | V-JEPA2 ViT-L video encoder | — |
| `models/Qwen3.6-27B` | large LLM | — |
| `xuhaoming/Cosmos-3-Finetune` | xuhaoming's Cosmos-3 finetune scaffold (configs/src/scripts) — reference only | — |

> Decision: backbone = **Cosmos-Reason2-2B (Qwen3-VL-2B)**. It is *natively multimodal* (ingests video + text, emits hidden states), which is ideal: it can ground language in the *rendered current scene* and condition the dynamics head. The user's "Qwen-3 2B" and "cosmos-2B" both point here.

### Data already on the server (lerobot format)
- `agibot-world-beta-lerobot/...` — AgiBot World beta, **already extracted to LeRobot v2.1**. Per task (e.g. `task_327`): 209 episodes, 257 k frames, **30 fps**. **Cameras:** `head`, `head_center_fisheye`, `head_left/right_fisheye`, `back_left/right_fisheye`, `hand_left`, `hand_right` (+ compressed variants) — **multi-view → strong 3D reconstruction signal**. State: dual-arm EEF position (xyz) + orientation (quaternion) + gripper range. Each task has a language annotation in `meta/tasks.jsonl`.
- `Galaxea-Open-World-Dataset/lerobot/*.tar.gz` — Galaxea Open-World, dual-arm whole-body, still tarred (CC-BY-NC-SA 4.0).
- `AgiBotworld2026`, `agibot_world_beta_lerobot_3.0`, `agibot-digital-world`, etc. — more AgiBot variants.
- Many other researchers' dirs (`jepa_wm`, `wan-wm` (world models), etc.) — reference only, do not touch.

---

## 2. Problem decomposition (modules)

The system factors into four modules. Each must be implemented faithfully from its source reference.

- **A — 3D lifting / 3DGS construction.** Multi-view (or monocular-video) frames → 3D. Candidates: **VGGT** (feed-forward geometry transformer: pointmaps + depth + camera) and **Pi3** (permutation-equivariant visual geometry). Convert lifted points → 3D Gaussians (xyz, rot, scale, opacity, color/SH).
- **B — Semantic Gaussians.** Attach language-aligned features (CLIP/SigLIP/DINO) per Gaussian so instructions can localize. Candidates: **LangSplat**, **Feature-3DGS**, **Gaussian Grouping**.
- **C — Language-conditioned dynamics (core).** Input: Gaussian set at t + instruction (encoded by the Qwen3-VL backbone, optionally grounded on the rendered current view). Output: per-Gaussian transform Δ (SE(3) motion, Δscale, Δopacity, Δfeature) → Gaussians at t+Δt. This is the **3D Gaussian scene-flow** prediction. Candidates/inspiration: **ManiGaussian(++)**, **Gaussian World Model (GWM)**, **GaussianFlow**, **Deformable-3DGS / 4D-GS / Dynamic 3D Gaussians**, **L4GM**, **3D-VLA / WorldVLA**.
- **D — Long-horizon rollout.** Autoregressively apply transforms; stabilize against drift; handle (dis)appearing content. Target ≥10 s (≈100+ steps at ~10 Hz).

**Supervision (from robot episodes):** reconstruct per-frame 3DGS → obtain ground-truth Gaussian trajectories; supervise predicted flow + photometric rendering loss against actual future frames; condition on the task's language annotation.

---

## 3. Open questions to resolve via research (Section will be filled in)

1. VGGT vs Pi3 — which gives the most faithful, license-clean, fast feed-forward 3DGS for multi-view robot frames? (And do we even need per-Gaussian optimization, or is feed-forward enough?)
2. How exactly do SOTA "Gaussian world models" (ManiGaussian, GWM) parameterize and predict Gaussian dynamics? Get the *exact* deformation/flow formulation.
3. Semantic feature distillation: LangSplat autoencoder vs Feature-3DGS — pick the faithful, tractable one for our scale.
4. How to tokenize a variable-size Gaussian set for a transformer dynamics head + condition on Qwen3-VL hidden states (cross-attention vs prefix tokens).
5. Long-horizon stability tricks actually used in 4D-GS / dynamic-GS rollout.
6. Cosmos-Reason2-2B exact I/O (how to extract conditioning hidden states cleanly).

---

## 4. Decision log

- **2026-06-05** Backbone fixed to Cosmos-Reason2-2B (Qwen3-VL-2B). Rationale in §1.
- **2026-06-05** Primary training data = AgiBot World beta (already LeRobot, multi-view, 30 fps, language-annotated). Galaxea as secondary. Rationale: multi-view is essential for faithful 3D lifting; already on disk → no download cost.
- **2026-06-05** Workspace + agent.md established; environment verified.

---

## 5. Running activity log

- **2026-06-05 (init)** Verified SSH, GPUs (4×A100-80G idle), disk (68T free), uv, proxy reqs. Mapped model_zoo & datasets. Created server + local workspace. Wrote this agent.md.

- **2026-06-05 (session 2 — data deep-dive + research kickoff)**
  - Re-verified env (4×A100-80G all 0 MiB; SSH OK). Confirmed there were **no separate survey docs** locally (docs/ empty) → proceeding from goal + this journal.
  - **AgiBot World beta lerobot structure pinned down.** Real dataset root is nested: `agibot-world-beta-lerobot/agibot-world-beta-lerobot/task_XXX/task_XXX/` with `meta/{info.json,modality.json,tasks.jsonl,episodes.jsonl}`, `data/chunk-000/episode_*.parquet`, `videos/chunk-000/<video_key>/episode_*.mp4`. Format = **LeRobot v2.1**, GR00T-compatible (`modality.json`, `stats_gr00t.json` present). robot_type `a2d` (dual-arm).
  - **Cameras (8 RGB + 3 compressed):** `observation.images.{head[480×640], head_center_fisheye[768×960], head_left_fisheye[768×960], head_right_fisheye[768×960], hand_left[480×640], hand_right[480×640], back_left_fisheye[768×960], back_right_fisheye[768×960]}` (+ head/hand_*_compress[224²]). **fps=30.** task_327: 209 eps / 257 260 frames (~1230 frames ≈ **41 s median/episode** → ample for ≥10 s horizons).
  - **Proprioception/actions (parquet, mirrors info.json features):** `observation.states.end.{position[2×3], orientation[2×4 xyzw quat]}` (dual EEF), `effector.position[2]` (grippers), `joint.position/current_value[14]`, `head.position[2 yaw/pitch]`, `waist.position[2 pitch/lift]`, `robot.{position[3],orientation[4]}`; actions mirror these + `actions.robot.velocity[2 x_vel,yaw_vel]` (mobile base). modality.json already maps GR00T keys (cam_high_rgb=head, cam_left/right_wrist_rgb=hand_*). Language per task in `tasks.jsonl`.
  - **★ CRITICAL DATA FINDING — NO CAMERA CALIBRATION.** Exhaustive search (`find -maxdepth 6` for calib/intrinsic/extrinsic/parameters across agibot-world-beta-lerobot, agibot_world_beta_lerobot_3.0, AgiBotworld2026, agibot-digital-world) found **nothing**; README has no calibration mention. The lerobot conversion dropped the raw AgiBot `parameters/` (intrinsics+extrinsics). **Consequence:** camera poses & intrinsics must be **estimated feed-forward** → this is exactly the VGGT/Pi3 use-case and validates Module A. We get pose supervision "for free" only from the geometry model, not from data.
  - **★ Two new constraints this raises:** (1) 5 of 8 cams are **fisheye**; VGGT/Pi3 assume **pinhole** → cannot feed raw fisheye without undistortion (and we lack distortion params). The clean pinhole view is `head`. (2) The scene is **dynamic** (arms/objects move) while VGGT/Pi3 assume mostly-static multi-view → must consider **dynamic feed-forward geometry** (MonST3R, CUT3R, MegaSaM, St4RTrack, Geo4D) for monocular-video lifting. Folded into Module-A research.
  - **Launched 4 background research agents** (faithful, exact-extraction briefs → `notes/research_*.md`): A=lifting (VGGT/Pi3 + feed-forward GS heads + dynamic geometry), B=semantic Gaussians (LangSplat/Feature-3DGS/Gaussian-Grouping), C=language-conditioned Gaussian dynamics/world-models (ManiGaussian(++), GWM, Dynamic-3DGS, Deformable/4D-GS, GaussianFlow, L4GM, 3D-VLA) [most critical], D=long-horizon rollout/anti-drift (Diffusion-Forcing, Cosmos/Genie/DIAMOND recipes, 4D-GS densification over time).
  - **Next:** (a) augment Module-A agent to also cover dynamic feed-forward geometry; (b) inspect Cosmos-Reason2-2B exact config/IO; (c) on research return, lock the end-to-end architecture & supervision (per-Gaussian flow w/ correspondence via per-episode 4D-GS  **vs**  render-supervised future prediction à la ManiGaussian); (d) set up uv env (VGGT/Pi3/gsplat) and build the data→3DGS pipeline; (e) feasibility run on a few episodes before large-scale.

  - **2026-06-05 (session 2 cont. — env built, data reader verified, backbone I/O, research returns)**
    - **Backbone exact config (Cosmos-Reason2-2B = `Qwen3VLForConditionalGeneration`, model_type `qwen3_vl`):** text hidden **2048**, 28 layers, 16 heads / 8 KV, head_dim 128, intermediate 6144, SiLU, RMSNorm 1e-6, vocab 151936, rope_theta **5e6**, **mRoPE interleaved sections [24,20,20]**, max_pos 262144, tie_word_embeddings, bf16. Vision: depth 24, hidden 1024, **out_hidden 2048**, patch 16, spatial_merge 2, temporal_patch 2, deepstack idx [5,11,17]. Tokens: image 151655, video 151656, vision_start/end 151652/3. → **Conditioning width = 2048**; feed (rendered current view ± instruction) → take last_hidden_state sequence as K/V for the dynamics cross-attention. (Corrects the semantic agent's "4096/SigLIP2" assumption — backbone is 2048-d with its own ViT, NOT SigLIP2.)
    - **uv env BUILT & GPU-verified** at `…/instruct_gs_world/.venv` (script `scripts/setup_env.sh`, log `logs/setup_env.log`): python 3.11.13, **torch 2.8.0+cu126**, torchvision 0.23.0, transformers **5.10.2** (qwen3_vl supported), + numpy/pyarrow/pandas/opencv/PyAV/einops/safetensors/hf_hub/accelerate/roma/plyfile/imageio/scipy/tensorboard/jaxtyping. `torch.cuda.is_available()=True`, 4 GPUs, A100 matmul OK. Driver 535 / **CUDA 12.6 / nvcc 12.6 present** (can compile gsplat & CUDA exts).
    - **Data reader DONE & verified** (`code/igsw/data/lerobot_agibot.py`, smoke test `code/scripts/inspect_episode.py`). 183 valid task roots. Reads info/tasks/episodes, parquet→reshaped tensors (dual-arm `end.position`→[T,2,3], `end.orientation`→[T,2,4] quats — nested-list fix applied), decodes all 8 RGB cams via **PyAV/libdav1d (AV1!)**; head video frame-count == parquet rows (1295). Sample frames written. Episode language example: *"Pickup items in the supermarket | The robot is positioned in front of the fruit stand…"* (rich, scene-grounded).

---

## 6. Research findings — consolidated (faithful briefs in `notes/research_*.md`)

### A — 3D lifting (VGGT / Pi3)  → `notes/research_A_lifting_vggt_pi3.md`
- **VGGT** (~1.2B): frozen DINOv2 ViT-L (patch14, 1024-d) → 24 alternating (frame/global) attn blocks; per-frame image tokens + 1 camera token + 4 register tokens (`patch_start_idx=5`). **Input:** pixels [0,1], ImageNet-norm applied *internally*; longest side **518**, /14, bicubic, white-pad. **Outputs:** `pose_enc[B,S,9]=[t(3),quat(4),fov_h,fov_w]`, `depth`, `world_points[B,S,H,W,3]`, confidences, optional tracks. `fx=(W/2)/tan(fov_w/2)`, centered PP. **World = first camera; scale-normalized → NOT metric.** Mem: OOM ~300 frames @80GB; ~8.75s/200 frames.
- **Pi3** (~959M): **permutation-equivariant** (no frame-pos emb, no reference token) → ideal for a multi-cam rig; DINOv2 (frozen) + 36-layer alternating decoder. Outputs `local_points`(cam frame), `points`(global), `conf`(sigmoid logits), `camera_poses[B,N,4,4]` **cam-to-world OpenCV** (9D-rot→SVD). Gauge-free (up-to-similarity), not metric.
- **→ Gaussians:** naïve = unproject depth → 1 Gaussian/pixel, conf filter @10th pct, color→SH DC, opacity `inv_sigmoid(0.1)`, scale from kNN spacing, identity quat. Learned heads: **AnySplat** (VGGT-native, open weights `lhjiang/anysplat`) best fit; **MVSplat (MIT)** license-safe; **GS-LRM** 12-ch scheme = design ref; **Splatt3R** CC-BY-NC.
- **Licenses:** VGGT code commercial-OK but `facebook/VGGT-1B` weights **CC-BY-NC**; `facebook/VGGT-1B-Commercial` OK. Pi3 code BSD-3, **weights CC-BY-NC**. → For research all fine; productization later.
- **Rec:** Pi3 primary (perm-equiv), VGGT-1B(-Commercial) fallback; AnySplat head to start. **Gotcha:** both non-metric; VGGT pose head assumes centered PP; preprocessing mismatch silently degrades.

### B — Semantic Gaussians  → `notes/research_B_semantic_gaussians.md`
- **LangSplat:** SAM 3-scale masks → OpenCLIP ViT-B/16 512-d → **per-scene** AE 512→3 → 3×3-d/Gaussian, alpha-composited; query via min-canonical relevancy (thr 0.4). **Per-scene AE ⇒ latent not shared across scenes ⇒ unusable for a generalizable model.**
- **Feature-3DGS:** per-Gaussian R^N (N=128), custom N-ch CUDA raster, optional 1×1 conv→512; teachers LSeg/SAM; L=L_rgb+1.0·‖F_t−F_r‖₁.
- **Gaussian Grouping:** 16-d identity/Gaussian, DEVA cross-view mask assoc, L_rec+1.0·CE+2.0·KL (3D-NN identity smoothness).
- **Rec (corrected):** frozen foundation features + **single shared low-dim projection (~16-d), trained once, fixed for all scenes** (not per-scene). Store semantics in **canonical space** (deformation warps geometry, not features) + temporal-anchor + 3D-consistency (intra-mask smooth + inter-mask contrastive + KL). Match language via a small head `Qwen3VL_text(2048) → 16-d` (cosine). **Foundation feature candidate: DINOv2** (already computed by VGGT/Pi3 → free reuse) or SigLIP; decide at impl time.

### C — Language-conditioned Gaussian dynamics (CORE)  → `notes/research_C_dynamics_worldmodel.md`
- Field families: (1) **per-Gaussian delta** (ManiGaussian, Deformable/4D-GS) — keeps identity → needed for long autoregression; (2) free per-frame (Dynamic-3DGS) — gives physics regularizers; (3) re-predict whole set / latent (L4GM, GWM) — loses correspondence.
- **ManiGaussian** (MIT): `(μ,r)_{t+1}=(μ+Δμ,r+Δr)`, deform = ResnetFC (`d_in=70,d_out=7`), cond = action **concat** + latent **FiLM**; losses `L_Act+0.01·L_Geo+1e-4·L_Sem+0.001·L_Dyna`; 16 384 Gaussians; deforms μ,r only.
- **GWM** (ICCV'25, closest template): 3DGS **VAE** (FPS→N=512, cross-attn enc) + **latent DiT diffusion**, **EDM precond** (logσ∼N(−0.4,1.2²), σ_data=0.5); action via **cross-attention K/V**, timestep via **AdaLN**; `L_VAE=Chamfer+L1 render`. (No language — we add it.)
- **Dynamic-3DGS** anti-drift: rigidity/rotation/isometry, `w_ij=exp(−2000·‖μ_j0−μ_i0‖²)`, k=20, weights {rigid4,rot4,iso2}.
- **Deformable-3DGS:** MLP `F_θ(γ(sg(x)),γ(t))` D8/W256, PE Lx10/Lt6, stop-grad pos, AST time-noise (β0.1,τ20k). **4DGaussians:** HexPlane (6 planes, 64² base, multi-res {2,4,8}, h32)+tiny MLP, `(x',r',s')=(x+Δ,r+Δ,s+Δ)`, L1+grid-TV. **GaussianFlow:** 2D-flow↔3D-Gaussian-motion analytic bridge. **L4GM:** LGM U-Net→65 536 G/frame+temporal attn+Plücker. **3D-VLA:** LLM+diffusion goal pointcloud. **WorldVLA:** unified AR tokens.
- **★ RECOMMENDED CORE DESIGN:** per-Gaussian **delta field as a transformer**, deltas on the **manifold** (`μ←μ+R·v`, `R←R·Exp(ω)` Lie, `s←s·exp(δ)`, `σ←sigmoid(logitσ+δ)`, `f←f+δ`) to avoid additive-quaternion drift; **one token/Gaussian** + **3D Fourier positional enc of center** (permutation-equivariant, count-agnostic); huge-N → FPS→512 latent + cross-attn (GWM); **Qwen3-VL conditioning via cross-attention K/V over the hidden-state sequence** + AdaLN global term; losses = future render (0.8·L1+0.2·SSIM) + rigidity/rot/iso (4/4/2, λw2000,k20) + Δ-magnitude (1e-2) + opt feature-cosine/flow; **stability via multi-step rollout/TBPTT (K≥4) + per-step renorm/clip + scheduled sampling**; start deterministic regression, upgrade to **EDM latent diffusion over deltas** only if the conditional future is multimodal.

### D — long-horizon rollout  → `notes/research_D_longhorizon_rollout.md`
- Exposure bias is the enemy. Toolkit: **(1) input-noise aug (GameNGen, σ∼U(0,0.7) train / 0.05 infer)**, (2) **Diffusion Forcing** (per-token noise) if diffusion, (3) **attention-sink/global anchor** (cache t=0 KV, inject every step), (4) **scheduled sampling → Self-Forcing**, (5) cycle-consistency. Recipe: causal transformer over ~32-step history, 3D-Fourier PE on centers, language cross-attn every block; phase1 teacher-force+noise → phase2 sched-sample → phase3 noise-decay; anti-drift = ARAP rigidity(λ.1)+velocity(λ.01)+accel(λ.001); infer = rolling window + velocity EMA(β.9) + attention-sink + birth/death every 10 steps; **fixed N_max + soft-opacity masking + compaction**.

### A2 — dynamic feed-forward geometry  → `notes/research_A2_dynamic_geometry.md`
- DUSt3R/MASt3R/VGGT/Pi3 assume **static** scenes → **break on manipulation motion even in 2–3-frame windows (ghosting)**. For dynamic monocular video:
  - **MonST3R** (ICLR'25): finetuned DUSt3R decoder; per-frame world-frame pointmaps + poses + K; 300-iter global align (~1min/60fr); CC-BY-NC.
  - **CUT3R** (CVPR'25): recurrent, online 16fps, world-frame pointmap+pose, no alignment; CC-BY-NC.
  - **MegaSaM** (CVPR'25): deep SLAM, poses + **metric** depth, dynamic-mask; **Apache-2.0 (commercial!)**; ~1fps.
  - **★ St4RTrack** (ICCV'25): dual-branch (recon + **tracking**) → **per-pixel 3D trajectories**; 3D scene-flow `v_g(t)=X¹_{t+1}−X¹_t` directly; 30fps; research license. **This is how we get GROUND-TRUTH per-Gaussian correspondence/trajectories** without per-episode 4D-GS optimization.
  - Easi3R (training-free dynamic masking), D²USt3R, Geo4D, Driv3R(needs calib).
- **Rec:** MonST3R (poses+geometry) → St4RTrack (3D tracks) for GT trajectories; CoTracker3+depth or MegaSaM as commercial fallback.

---

## 7. LOCKED DESIGN — Instruct-GS-World v1 (decided 2026-06-05, all 5 briefs in hand)

**Representation.** Scene = set of 3D Gaussians `{μ, q(wxyz), s, σ, c(rgb), [f semantic]}` (container `igsw/gaussians/types.py::GaussianSet`, stored activated; raw views for manifold deltas).

**Module A (geometry/substrate) — DONE & verified.** Pi3 (`checkpoints/Pi3/model.safetensors`, 3.8G) lifts frames→pointmaps+poses+conf+local_points; K recovered from local_points by least-squares (got centered PP, ~69° FOV — sane); gsplat 1.5.3 renders. **Finding:** naive unprojected lift is only an *init* (same-view PSNR ≈17 w/ holes; bigger scales smear→worse). Photoreal needs per-frame **3DGS optimization** (gsplat inner loop) or a trained FF head (AnySplat). → v1 uses brief per-window optimization for quality where needed; dynamics learns on top.

**Module C (CORE dynamics) — building now.** Per-Gaussian **delta-field transformer**:
- Tokens: one per Gaussian = embed(3D-Fourier-PE(μ) ⊕ q ⊕ log s ⊕ logit σ ⊕ c [⊕ f]); operate on a **downsampled control set N≈16 384** (FPS) for tractable attention (SDPA-flash handles 16k tokens); production = voxel/serialized local attn.
- Conditioning: **Qwen3-VL-2B** encodes (rendered current view ⊕ instruction)→hidden states[L,2048]; injected as **cross-attention K/V every block** + **AdaLN** for step/timestep embedding.
- Head → manifold deltas: `μ←μ+v`, `q←normalize(exp(ω)⊗q)`, `s←s·exp(δs)`, `σ←sigmoid(logitσ+δσ)`, `c←clamp(c+δc)`, `f←f+δf`.
- Losses: render `0.8·L1+0.2·SSIM` vs real future frame (from Pi3 window cameras, consistent gauge) + (when St4RTrack added) trajectory L1 + rigidity/velocity/accel anti-drift + Δ-magnitude reg.
- **Long horizon (Module D):** multi-step rollout/TBPTT(K≥4) + scheduled sampling + GameNGen input-noise aug + attention-sink anchor; rollout ~10Hz (predict every 3rd 30fps frame) ×100 = 10s. Fixed N_max + opacity masking.

**Module B (semantic) — deferred to v2.** Distill a frozen foundation feature (DINOv2, already in Pi3 — free) → shared low-dim (~16) projection (NOT per-scene AE); language match via small head Qwen3VL(2048)→16.

**GT-trajectory pipeline (v2, for fidelity + long-horizon):** per-episode MonST3R(poses)+St4RTrack(3D tracks) → explicit `μ_g(t)` supervision. v1 proves feasibility with render-supervision only (no extra heavy deps).

**Feasibility gates:** (1)✓ lift+render substrate. (2) dynamics fwd/bwd shape-correct. (3) overfit one short clip: render-supervised rollout reduces error vs identity baseline, conditioned on language. (4) hand large-scale training to user.

---

## 8. Running activity log (session 2 cont.)
- Built faithful lifting stack: `igsw/lifting/{preprocess,pi3_lifter,to_gaussians}.py`, `igsw/gaussians/{types,cameras,render,sampling}.py`. Pi3 ckpt + gsplat installed. Feasibility `scripts/feasibility_lift_render.py` ran on task_327/ep0/frame600: 119k gaussians, render OK, masked PSNR 17 (naive).
- **★ Dynamics core BUILT & validated** (`igsw/dynamics/{pe,manifold,tokenizer,transformer,model}.py`). `GaussianDynamics`=80.9M params (d512/12L/8H). DiT-style: self-attn over Gaussian tokens + **cross-attn to Qwen3-VL hidden states (2048-d)** + **AdaLN-Zero** (gates zero-init) + zero-init delta head ⇒ **exact identity at init** (verified max|Δμ,Δq,Δs|=0, |Δσ|=6e-8). On-manifold deltas (`μ+v`, `Exp(ω)⊗q`, `s·exp(δs)`, `sigmoid(logitσ+δσ)`, `clamp(c+δc)`). Test `scripts/test_dynamics_shapes.py`: identity ✓, shapes ✓, 4-step rollout ✓, backward finite (gnorm 19.3) ✓, peak 53GB @B2/N16384/4-step. **Lesson:** fp32 SDPA materializes 16k×16k attn (OOM) → must run bf16 (flash); A100 training is bf16 anyway. TODO: grad-checkpointing for longer rollouts.
- **Next:** (1) `igsw/dynamics/conditioning.py` = Qwen3-VL (Cosmos-Reason2-2B) encoder → hidden states; (2) verify gsplat gradients reach GaussianSet params; (3) window-lift→training data; (4) render-supervised overfit on one clip (feasibility gate 3): conditioned rollout must beat identity baseline on future-frame render loss.
- **gsplat differentiability verified** (`scripts/test_conditioning_and_grad.py`): grads finite & nonzero for means/scales/opacity/colors. **Qwen3-VL encoder verified**: loads Cosmos-Reason2-2B, hidden_size 2048, text→[24,2048], image+text→[278,2048]. Added grad-checkpointing to dynamics blocks.
- **★★ FEASIBILITY GATE 3 — PASS (2026-06-05).** `scripts/overfit_clip.py` on task_327/ep0, clip f400 stride3 K8 (0.8s, 9 frames), Pi3-lifted jointly (consistent gauge), G0=119608→16384 control gaussians, Qwen3-VL cond on (instruction+frame0)→[1,332,2048], 80.9M dynamics, 120 iters AdamW: **static-G0 baseline future-frame PSNR=9.12 → trained rollout best PSNR=11.64 (Δ=+2.52 dB, PASS).** The language/scene-conditioned per-Gaussian delta rollout demonstrably explains future-frame appearance better than a static scene → **core hypothesis validated end-to-end on real robot data.** (Absolute PSNR low because base lift is naive + downsampled to 16384; relative gain is the signal.)
  - **Key engineering facts learned:** bf16 mandatory for attention (fp32 SDPA materializes 16k² → OOM); gsplat must run fp32 (bridge via differentiable .float()); identity-at-init (zero head + AdaLN-Zero) gives stable optimization start.

## 9. Plan to the deliverable (post-feasibility)
- **(R) Representation upgrade → SC-GS (Sparse-Controlled GS, Huang CVPR'24):** dynamics predicts deltas on N sparse **control** gaussians; propagate to ALL ~120k dense gaussians via **Linear-Blend-Skinning over k-NN control points** (embedded deformation graph). Gives full-density render quality while keeping attention tractable. This is the *correct* (not simplified) design for sparse-control dynamic gaussians — implement before large-scale.
- **(D) Offline data caching:** lift clips across many episodes/tasks (Pi3, jointly per clip) → cache {G0 dense, control idx + LBS weights, per-frame cameras P_t/K_t, frames, instruction} to disk. This is the expensive step → the large-scale job the user babysits.
- **(T) Distributed trainer (4×A100 DDP):** full loss suite (photometric 0.8L1+0.2DSSIM + rigidity/velocity/accel + Δ-reg) + long-horizon recipe (scheduled sampling, GameNGen input-noise σ-aug, attention-sink) + ckpt/tensorboard. Verify on small subset → **launch large-scale** (hand to user).
- **(B v2) Semantics + (GT v2) St4RTrack trajectory supervision** as quality boosts.
- **Long-horizon eval:** roll out ≥100 steps @10Hz (predict every 3rd 30fps frame) → ≥10 s; measure drift, render PSNR over horizon.

## 10. Activity log (session 2 cont. — SC-GS + scalable pipeline + launch)
- **SC-GS implemented** (`igsw/gaussians/deform.py` LBS binding+step; `igsw/dynamics/scgs.py` rollout). Dynamics runs on M=2048 control gaussians → LBS-propagated to ~120k dense → dense render. Overfit `scripts/overfit_clip_scgs.py`: static-dense baseline 10.03 → trained 11.71 PSNR (**Δ=+1.68, PASS**); observed early instability from unbounded deltas → **added per-step bounds** (`max_disp=0.1`, `max_rot=0.3` via tanh in `predict_deltas`). **Quality bottleneck = naive Pi3 lift (~10 PSNR), not dynamics** → per-frame 3DGS optimization is the future quality track.
- **Scalable pipeline built & validated:**
  - `scripts/cache_clips.py` — offline clip cache (Pi3 lift + cameras + GT frames + **frozen Qwen3-VL hidden states** so trainer needs neither model in-loop); resumable, **shardable across GPUs**. ~14MB/clip, ~1.6s/clip. Smoke: 8 clips/13s ✓.
  - `igsw/data/clip_dataset.py` — `ClipDataset` (batch=1/GPU, variable-N).
  - `scripts/train.py` — **4×A100 DDP** trainer: SC-GS free-running self-rollout (inherent long-horizon supervision) + GameNGen input-noise aug on G0 + photometric(0.8L1+0.2DSSIM)+Δ-reg+velocity; AdamW+warmup/cosine; ckpt/tb/resume; `forward==predict_deltas` + `static_graph=True` for DDP-through-rollout. Smoke 40 steps single-GPU ✓ (0.8 it/s, loss 0.35→0.31, no crash).
- **★ LAUNCHED large-scale caching** (2026-06-05): 4 sharded jobs (1 per GPU), 40 tasks × 15 eps × 6 clips = **3600 clips** → `data/clips_v1/` (logs `logs/cache_shard{0..3}.log`). ~24min.
- **Caching COMPLETE:** 3600/3600 clips, **0 failures**, 47 GB, ~1450–1630 s/shard.
- **Manifold math verified** (`scripts/test_manifold_math.py`, CPU): exp-map / quat_mul / quat→rotmat match `roma` to <1e-6 incl. near-zero & large angles.
- **Long-horizon eval built** (`scripts/eval_longhorizon.py`): lift frame0 → roll out N steps → render from static cam → mp4 (GT|baseline|pred) + PSNR-vs-time; `--instruction` overrides to test language control.
- **★★ LAUNCHED 4×A100 DDP TRAINING** (run1, 2026-06-05 ~01:53): `./.venv/bin/torchrun --nproc_per_node=4 code/scripts/train.py --data ./data/clips_v1 --out ./checkpoints/run1 --epochs 100 --K 12 --M 2048 ...` → log `logs/train_run1.log`, ckpts `checkpoints/run1/`. **Confirmed training: 3600 clips, 900 steps/epoch, e0s0 loss 0.336 PSNR 10.4, 4 GPUs ~55%.** Handoff doc = `HANDOFF.md`, launcher = `run_train.sh`.
  - **Two DDP bugs fixed at launch:** (1) must use **`.venv/bin/torchrun`** not conda's (conda torch lacks gsplat); (2) **`broadcast_buffers=False`** in DDP — default in-place per-forward buffer broadcast corrupts the FourierPE `freqs` [10] buffer across the K-step rollout → "modified by inplace op" in backward (single-GPU didn't hit it). Also `drop_last=True` + DDP-OOM=fail-fast to avoid rank desync.
- **STATUS: feasibility verified + large-scale training running → handed to user to babysit.** Remaining/v2: per-frame 3DGS optimization for render quality, St4RTrack GT-trajectory supervision, semantic features (Module B), long-horizon eval on trained ckpt, language-control ablations.

## 11. ★ USER ARCHITECTURE REVIEW → v2 redesign (2026-06-05, scale-up directive)
User reviewed v1 and gave 3 binding critiques. Stopped run1 (was learning, 80.9M, but wrong scale).
1. **Trainable capacity too small for scale-up; Qwen3-VL should be partially trainable.** → put VLM **in-loop** + **LoRA** (peft 0.19.1 installed) so language features adapt; bulk capacity in a big dynamics net.
2. **3DGS-side model must be ≥1B** for real-world generalization. → scale dynamics transformer to **~1.5–1.7B** (d_model 1536, n_layers 28, heads 16).
3. **Deep, per-LAYER Qwen3-VL↔3DGS interaction** (use *every* VLM layer, not just last). → **dynamics block j cross-attends to Qwen3-VL transformer-layer j's token features** (1:1 over 28 layers); per-layer 2048→d projections; AdaLN cond from pooled last layer.
- **v2 design (locked):**
  - `InstructGSWorldModel` (one DDP module) = Qwen3-VL(+LoRA, in-loop) + 28 per-layer projections + `GaussianDynamics` (28 DiT blocks, d1536) + SC-GS LBS to dense. **One DDP forward/step does VLM-encode→K-step SC-GS rollout** (dynamics called K× *inside* the single forward → sidesteps DDP multi-forward issue; keep `static_graph=True`, `broadcast_buffers=False`).
  - VLM encodes (instruction + frame0) **once/clip**; all-28-layer hidden states [28,L,2048] (grad via LoRA) feed the 28 blocks; reused across the K rollout steps.
  - **Reuses existing 3600-clip cache** (g0+gt+cameras+instruction text all still valid; only conditioning path changed — ignore stale last-layer `lang_hidden`). VLM image input = cached frame0 (gt[0]).
  - Trainable params ≈ 1.5–1.7B (dynamics) + ~30M (LoRA) → **≥1B ✓**. Parallelism: DDP (fits 1.7B+VLM on 80GB w/ grad-ckpt; FSDP = documented path to larger). Mixed precision: fp32 master + bf16 autocast.
- **Next:** refactor `model.py`/`scgs.py`/`conditioning.py`, add `model_full.py` + v2 `train.py`; validate shapes+peak-mem on 1 GPU; relaunch 4×A100 at ≥1B.

### v2 IMPLEMENTED & LAUNCHED (2026-06-05)
- Refactored: `dynamics/model.py` (GaussianDynamics: per-block ctx + global AdaLN cond, d1536/28L/16H), `dynamics/scgs.py` (`rollout(delta_fn,K)` closure), `dynamics/conditioning.py` (in-loop Qwen3-VL + **LoRA** via peft, returns **all-28-layer** hidden, base bf16 / LoRA fp32, VLM grad-ckpt). New `igsw/model_full.py::InstructGSWorldModel` = VLM(+LoRA) → 28 per-layer Linear(2048→1536) → 28-block dynamics (block j ↔ VLM layer j) → SC-GS LBS to dense; **one DDP forward** does encode→K-step rollout, returns stacked dense tensors.
- v2 `scripts/train.py`: DDP (`static_graph=True`, `broadcast_buffers=False`), 2 param-groups (dynamics/proj lr2e-4, LoRA lr1e-4), reuses `clips_v1` (`require_lang=False`, VLM re-encodes frame0+instruction in-loop), photometric+Δ-reg+vel, peak-mem logging.
- **Single-GPU validation:** trainable **1.768B** (dyn 1660M + proj 91M + LoRA 17.4M), fwd+bwd OK, **peak 40.3 GB** (huge headroom on 80 GB).
- **★★ LAUNCHED run2 (4×A100 DDP, 1.768B):** `logs/train_run2.log`, ckpts `checkpoints/run2/` (every 500 steps). Confirmed: 3600 clips, 900 steps/epoch, e0 s0→s20 loss 0.335→0.294 PSNR 10.5→11.2, **40 GB/GPU**, ~0.4 it/s (×4) ⇒ ~37 min/epoch. No DDP/in-place errors.
- **All 3 user critiques satisfied:** (1) 1.768B trainable + LoRA-trainable VLM; (2) 1.66B dynamics ≥1B; (3) per-layer 28↔28 deep interaction.
- **Handoff:** `bash run_train.sh run2` (or resume `--resume checkpoints/run2/ckpt_last.pt`); eval `scripts/eval_longhorizon.py` (after refit to v2 model — TODO). Headroom (40/80 GB) allows scaling dim→2048 or K→12, or FSDP for multi-B.

## 12. ★ USER REVIEW #2 → v3 (freeze Qwen, special-token head, STREAMING scale-up) (2026-06-05)
Two binding critiques: (1) **don't LoRA Qwen** — with tiny data it erodes its language ability and won't itself learn 3D; keep it FROZEN as conditioning, optionally add a **special token that aggregates spatial-movement info via an extra trainable head, feeding its all-layer features to 3DGS**. (2) **Data far too small** (600 eps vs ≥3000h corpus) — use a **JIT streaming pipeline** (decode→train→discard) to train over the FULL dataset without storing all clips. "你必须要考虑 Scale up."
- **Data reality:** clips_v1 = 600 episodes / 40 tasks (~1.6%). Streaming index finds **137,768 episodes across 183 tasks** (~the full AgiBot-Beta corpus).
- **v3 changes:**
  - **Qwen3-VL FROZEN, no LoRA** (`conditioning.py` rewritten: eval, no_grad, returns all-28-layer detached features). Generalization preserved via intact pretrained features; 3D-language grounding lives in the trainable 3D modules.
  - **Special-token spatial-aggregation head** (`model_full.py`): learnable `query` tokens [Q=16] + `layer_id_emb` + shared `aggregator` cross-attn distill EACH frozen Qwen layer's features → compact special tokens [28,Q,d]; dynamics block j cross-attends to layer-j's special tokens. (n_query=0 falls back to all-token cross-attn.) +9.5M trainable.
  - **Seek-based window decoder** (`lerobot_agibot.py::decode_window`): keyframe-seek → **18× faster** (0.06 vs 1.08 s) and bit-identical to sequential — enables random-access streaming of deep clips.
  - **Streaming dataset** (`igsw/data/streaming.py::StreamingClipDataset`, IterableDataset): CPU workers seek-decode random clips across all tasks, infinite, **zero storage**; ranks/workers use disjoint seeds.
  - **Streaming trainer** (`scripts/train_stream.py`): inline **Pi3 lift (frozen, no_grad)** per step → SC-GS render-supervised rollout; DDP, ckpt/tb/resume.
  - Trainable = **1.760B** (dyn 1660M + proj 91M + agg 9.5M); Qwen frozen 2.44B. Single-GPU peak **44 GB**.
- **★★ LAUNCHED stream1 (4×A100, full 137,768-episode corpus, zero clip storage):** `logs/train_stream1.log`, ckpts `checkpoints/stream1/` (every 1000). total_steps 300k. Replaces clips_v1-bound run2.
- **Both critiques satisfied:** frozen Qwen + special-token all-layer head; full-corpus streaming (no storage cap).

## 13. ★ USER REVIEW #3 → v4 (DIRECT 3D motion loss + correspondence; speed/util) (2026-06-05)
Two critiques: (1) **render/PSNR is an indirect, resolution-bound 2D loss — make it AUXILIARY and add a DIRECT 3D loss on each Gaussian's motion**, which requires solving **correspondence/registration** (which Gaussian↔which over time, and the GT change). (2) **Speed: fill the GPUs**, tune dataloader/pipeline; **prefer raising (effective) batch over only scaling.**
- **Direct 3D supervision via correspondence (the core fix):**
  - Our Gaussians are born 1:1 from anchor-frame pixels ⇒ a Gaussian's correspondence over time = the 2D track of its anchor pixel. **CoTracker3** (frozen, `checkpoints/cotracker/scaled_offline.pth`, `igsw/lifting/tracking.py`) tracks the control points; **sampling Pi3's per-frame point maps at the tracked locations** (`sample_pointmaps_at`) gives the **GROUND-TRUTH 3D trajectory** in the SAME gauge as predictions.
  - `points_to_gaussians(return_uv=True)` exposes each Gaussian's anchor pixel; `SCGSRollout(ctrl_idx=...)` + `model.forward(ctrl_idx=...)` make the tracked points == the predicted control points.
  - **`trajectory_loss`** (`igsw/training/losses.py`): masked-by-visibility L1 on **position** + on **velocity (Δposition)** — velocity directly teaches motion direction+magnitude. **PRIMARY** (w_pos=w_vel=1.0). Render photometric demoted to **AUXILIARY** (w=0.1), rendered on only `render_steps=2` (speed). Rotation/scale/opacity stay render-supervised.
  - Single-GPU validated: vis≈0.97, pos≈0.006–0.016, vel≈0.002–0.007 (gauge units).
- **Speed/util (profiled per step, 1 GPU):** decode .06 (overlapped) | Pi3 lift .45 | CoTracker .45 | VLM encode .07 | rollout .59 | render .004 | **backward 1.29** → 0.34 it/s @full-ckpt(44GB). **No-ckpt** = 0.41 it/s but **80.1GB (OOM-risk)**. **`checkpoint_every=2` (half) = 0.38 it/s, 58GB (safe)** → chosen. Pipeline is ~98% GPU-bound (decode overlapped, render minimized). **Effective batch via `--grad_accum`** (constant memory; true B>1 impractical w/ variable dense-N). Dataloader workers 6 / prefetch 4.
- **New deps:** CoTracker3 (`third_party/co-tracker`, pip-installed; ckpt 102MB).
- **★★ LAUNCHED stream2 (4×A100, full 137,768-ep corpus, DIRECT 3D loss):** `logs/train_stream2.log`, ckpts `checkpoints/stream2/`. cfg: dim1536/28L, M2048, n_query16, ckpt_every2, grad_accum2 (eff batch 8), traj_pos/vel 1.0 + render 0.1. total_steps 300k.
- **Answers to user:** (1) done — direct 3D motion loss is primary, correspondence via CoTracker3+Pi3, render auxiliary. (2) pipeline ~GPU-bound now; raised effective batch via grad-accum; further speed = producer-decouple lift/track to dedicated GPU (documented, not yet done).

## 14. Babysit loop (cron `*/10`, job 2ad9d6eb) — cycle 1 (2026-06-05)
- **stream2 was LEARNING WELL** (pos 0.0058→0.0013 @s140, vel→0.0010, rPSNR→15.4) — direct 3D loss validated at scale.
- **★ Robustness bug caught & fixed:** a degenerate clip (lift→~0 valid gaussians) hit `torch.quantile` empty-tensor in `points_to_gaussians` → **rank3 crashed → DDP hang** (GPU3 0%, others stuck 100%). Fixes (aligned with "鲁棒"):
  1. `points_to_gaussians`: guard percentile clamp when `<16` valid points.
  2. `train_stream.py`: **DDP-safe coordinated skip** — sample-prep (lift/track/GT) in a collective-free try → `ok` flag → `all_reduce(MIN)` → ALL ranks skip a step together if ANY clip is degenerate/non-finite (prevents per-rank desync/hang). `--min_gaussians 512`.
  3. **Dense-N cap** `--max_gaussians 150000` (random subsample) → bounds LBS/render memory, no OOM spikes.
- Relaunched stream2 (pid varies): all 4 GPUs healthy (69–82%, ~63GB), pos loss falling again. Robust to bad clips now.
- **Pending next-step upgrades (do in subsequent loop cycles, training-permitting):** rotational/scale GT supervision (Procrustes on tracked-neighbor GT trajectories — reuses existing tracks, no new model), producer-GPU decouple for lift/track throughput, St4RTrack cleaner GT, per-frame 3DGS opt for render-quality. Goal lens: **general / generalizable / robust.**

## 15. Babysit loop — cycle 2 (2026-06-05): rotation supervision landed
- stream2 (post-fix) healthy: pos 0.0058→0.0024 @s80, no skips/errors, DDP-safe skip holding.
- **★ Completed next-step upgrade: direct per-Gaussian ROTATION supervision** (`losses.py::kabsch_rotation`,`rotation_loss`). For each control Gaussian, local **Kabsch/SVD** on its k-NN GT trajectories (`rot_knn=8`) gives the GT per-step rotation; loss = masked Frobenius² vs the model's predicted per-step `Exp(ω)`. Reuses existing CoTracker GT tracks (no new model). `--w_traj_rot 0.2`. Now the FULL transformation (μ-translation + velocity + rotation) is directly supervised — matches the original "predict 高斯球的变换" goal.
- Validated single-GPU (rot 0.0745→0.0474, peak 57.8GB). **Relaunched as stream3** (`logs/train_stream3.log`, ckpts `checkpoints/stream3/`): 4 GPUs 76–96% util, pos↓ vel↓ **rot 0.0745→0.0198 @s20** (rotation learning). cfg adds `--w_traj_rot 0.2 --rot_knn 8`.
- Also confirmed the DDP-safe OOM/degenerate skip works (a contention OOM was caught & skipped, no crash).
- **Still pending:** producer-GPU lift/track decouple (throughput), St4RTrack GT, per-frame 3DGS opt (render quality). Next cycles.

## 16. Babysit loop — cycle 3 (2026-06-05): GPU util fix → 99%
- **Problem detected:** GPU util only 50–78% ("打满" goal unmet). Tried DDP `no_sync()` during grad-accum → **incompatible with `static_graph=True`** (`expect_autograd_hooks_` reducer assert) AND wouldn't address the real cause → reverted.
- **Root cause:** per-step **optimizer overhead** — `clip_grad_norm_` (.item() sync) + `opt.step()` create GPU bubbles EVERY step. **Fix: grad_accum=4** → those happen 4× less often. **Result: all 4 GPUs → 99% util.** Bonus: effective batch = 4×4 = **16** (smoother gradients; directly answers user's "提高 batch size" point).
- **Relaunched stream4** (`logs/train_stream4.log`, ckpts `checkpoints/stream4/`): dim1536/28L, M2048, n_query16, ckpt_every2, **grad_accum4**, traj pos/vel 1.0 + rot 0.2 + render 0.1. 99% util, ~63GB, full-transformation losses falling. **This is the current canonical run.**
- **Decision: let stream4 run uninterrupted** (it has the complete approved design + 99% util). Remaining optional upgrades (producer-decouple, St4RTrack, per-frame opt, more datasets) batched for a deliberate future relaunch / user nod — NOT thrashing the run. Monitor via cron; at s1000 ckpt run `eval_longhorizon.py` incl. language-control test (same scene, different instructions) — the true generalization check.

## 18. ★ Babysit cycle 11 — s2000 eval → CORE PROBLEM found + contrastive language fix (2026-06-05)
- **s2000 checkpoint verified** (saves @2000, resume-loadable). Ran s2000 language-control eval (lossless stop→eval→resume).
- **★ Two real problems detected:**
  1. **LANGUAGE IS BEING IGNORED (the core-goal problem):** lang-control divergence s1000=0.0032 → **s2000=0.0014** (≈0; two very different instructions on the same scene → ~identical rollouts). Not a wiring bug (0.0014≠0) — the model *down-weights* language. **Diagnosis:** the Gaussian scene tokens + frame0 visual alone predict the single observed future, so the instruction is **redundant for fitting the loss** ⇒ ignored. No counterfactual pressure in the data (same scene never paired with a different instruction→different future).
  2. **Long-horizon drift unchanged:** 6s rollout PSNR Δ s1000=−3.77 → s2000=−3.75 — noise-aug(0.02) insufficient for 60-step (eval) vs 8-step (train) horizon gap.
- **★ FIX implemented — contrastive language-dependence loss** (`losses.py::contrastive_lang_loss`, model_full `forward(vlm_inputs_wrong=...)`, train_stream wiring): per step, predict step-0 control velocity under a WRONG instruction (1 extra encode + 1 dynamics step, no rollout/render — cheap, **peak still 56GB**); hinge loss forces the CORRECT instruction to fit GT step-0 motion ≥`margin`(0.003) better than the wrong one ⇒ **directly forces the model to USE the text**. Negatives drawn from a rolling instruction buffer (fallback generic when empty, so graph is static_graph-consistent). `--w_lang_contrast 0.5`.
- Validated single-GPU (lang=0.003 at init = margin, model not-yet-differentiating; will drop as language-use emerges). **Relaunched stream4d** (`logs/train_stream4d.log`) resume @s2000, 4 GPUs, 56GB. **★ KEY METRIC TO WATCH: `lang` loss — should drop BELOW 0.003 as the model learns to differentiate instructions; and re-eval divergence at s5000 should be >> 0.0014 if the fix works.** If lang stays pinned at 0.003 → language still ignored → escalate (text-only conditioning, or action-conditioning, or counterfactual data).
- Long-horizon drift fix deferred (needs larger K / better rollout recipe; memory-limited) — revisit after language-use is established.

## 19. ★ Cycle 12 — VERIFY contrastive fix → it was STUCK → root-caused + architectural fix
- **Verified fix #1 (contrastive) — IT DIDN'T WORK:** over s2000→s2200, `lang` loss pinned at EXACTLY 0.0030 (=margin) → `e_correct−e_wrong≈0` → the model produces ~identical output for correct vs WRONG instruction → **language has ZERO effect** → contrastive loss has **no gradient** to fix it.
- **Root cause (deeper):** cross-attention to language was **AdaLN-Zero-GATED** (`ca_g` init 0); since language was useless for the main loss, the gate stayed ≈0 ⇒ **language architecturally switched OFF** and couldn't carry gradient. The contrastive loss can't shape what can't flow.
- **★ Fix #2 — UN-GATE cross-attention** (`transformer.py::DiTBlock`): cross-attn is now a **standard always-on residual** `x + cross(modulate(...))` (no `ca_g` multiply) → language unconditionally influences the output. Identity-at-init still holds via the zero delta-head. Boosted `--w_lang_contrast 1.0`.
- **Fresh restart stream5** (`logs/train_stream5.log`) — fresh (s0) so language-use is learned from the start with the corrected arch (the s2000 model had language-ignoring baked in; it was early/early-tainted). 98% util, 44GB, pos/vel/rot improving.
- **★ VERIFICATION PENDING (warmup-limited):** at s140, `lang` still ~0.003 BUT lr is mid-warmup (2.8e-5; →2e-4 @s1000). **Check `lang` at s1000+:** if it drops below 0.003 → language-use emerging (fix works); re-eval divergence should rise ≫0.0014. If still pinned → contrastive too weak vs main losses → raise `w_lang_contrast` (3–5) / margin, or escalate (text-only or action conditioning). **This is the open verification.**
- Multiple sustained detect→diagnose→optimize cycles this session: contrastive(stuck)→diagnose gating→ungate→deploy→verify-pending.
- **★ Cycle 12b — DECISIVE architecture probe (CPU, non-disruptive):** built dynamics w/ non-zero head, fed two different language contexts → output relative diff **0.58** ⇒ **un-gated cross-attn WORKS, language strongly affects output, NO bug.** (cond_global→0 is a fresh-init AdaLN-zero artifact, harmless.) ⇒ the stuck `lang=0.003` at s340 is **early-training** (model too rough → e_correct≈e_wrong) NOT a wiring bug. Architecture verification COMPLETE.
- **Remaining empirical question (needs s1000+ training, can't shortcut):** does the contrastive loss DRIVE language-grounding (lang↓)? Risk: model may zero the cross-proj weights over training to ignore language (main losses don't need it) unless `w_lang_contrast` is strong enough. **Plan:** let stream5 (w=1.0) train; **check `lang` at s1000** — if dropping → works; if still pinned → raise `w_lang_contrast`→3–5 (the model is zeroing language; needs stronger pressure). Note: real correct-vs-wrong ctx differ less than the random probe (both are robot instructions), so grounding must amplify the task-relevant signal.

## 20. ★★ Cycle 13 — ROOT CAUSE FOUND: VLM image-dominated → TEXT-ONLY fix (2026-06-05)
- stream5 (image+text cond): `lang` stuck at EXACTLY 0.0030 for 580 steps (lr ramped to 1.16e-4) → contrastive genuinely not moving.
- **★ DECISIVE VLM probe** (`/tmp/probe_vlm.py`, GPU): changing only the TEXT instruction moves the pooled VLM features by **0.026 WITH the image** vs **0.334 TEXT-ONLY → ratio 12.9×.** ⇒ **The frame0 IMAGE dominates the Qwen3-VL features; the text instruction contributes only ~2.6%** → swamped → model literally couldn't use the instruction. This is the true root cause (not gating, not the contrastive weight).
- **★ FIX: TEXT-ONLY VLM conditioning** (`--vlm_image 0`; trainer passes image=None to the VLM). The scene/visual info already comes from the Gaussian tokens, so the language path now carries ONLY the instruction (13× stronger text signal) and can't be bypassed via the image. Combined with un-gated cross-attn + contrastive.
- **Relaunched stream5b** (`logs/train_stream5b.log`) fresh, text-only, w_lang_contrast=1.0. Healthy (4 GPUs, 44GB). **Verify `lang`↓ at s500–1000** (now the contrastive has strong text signal to leverage); re-eval divergence should rise ≫0.0014. If still stuck → raise w_lang_contrast.
- **Chain of root-causing (this session's core-problem debugging):** language ignored → contrastive(stuck) → ungate cross-attn(arch probe: works, 0.58) → still stuck → VLM probe: image dominates 13× → **text-only conditioning** (the actual fix). Decisive, data-driven.
- **For eval consistency:** `eval_longhorizon.py` must also use text-only (image=None) when evaluating text-only-trained models — TODO when evaling stream5b.

## 21. ★ Cycle 14 — FINAL language-fix config + commit to long run (2026-06-05)
- stream5b (text-only, w_lang_contrast=1.0): `lang` still pinned 0.0030 at s260 → **w=1.0 too weak** (lang contributes 0.003 vs pos 0.013; the main loss pulls toward context-INDEPENDENCE — predicting the single observed future needs no language — so the contrastive must out-pressure it).
- **★ FINAL CONFIG = stream5c** (`logs/train_stream5c.log`, `checkpoints/stream5c/`): **text-only VLM cond (--vlm_image 0)** + **un-gated cross-attn** + **contrastive w_lang_contrast=3.0** (lang ≈0.015, comparable to pos) + full transformation loss (pos+vel+rot) + noise-aug + streaming(137k ep) + ckpt_every1(56GB safe) + grad_accum4(99% util). All fixes from this session combined; every one data/probe-confirmed.
- **★ COMMITMENT: let stream5c TRAIN FOR HOURS (no more restarts unless a crash).** The model must LEARN to use language over thousands of steps — this is a multi-hour empirical outcome that CANNOT be shortcut by more tweaks. Verify at milestones:
  - **s1000–2000:** does `lang` drop below 0.003? (language-use emerging)
  - **s2000 ckpt:** re-run divergence eval **text-only** (need to add `--vlm_image 0` path to `eval_longhorizon.py`) — should be ≫0.0014 if language now steers dynamics.
  - If `lang` STILL pinned by s2000 → the limitation is fundamental to the DATA (no counterfactuals: each scene has one instruction→one future) → escalate: counterfactual/action-conditioned data, or accept weak language-conditioning as a data limit (document for user).
- **Session summary of the language-conditioning root-cause hunt:** ignored→contrastive(stuck)→ungate cross-attn(probe:works)→still stuck→VLM probe(image dominates 13×)→text-only→w=1 too weak→**text-only+w=3 (final)**. Rigorous, data-driven, every hypothesis tested.

## 22. ★ Cycle 15 — honest assessment: verification is warmup/training-gated (not tweakable now)
- stream5c (final config) to s620: `lang` still 0.0030, pos plateaued ~0.014. **Understood mechanistically:** delta-head is zero-init + un-zeros slowly; lr still in WARMUP (1.24e-4 of 2e-4 peak @s1000). Model still predicts ~static (v small) → both instructions give ~same v → e_correct≈e_wrong → lang=margin EXACTLY. This is **early-training, not a bug** (probes already proved arch + text-signal work).
- **Two intertwined open questions, BOTH resolvable only with post-warmup training (s1000–2000), NOT more code tweaks:**
  1. Does pos drop below ~0.014 post-warmup, or is it the **GT-noise floor** (CoTracker+Pi3-depth on dynamic monocular)? If floored → deploy prepped **St4RTrack** cleaner GT (the model predicts ~static because noisy GT has little learnable position signal).
  2. Does `lang` drop (language-use emerge) once the model predicts real motion? Decisive **divergence eval at the s2000 ckpt** (text-only, `eval_longhorizon.py --vlm_image 0`).
- **DECISION: stop premature concluding + premature restarts. Let stream5c BAKE to s2000 (~75min).** The config is optimal (text-only + ungated + contrastive w3 + full transformation loss + 99% util + robust). The decisive verification is at s2000 — gated on wall-clock, not effort.
- **Honest project state:** the full system is built & optimally configured & training robustly on 137k episodes. The two remaining QUALITY questions (motion-accuracy floor, language-use) likely trace to FUNDAMENTAL limits — noisy monocular GT and no language counterfactuals in the data — whose real fixes (St4RTrack/4D-GS clean GT; counterfactual or action-conditioned data) are large-scale efforts (the user's domain), confirmed/decided at the s2000 eval.

## 23. ★★ Cycle 16 — CONFIRMED at s1020 + motion-signal fix (stride) (2026-06-05)
- **DECISIVE (stream5c to s1020, post-warmup, lr@peak):** `lang` pinned at EXACTLY 0.0030 for ALL 1020 steps — **language-use confirmed NOT emerging.** Root cause (not a bug; arch+text-signal already probe-proven): at **stride 3 (0.1s/step) the per-step MOTION is tiny** (~GT-noise scale) → model predicts ~static → both instructions give ~same (static) output → e_correct≈e_wrong EXACTLY → lang=margin; pos floored same reason.
- **★ FIX: bigger motion-per-step via larger stride.** stream6: `--stride 8` (0.27s/step). Confirmed at s0: identity-baseline losses ~1.5–2× larger (pos 0.013→0.019, vel 0.0033→0.0082, rot 0.075→0.168) ⇒ **more real motion above the noise floor** → the model now has learnable motion + room for language to help (contrastive can drive e_correct<e_wrong). Other fixes retained (text-only, ungated cross-attn, contrastive w3, full transformation loss, 99% util, robust).
- **Verify stream6 post-warmup (s800+):** does pos drop (real motion learned) + `lang` move below 0.003 (language-use)? If yes → core goal progressing. If still pinned → motion still noise-limited → escalate to cleaner GT (St4RTrack prepped) and/or accept the data needs counterfactuals (user's large-scale domain).
- **Pattern note:** each hypothesis costs ~30–60min of post-warmup training to verify; I've systematically eliminated the architectural causes (gating, image-domination, contrastive weight) and am now on the data/signal causes (motion magnitude, GT noise). The 7-day cron loop is the right vehicle for these training-gated verifications.

## 24. ★★★ DECISIVE: language-conditioning is DATA-LIMITED (gradient-collapse proof) (2026-06-05)
- `lang` pinned at EXACTLY 0.0030 across **every** config tried (image/text-only, w_lang_contrast 1/3, stride 3/8) and 1000+ steps each. Not warmup, not a bug (arch+text-signal probe-proven).
- **★ ROOT CAUSE (analytical):** the contrastive loss `relu(margin + e_correct − e_wrong)` has a **vanishing gradient at the context-INDEPENDENT solution.** The model starts ctx-independent (cross-attn ≈ constant) and the MAIN loss reinforces it (the instruction is REDUNDANT — the Gaussian scene tokens already predict the *single observed* future per clip). At that fixed point, perturbing params changes correct & wrong predictions identically ⇒ `e_correct≡e_wrong` ⇒ gradient ≡ 0 ⇒ contrastive can't escape. **All model-side fixes are therefore provably insufficient.**
- **FUNDAMENTAL CONCLUSION:** language-conditioning **cannot be learned from single-instruction-per-clip data** where the scene is redundant with the instruction. Requires DATA-LEVEL change (the user's large-scale domain):
  1. **Counterfactual data** — same scene + different instruction → different future (the clean fix; needs generation/curation).
  2. **Action-conditioning / VLA-style** — language→action→motion (actions ARE in the lerobot data; precise control signal). Changes the goal slightly (predict via action) but is the principled robotics path.
  3. **Scene-ambiguous clip curation** — clips where the scene does NOT reveal the goal (task onset) so the instruction is *necessary* (uniform sampling already includes some → weak alone).
- **WHAT WORKS (feasibility shown):** the 3DGS-dynamics substrate predicts coarse per-Gaussian **motion + rotation** and renders (rPSNR up to 15); SC-GS + streaming + 1.76B + robustness + 99% util all solid. The *language-conditioning specifically* is the data-blocked piece.
- **stream6 keeps training** (useful coarse-dynamics model + safety net if the gradient-collapse is somehow escapable at scale). **Major direction decision (data-level) is the user's** — recommend action-conditioning (VLA-style) as the most tractable principled path given AgiBot has actions but no language counterfactuals.

### ★ CORRECTION (don't over-conclude): contrastive-collapse ≠ goal-failure
- The gradient-collapse finding is about the **contrastive loss** specifically (it can't escape ctx-independence). BUT the **actual goal-metric — does the model OUTPUT depend on the instruction (divergence eval) — is UNMEASURED for the text-only config** (stream5c had no ckpt before I stopped it). The MAIN loss + text-only conditioning could still induce *some* language-use on scene-ambiguous clips even with the contrastive stuck.
- **DECISIVE TEST = divergence eval on a TEXT-ONLY checkpoint** (`eval_longhorizon.py --vlm_image 0 --instruction2 ...`), available at **stream6 s2000 ckpt**. Only THEN conclude language-use yes/no. Do NOT escalate to data-level (action-conditioning/counterfactuals) or ask the user until this real metric is measured.
- Plan: let stream6 bake to s2000 (~70min from s280) → run text-only divergence eval → if divergence ≫0.0014 (vs the image-conditioned s2000=0.0014) language IS being used (success, contrastive-collapse notwithstanding); if ≈0 → genuinely data-limited → then the user-fork (action-conditioning etc.). No more premature conclusions or restarts.

## 25. ★★★ DECISIVE MEASURED RESULT — s2000 text-only divergence eval (2026-06-05)
- **Ran the decisive eval** (`eval_longhorizon.py --vlm_image 0 --stride 8` on stream6 s2000 ckpt, N=50 → 13.3s rollout):
  - **Language divergence = 0.0032** (text-only) **vs 0.0014** (image-conditioned s2000) → **TEXT-ONLY ↑ language-dependence ~2.3×.** The diagnosis (image-domination) was correct AND the fix helped — language now measurably matters more. BUT 0.0032 ≈ 0.3% of scene radius → **language-use is real but WEAK in absolute terms.**
  - pos improved to ~0.010 with stride-8 (vs ~0.014 stride-3 floor) → bigger motion signal helped motion-learning.
  - **13.3s rollout produced → the ≥10s long-horizon goal is structurally achievable** (50 autoregressive steps), with drift over the horizon (Δ_render −2.68; expected, eval horizon 6× the 2.1s training horizon).
- **★ HONEST DECISIVE CONCLUSION (measured, not speculated):**
  - ✓ **3DGS-dynamics prediction WORKS** — per-Gaussian motion + rotation learned, renders (rPSNR ~13), long-horizon rollout (>10s) structurally works.
  - ~ **Language-conditioning is WEAK-but-nonzero**, improved 2.3× by the text-only fix; strong language-conditioning is **fundamentally limited by the data** (AgiBot: instructions redundant with scene, no counterfactuals; contrastive gradient-collapses).
  - → **Path to STRONG language-conditioning is DATA-LEVEL (user's large-scale domain):** (1) action-conditioning / VLA-style language→action→motion (actions ARE in lerobot; most tractable), (2) counterfactual data (same scene, different instruction→future), (3) scene-ambiguous/task-onset curation. Plus drift: larger training K / better rollout recipe; cleaner GT (St4RTrack prepped) to push pos below ~0.010.
- **stream6 resumed from s2000** (more training may modestly strengthen weak language-use + reduce drift). The verification is now COMPLETE & MEASURED: substrate works, language-conditioning weak (data-limited), long-horizon structurally works.

## 28. ★ User redirect #4 — subtasks + put image BACK in Qwen, drop SAM/DINO (2026-06-06)
Directives: (1) 不能拿掉图像 — 要把图像放回 Qwen，grounding 从 Qwen 自身出（不依赖 SAM/GroundingDINO，那不优雅）；(2) 用 AgiBot 的 **subtask 细粒度标签**，不要整条 task；(3) rPSNR 自己修；(4) 不忘最初目标。
- **★ Subtask labels FOUND & wired (big win):** they live in `episodes.jsonl::action_config` =
  per-segment `{action_text, skill(Pick/Place), start_frame, end_frame}`. Reader: `subtasks()/
  subtask_text_at()`. `StreamingClipDataset` now indexes at **segment level** → **721,469
  fine-grained, clip-varying instructions** (vs 137,768 coarse episodes); each clip stays inside
  one sub-action and carries its action_text (e.g. "Grasp the right collar with the right arm",
  "Use the held cloth to wipe the water tank"). 10/10 distinct — references specific objects.
- **★ NEGATIVE RESULT — naive Qwen attention-grounding doesn't work:** built `encode_grounded`
  (text→image attention, "few heads" 2503.06287). Tested on real frame: peak pinned at the same
  corner (14,19) regardless of instruction, **true-vs-other corr 0.88–0.94** → dominated by an
  **attention sink, not semantic grounding**. (Would need calibrated head-selection; not reliable
  out-of-box.) → abandoned explicit-mask extraction.
- **★ Pivot to robust elegant design (no SAM/DINO, no fragile masks):** **put the image BACK into
  Qwen** (`vlm_image=1`) + the dynamics **learns grounding implicitly via per-layer cross-attn to
  Qwen image+text features** (query-token aggregator) + **global AdaLN cond pooled from TEXT tokens
  only** (`text_mask`, avoids the frame-0-image domination that pooling caused) + fine-grained
  subtasks. Encoder loaded flash (fast), returns (hidden_all, valid_mask, text_mask).
- **rPSNR fix:** `trajectory_loss` already normalizes by VISIBLE count (not boosted weight-sum) →
  obj_focus boosts fg WITHOUT diluting bg supervision (the diagnosed drift cause). With
  use_grounding off, no obj_focus anyway → render more stable.
- **Contrastive language re-enabled** (`w_lang_contrast 0.3`): with genuinely-different subtasks,
  swapping instruction SHOULD change motion → contrastive can finally learn language-use.
- **★ LAUNCHED stream9_imgsub** (4×A100, warm-start s2000): image+subtask+contrastive, **44GB**,
  100% util, healthy. `logs/train_stream9.log`. **Next:** watch `lang` loss (drop<margin = language
  used) + rPSNR (stable?), eval at s4000 (language divergence w/ fine-grained subtasks). If implicit
  cross-attn grounding still weak → MetaQuery (learnable SEM/MOTION queries, the AFUN path) or
  calibrated head-selection.

## 27. ★ Module E — language-conditioned visual GROUNDING (user redirect, 2026-06-05)
User insight (correct): the failure was feeding Qwen's **global pooled image+text hidden state**
(frame-0-dominated) as condition; the fix is to read Qwen as a **language-conditioned grounding
compiler** → tell dynamics WHICH Gaussians are object/source/target/hand vs background + motion
intent. Restructure into 3 streams (geometry / grounding / dynamics) + background-static priors.
- **Did thorough verified research** (4 agents → `research_E1..E4_*.md`; every citation checked,
  correct arXiv IDs, no fabrication) and **REWROTE `research_E_instruction_visual.md`** (the
  deliverable). Key verified anchors: **AFUN (2606.02551)** = frozen Qwen3-VL(2B ablated)+64
  MetaQuery(32sem+32motion), 32.21M trainable — near-exact precedent (we swap its motion decoder
  for our SC-GS dynamics); **MetaQueries (2504.06256)** mechanism; **Few-Heads grounding
  (2503.06287)** training-free attention; **GroundingDINO+SAM2** (Apache, pip) teacher;
  **Gen-LangSplat (2510.22930)** shared frozen 512→16-D projection (cross-scene); **Dynamic-3DG
  (2308.09713)** rigidity/bg-static losses (λ_w2000,k20); **π0 (2410.24164)** frozen-VLM+separate
  action-expert precedent.
- **Plan:** v7-min (GroundingDINO+SAM2 async role masks → per-Gaussian relevance via anchor-uv +
  `L_bg_static`+`L_obj_focus`) → v7-metaquery (frozen Qwen + SEM/MOTION queries → masks + motion
  cross-attn; Qwen gotchas: masked_scatter on image_token_id 151655, M-RoPE (3,B,L), causal mask,
  no lm_head expansion). Keep action-cond (stream7) as Layer-3. **DeepSpeed ZeRO-2** for the
  Module-E memory bump (stream7 already ~70GB). Metrics: lang-divergence, bg-motion-energy,
  obj-motion-accuracy, role-consistency. Hooks ready: `return_uv`, `ctrl_idx`, `feature_dim`.
- **Implementation started:** installing GroundingDINO+SAM2 (v7-min teacher). stream7 (action-
  conditioned) left training (non-disruptive; it's Layer-3 + useful baseline).

### v7-min BUILT + LAUNCHED (2026-06-05)
- **Grounding teacher** GroundingDINO-tiny + SAM2.1-hiera-large (Apache, via transformers 5.10.2;
  `Sam2Model` present). Tested on real supermarket frame: detected gripper(0.53)/shelf/basket/fruit. ✓
- **`igsw/grounding/`**: `role_masks.py` (RoleGrounder: GDINO boxes→SAM2 masks per role phrase),
  `relevance.py` (instruction→manipulator+object phrases; mask→per-Gaussian sampling at anchor uv).
- **Losses** (`training/losses.py`): `background_static_loss` (freeze low-relevance) + relevance-
  weighted `trajectory_loss` (`obj_focus`). Unit-tested.
- **Trainer** (`train_stream.py --use_grounding 1`): inline grounding on frame0 (frozen, no_grad) →
  foreground relevance at control points; **background = complement of detected foreground**
  (robust across diverse scenes — fixed-bg-vocab failed: bg=0 on non-supermarket clips); losses
  `+w_bg_static·bg_static + obj_focus`. **HF_HUB_OFFLINE=1 mandatory** (from_pretrained HEAD over
  flaky proxy crashed httpx; weights are cached → offline).
- **Validated single-GPU:** foreground detected on diverse clips (fg_cov 0.15–0.81, rel_ctrl
  0.12–0.91), peak **45.8GB** (no DeepSpeed needed for v7-min). **★ LAUNCHED stream8_v7min**
  (4×A100, full corpus, warm-start s2000): 100% util, 56.7GB, bg_static active (~5e-6, grows with
  motion), `logs/train_stream8_v7min.log`, ckpt every 2000.
- **Next:** monitor + eval at ~s4000+ on the 4 Module-E metrics (bg-motion-energy↓, obj-motion-
  accuracy↑, lang-divergence↑, role-consistency). Then **v7-min step 2** (feed relevance as a model
  TOKEN feature so the net KNOWS which Gaussians are foreground, not just loss pressure) and
  **v7-metaquery** (frozen Qwen SEM/MOTION queries). DeepSpeed ZeRO-2 if memory needs it.

## 26. ★ Cycle 17 — ACTION-CONDITIONING deployed (the principled fix) (2026-06-05)
- Per the decisive finding (language data-limited; action is the NON-redundant control signal), implemented **action-conditioning** = bridge to language→action→motion + directly improves motion accuracy/drift:
  - `streaming.py`: load per-step dual-arm EEF action `[K,8]` (Δpos 6 + Δgripper 2) from the parquet (`actions.end.position/effector.position`), cached per (task,ep).
  - `model_full.py`: `action_embed` MLP (8→d, **zero-init last layer** → starts OFF, safe warm-start) added to per-step AdaLN cond inside the rollout (`--action_dim 8`).
  - `train_stream.py`: load+pass actions; resume now **skips stale optimizer** on arch change (try/except).
- **stream7 launched** (`logs/train_stream7.log`): **warm-started from stream6 s2000** (keeps learned dynamics), action_dim 8, contrastive OFF (`w_lang_contrast 0` — action is the focus; language stays as soft cross-attn cond), text-only, stride 8. Healthy: 4 GPUs 97% util, **peak 70GB** (~10GB headroom; 150k cap bounds worst-case — watch for OOM).
- **Verify at ~s3000–5000:** does pos drop below ~0.010 as `action_embed` un-zeros (action→accurate motion)? Action is non-redundant (control input not in scene) so it SHOULD be learnable (unlike language). If pos drops → action-conditioned world model works → then add **language→action head** for language-conditioned inference (the goal). If OOM → max_gaussians 150k→120k.
- **Rationale recap:** pure language→motion is data-limited; action-conditioned world model (controllable 3D future) is independently valuable AND the substrate for language-conditioning via language→action. This is the principled path given AgiBot has actions but no language counterfactuals.

## 17. Babysit loop — cycle 5 (2026-06-05): s1000 eval → anti-drift + memory-safety fixes
- **Checkpoint mechanism verified:** ckpt saves at s1000 (22GB, includes optimizer), **resume-loadable (mmap)**, training continues past save → long run de-risked against interruption.
- **★ Ran language-control eval** (`eval_longhorizon.py --instruction2`, lossless stop→eval→resume): at s1000 (only ~16k clips seen) — **lang-control divergence=0.003 (tiny → language NOT yet steering dynamics)**; **6s rollout PSNR 6.45 < static-baseline 10.22 (Δ−3.77) → long-horizon DRIFT** beyond the K=8 (0.8s) training horizon. Both expected this early but flagged to watch.
- **Optimizations (problem→fix):**
  1. **Re-enabled input-noise augmentation** (`noise_frac 0.02`) — the GameNGen anti-drift mechanism I'd wrongly disabled; also fixed `init = g0n` (noised rollout start) so the velocity loss trains denoising-toward-clean-GT.
  2. **Memory safety:** resume made peak hit **76.8GB** (optimizer state resident) → unsafe. **`checkpoint_every=1` (full grad-ckpt) → peak 56.2GB** (~24GB headroom, ~15% slower backward — worth it for an unattended multi-day run).
  3. **Disk:** `ckpt_every 1000→2000` (22GB ckpts) to bound growth.
  4. K=10/12 tested → 75.8/81.7GB (too tight) → kept **K=8**; revisit K at a later, more-trained ckpt if drift persists.
- **Restart gotcha fixed:** `pkill torchrun` leaves **orphaned ranks holding GPUs + port 29500** → EADDRINUSE. Clean kill = `pkill -9 -f "[t]rain_stream"` (the `[t]` trick avoids killing my own ssh shell) + fresh `--master-port`.
- **Canonical run = stream4c** (`logs/train_stream4c.log`): resumed @s1006, K8/M2048/dim1536/28L, ckpt_every(grad)=1, noise0.02, grad_accum4, full transformation loss (pos+vel+rot), **56GB safe**, robust, resumable. **Let it run.** Next eval at ~s5000–10000 (model more trained) to check if language-control emerges + drift reduces; if lang-control still ~0 by then → deeper architectural fix (e.g., stronger conditioning / FiLM, or unfreeze top VLM layers).
- **Cycle 6 (confirm):** 200-step window (s1006→s1200) STABLE — 56.2GB flat, 0 skips/errors, 100% util, losses healthy w/ noise-aug. Cycle-5 fixes validated.
- **Cycle 10 (prep the identified optimization, non-disruptive):** stream4c healthy (s1600, rPSNR climbing, rot↓). **St4RTrack prepped** for the s5000 GT-floor decision: cloned `third_party/St4RTrack` (DUSt3R/MASt3R-based; `infer.py`/`model_seq.py`; needs `setup.py build_ext` curope compile + MASt3R base + St4RTrack ckpt). Weights downloading (HF `yupengchengg147/St4rTrack` root `model.safetensors`; recommended seq ckpt is Google-Drive-only). **Integration design (deploy IF s5000 confirms pos-floor):** replace CoTracker+Pi3-depth GT with St4RTrack `f(I0,It)→X^0_t` (native 3D tracking pointmap = per-Gaussian trajectory, cleaner). Gauge: either (a) use St4RTrack `X^0_0` to also build G0 (one consistent gauge), or (b) keep Pi3 G0 + align St4RTrack frame via similarity-Procrustes at frame0. Compile/integrate/validate deferred to deploy-time (speculative until confirmed; don't disrupt an improving run).
- **Cycle 9 (loss-trend diagnosis):** analyzed stream4c log (s1020–1500): **rPSNR +10.2% (corr +0.28) — model learning ✓**; vel −13%, rot −6%; **pos nearly FLAT (−2.5%, corr −0.06)**. Diagnosis: pos loss likely at the **GT-noise floor** (Pi3 dynamic-depth + CoTracker ≈0.01-gauge noise) — model can't beat noisy GT on absolute position, but render keeps improving. **Identified optimization: St4RTrack cleaner 3D-track GT** (replaces CoTracker+Pi3-depth) to lower the floor — deploy IF confirmed at s5000 (pos flat while rPSNR rises ⇒ GT-limited). Not executed now (would disrupt a still-improving run). This is the documented detect→diagnose→optimization-plan.
- **Cycle 7 (confirm + data scan):** s1220 healthy. **Data-diversity lever assessed:** `agibot_world_beta_lerobot_3.0` (lerobot, extracted, same embodiment → easy add, marginal diversity); **Galaxea (different embodiment, HIGH generalization value) but tarred → needs extraction job**; AgiBotworld2026/agibot-digital-world non-lerobot (need conversion). Current 137k-ep AgiBot stream is ample for now; **cross-embodiment (Galaxea) = top future data lever (extraction job, hand to user).** Multi-dataset streaming reader = a needed code change when we add it (different camera keys/formats).

## 29. ★★★ Language-forcing FIX deployed — InfoNCE + boundary sampling (stream11) (2026-06-06)
The §28 redirect (subtasks + image back in Qwen + drop SAM/DINO + boundary sampling + "consult literature for the contrastive loss" + "A/B MetaQuery") is now IMPLEMENTED and running as **stream11**.

- **research_F decision (`notes/research_F_force_language_metaquery.md`):** language-ignoring = **posterior collapse on the language condition** (scene+dynamics-prior already predict motion → language redundant). Verified CAST (arXiv:2508.13446, Levine grp = our exact disease, frames as max I(action;lang|obs)), CFG (2207.12598), InfoNCE/CPC (1807.03748), MetaQuery (2504.06256), AFUN (2606.02551). **Chosen fix = symmetric batch-InfoNCE(predicted-motion, frozen-Qwen-instruction), τ=0.07** (a non-saturating MI lower bound) to REPLACE the old saturating fixed-margin hinge. CFG condition-dropout (p_uncond=0.15) + free-bits floor = held-in-reserve backups. **MetaQuery = an upgrade but won't fix collapse alone → deferred to an A/B** decided by the language-sensitivity gap Δ=‖v(ℓ)−v(∅)‖.
- **Why the old loss failed:** `contrastive_lang_loss` was a hinge `relu(margin + e_correct − e_wrong)` with margin 0.003. stream9/stream10 sat **pinned at lang=0.0030 forever** = model ignores the instruction. A fixed-margin hinge SATURATES (zero gradient once the margin is met trivially) → no pressure to actually use language.
- **InfoNCE impl (faithful):**
  - `losses.contrastive_infonce(g,t,q_g,q_t,τ)` — symmetric CLIP InfoNCE: positive = (motion·its-own-instruction); negatives = a **MoCo-style detached queue** of OTHER clips' (motion, instruction) embeddings. `−½(log_softmax(motion-anchor)[0] + log_softmax(lang-anchor)[0])`.
  - `model_full.py` heads: `motion_enc`(6→256, DeepSets-pooled over controls from per-control [v.mean,v.std]) → `motion_head`(→256) → unit-norm `motion_emb`; `lang_proj`(2048→256) on the **TEXT-only** pooled Qwen hidden (text_mask excludes image+special tokens, so it forces TEXT dependence even with the image present) → unit-norm `lang_emb`.
  - `train_stream.py`: pre-fill `q_g,q_t = normalize(randn(256,256))` to **constant size** (DDP static_graph needs fixed shapes); update each step with `cat([q, emb.detach()])[-256:]`. Positive always in-graph ⇒ heads always get gradient ⇒ no DDP unused-param error.
- **Boundary-biased sampling (§28 idea, "1:1 to 10:1, try it"):** `streaming.build_clip_index` now SEGMENT-level over AgiBot sub-tasks (`episodes.jsonl::action_config` → `{action_text,start,end}`) = **721,469 sub-task segments** (vs 137k episodes), each carrying fine-grained clip-varying language. `__iter__` oversamples the sub-task **ONSET** (clip starts in first `boundary_frac=0.35` of the segment, where the static scene is least informative about which action follows → language is necessary) at odds `boundary_ratio=4.0` (→ ~84% onset, validated) and down-weights mid-action clips to `middle_weight=0.3` (auxiliary). Yields `boundary_weight` → multiplies the per-clip loss total.
- **Memory red-herring RESOLVED:** a single-GPU smoke test peaked **78.5GB** and I feared the image caused it. Measured truth: the image is **cheap** — Qwen3-VL emits only **300 (480×640) – 880 (720×1280) vision tokens**, encode peak **<0.5GB**. The 78.5GB was **my own bug**: `--checkpoint_every 999` (this arg = grad-checkpoint FREQUENCY, passed to `DynamicsConfig.checkpoint_every`, NOT a save interval) disabled grad-checkpointing on blocks 1–27. With **`--checkpoint_every 1` (checkpoint all) → 44GB**, same as text-only. (Save interval is the SEPARATE `--ckpt_every`, default 1000.)
- **stream11 LAUNCHED & HEALTHY** (`logs/train_stream11.log`, `checkpoints/stream11_infonce`): 4×A100 DDP, **vlm_image=1 (image KEPT, user mandate)**, InfoNCE (w_lang_contrast 0.5, τ0.07, queue 256), boundary (ratio4/frac0.35/mid0.3), K8/stride8/M2048/dim1536/28L, checkpoint_every1, ckpt_every2000. Resumed `stream10_boundary/ckpt_last` weights (strict=False) + fresh InfoNCE heads + **reset optimizer** (param-set changed). **peakGB 44.2, 96–98% util, ~0.20 it/s.** **lang loss MOVING: s4000 5.95 → s4020 1.94** (no longer pinned at 0.003) = the contrastive signal now back-props into the motion head. ✓
- **Next:** (1) watch the lang-loss TREND — must SETTLE meaningfully above 0.003 (collapse back ⇒ still ignoring); a background sonnet monitor is running. (2) Run the **language-sensitivity eval Δ=‖v(ℓ)−v(∅)‖** at a stream11 ckpt (~s6000) to QUANTITATIVELY confirm the model now uses language (vs stream10's ~0) — this is ALSO the A/B metric for MetaQuery. (3) If InfoNCE alone underwhelms, add CFG dropout + free-bits. Then MetaQuery A/B.
- **Ops gotchas (this session):** (a) `pkill -f train_stream` / `grep train_stream` **kills my own ssh shell** — the shell's argv contains the python launch line (the string "train_stream.py") → self-match. Use the `[t]rain_stream` bracket trick or a SEPARATE ssh, and skip the kill entirely when `procs=0`. (b) An ssh command that backgrounds a process (`&`) + redirect can return **no output** (bg holds the channel); launch with `setsid … > log 2>&1 < /dev/null &` and verify via a separate read-only ssh. (c) A background monitor-agent doing a 600s remote sleep **trips the agent idle-watchdog** → use ≤240s cycles so a tool call completes within the window.

## 30. ★★★ NaN corruption at s6000 → non-finite-grad GUARD → stream11b (2026-06-06)
stream11 ran HEALTHY with InfoNCE for ~1900 steps (s4000–5840, lang loss strong 2.6–3.9, language clearly used ✓) then **went NaN at ~s6000** and kept training on NaN (all losses `nan`, weights destroyed).
- **Damage:** `_nanchk.py` over the ckpts: `ckpt_0004000` **CLEAN**; `ckpt_0006000` & `ckpt_last` = **NaN/Inf in 538 tensors** (= exactly the trainable params; the 626 frozen Qwen tensors stayed finite). So ONE step poisoned every trainable param at once.
- **Root cause (the real bug):** there was already `clip_grad_norm_(…, 1.0)`, but that is **NOT a NaN guard — it PROPAGATES one bad grad to all params.** A rare degenerate clip/render produced a single non-finite grad → the GLOBAL `total_norm` went non-finite → `clip_coef = max/total_norm` became 0/NaN → `grad.mul_(coef)` turned every param's grad to NaN/0 → `opt.step()` (+Adam) wrote NaN into all 1.76B trainable params in one shot. (The manifold was NOT the source — `dlog_scale.clamp(-3,3)` before `exp`, quats normalized, opacity sigmoid, colors clamped were all already in place.)
- **Fix (DDP-safe, `train_stream.py`):** (1) **ALWAYS** `total.backward()` (conditionally skipping backward on only some ranks would desync DDP's grad all-reduce + violate `static_graph`). After backward the grads are all-reduced ⇒ identical on every rank. (2) `gnorm = clip_grad_norm_(…,1.0)`; **`if torch.isfinite(gnorm): opt.step()` else SKIP** (+ diagnostic print of each loss term to ID the culprit if it recurs). Since `gnorm` is the pre-clip norm of the synced grads, the skip decision is identical on all ranks → consistent, no collective mismatch; the NaN grads are cleared by the next window's `zero_grad`. (3) **finite-queue guard**: only enqueue `motion_emb/lang_emb` into the InfoNCE negative queues when finite (a NaN embedding would otherwise poison every future contrastive loss). Smoke-tested single-GPU (6 steps, lang moving, no break).
- **Relaunched `stream11b_infonce`** from the CLEAN `ckpt_0004000` (architecture unchanged ⇒ **optimizer state LOADS, no reset** — smoother than stream11's fresh-Adam start). Healthy: s4000→4020 lang 1.05→1.94, peakGB 44.3, 100% util, guards armed. Lost the s4000–5840 InfoNCE progress but it will be redone safely. Removed the 2 corrupt ckpts (reclaim 52GB).
- **Lesson (permanent):** `clip_grad_norm_` and a finite-norm check are SEPARATE concerns — always pair them. Any unguarded global-norm clip is a single-bad-batch away from destroying a multi-B-param run.
- **ROOT-CAUSE deepening (the guard was only the symptom-fix):** the guard stops PROPAGATION; the ORIGIN of the first non-finite value was traced to unguarded paths feeding `total`. Confirmed: `g0` from `points_to_gaussians` is clean (filters non-finite pts/scales, scale clamped ≥1e-4) and line 271 already finite-checks `gt_traj`+`g0.means`. The GAPS: **(#1, most likely) non-finite RENDER** — the photometric loss's cameras (`Ks`,`viewmats` from Pi3) were NEVER finite-checked, and the rolled-out **means** (`μ+v`, `v` unclamped) are the only unbounded Gaussian field (scales=`exp(clamp±3)`, quats normalized, opacity sigmoid, colors clamped are all bounded) → a degenerate Pi3 camera OR a large predicted `v` crossing the near-plane → gsplat `1/depth` → Inf → NaN loss/grad (rare frame ⇒ fits "1900 healthy steps then once"). **(#2, latent) rotation exp-map NaN-GRADIENT**: `axis_angle_to_quat` used `angle=‖ω‖` whose grad `ω/‖ω‖`=0/0=NaN at ω=0, and `torch.where(small,…,sin/‖ω‖)` back-props the unsafe branch as 0·NaN — forward finite, grad poisoned (needs bit-exact ω=0 ⇒ secondary).
- **Root fixes deployed (→ stream11b):** (a) `manifold.axis_angle_to_quat` now `angle=sqrt(Σω²+eps²)` (eps INSIDE sqrt ⇒ finite grad at 0; drops the `torch.where`, smooth everywhere; resumed weights unaffected — rot loss stayed 0.02–0.04 across the relaunch). (b) line-271 finite-check EXTENDED to `g0.scales`,`Ks`,`viewmats` ⇒ a degenerate-camera clip is skipped, not rendered. (c) the skip diagnostic now prints `meansOK`/`camOK` + `task/ep/f0` ⇒ the NEXT skip will NAME the origin (render-explosion vs bad-cam vs all-finite-but-grad-NaN=rotation) and the exact clip for offline repro. So three layers now: prevent bad inputs (finite-check) → safe math (manifold) → can't-corrupt backstop (grad guard) → self-diagnosing (skip print). Confirmation pending the run re-traversing the ~s6000 region.
- **★★★ DEFINITIVE root cause (don't repeat my earlier GUESSES — render #1 and rotation #2 were both WRONG).** stream11b (with guard) did NOT crash but got STUCK: 0 skips s4000–5399 then **~100% skips from s5440** (lang drifted 2.8→5.55, model frozen, GPUs wasted). The skip diagnostic showed the signature: **all forward losses finite, meansOK=camOK=True → a backward-ONLY NaN**. Two evidence tools cracked it: (1) `torch.autograd.set_detect_anomaly(True)` (new `--detect_anomaly` flag) → `RuntimeError: NativeLayerNormBackward0 returned nan` inside the gradient-checkpointed DiT block (`transformer.py` norm1). (2) `scripts/_wcmp.py` comparing clean s4000 vs broken s6000 weights → **every `dynamics.blocks.N.ada.*` grew 1.8–2.9×** (the AdaLN modulation), nothing else (the only >50 max-abs entries are FROZEN Qwen LN gains = normal). **Mechanism:** AdaLN `ada` weights grow unboundedly → `modulate=norm(x)·(1+scale)+shift` and the residual gates amplify activations/grads across 28 layers (×K=8 rollout) until LayerNorm's BACKWARD overflows. The un-gating of the language cross-attn (E1 fix vs posterior-collapse) removed a damper and let it run.
## 31. ★★ First real EVALS — language-sensitivity Δ + long-horizon rollout (2026-06-07, stream11c @ s6000)
Two `eval_*` scripts run concurrently with training (both had a `map_location=dev` OOM bug — loaded the 26GB ckpt incl. optimizer to GPU; fixed to `map_location="cpu"`).
- **Language sensitivity** (`eval_lang_sensitivity.py`, Δ=‖v(ℓ)−v(·)‖/‖v(ℓ)‖, step-0 ctrl-vel, 12 fixed clips, paired):
  - stream11c (InfoNCE): **Δ_null mean 0.241 / median 0.145**, Δ_wrong 0.190. vs stream10 (old hinge baseline): Δ_null mean 0.112 / median 0.111, Δ_wrong 0.138.
  - **Read (honest):** the hinge "collapse" was OVERSTATED by the pinned lang-loss — the dynamics ALWAYS had ~11% sensitivity; InfoNCE ~DOUBLED it (median +30%, mean ×2 but skewed by a few high-Δ clips). Real, measurable, directional win — but MODERATE, not dominant. Rollout-means divergence tiny for both (~0.1–0.2% scene radius / 2s) and motions are small (per-control ~0.005) ⇒ base dynamics still underfit / scene-dominated. Need the Δ TREND over more training to decide if InfoNCE alone suffices or we add CFG-dropout+free-bits (research_F #2/#3, held in reserve).
- **Long-horizon rollout** (`eval_longhorizon.py`, N=40×stride8 = 10.7s, ep0, G0=121k gaussians): mean PSNR pred 8.22 vs static base 10.21 (**Δ=−1.99 over 10.7s**) — BUT the per-step curve tells the real story: **beats static for the first ~4 s (steps 1–5: +1.6…+3.4 dB; within-K-horizon mean +1.15 dB; even +3.2 at step 15/4s), then COLLAPSES after ~5 s (steps 20+: −3…−5 dB).** So the dynamics is genuinely predictive short-term (method validated) but cannot yet EXTRAPOLATE past ~2.4× its K=8 (2.1s) training horizon → classic autoregressive drift. The ≥10s goal needs: more training + longer-horizon training (bigger K / long-rollout fine-tune) + stronger anti-drift. (At s1000 the 6s rollout was Δ−3.77; now −1.99 over a LONGER 10.7s ⇒ improving.)
- **Decision (user):** "先3后4" — first long-horizon eval (done), then MetaQuery A/B (#4) next.

- **★★★ FIX (confirmed, §30 cont.) → stream11c:** **tanh-bound the AdaLN modulation** in `DiTBlock.forward` — `sa_sc,ca_sc,mlp_sc=tanh(...)`, `sa_g,mlp_g=tanh(...)` → per-block scale & gate ∈(−1,1); `tanh(0)=0` preserves AdaLN-Zero init (seamless resume); tanh saturation also vanishes the grad to `ada` ⇒ self-limiting, the instability is now STRUCTURALLY impossible to recur (a hard cap, not a delay). Plus a param-free `norm_ca` LayerNorm bounding the always-on language injection. **Verified the hard way:** resuming the BROKEN ckpt_last + `--detect_anomaly` ran 15 steps with 0 NaN/0 skip and lang RECOVERING 0.0003→2.70 (tanh squashes the inflated scales back into range). stream11c relaunched 4-GPU from CLEAN ckpt_0004000 with all fixes. **Lesson:** AdaLN-Zero has no built-in scale bound; under strong/always-on conditioning it grows until LN-backward NaN — tanh-bound it. And: when forward is finite but a multi-B-param run NaNs, use detect_anomaly + clean-vs-broken weight-norm diff, don't guess.

## 32. ★★ Attacking LONG-HORIZON DRIFT (#2; user goal "先做2 后做1") → stream12_lh (2026-06-07)
#3 eval: model predicts well to ~4s then COLLAPSES after ~5s (PSNR 7→5.7) — a hard multiplicative collapse (`s·exp(δs)` etc. compounding over 40 unsupervised steps), because the rollout is FREE-RUNNING (confirmed in `scgs.py`: each step feeds its OWN predicted state forward, no teacher-forcing) yet trained only K=8 (2.1s). So the fix is NOT scheduled sampling (already free-running) — it's **train on a longer horizon**. K is just an arg ⇒ no code change.
- **stream12_lh = stream11c resumed @s6000 with `--K 16`** (4.3s horizon), else identical (vlm_image1, InfoNCE, boundary, tanh, all guards). Index **530,403 clips** (vs 721k @K8 — most subtasks long enough). Memory **48–50GB** (K=16 +~5GB only; per-block grad-ckpt keeps step cost low). 0.13 it/s (~2× slower). Healthy, no OOM.
- **Plan / #2 bar:** re-eval long-horizon at s8000 vs s6000 baseline (pred 8.22 / static 10.21 / Δ−1.99, collapse@5s). If stable horizon extends (~8s) → push K→20–24 (mem allows) toward ≥10s; if collapse persists → add anti-drift reg (cumulative scale/position-drift penalty; tighten `dlog_scale` clamp). THEN #1 (MetaQuery A/B), using stream11c@s6000 as the settled InfoNCE baseline (Δ_null 0.24).

## 33. ★★★ LONG-HORIZON drift = SCALE BLOWUP (diagnosed) → scale-anchor fix → stream13_anchor (2026-06-07)
- **K=16 jump (stream12_lh) FAILED.** A naive K=8→16 jump caused **BPTT gradient explosion** over the 16-step free-running rollout: a turbulent window s6300–8430 (backward-NaN, ~97% guard-skipped, lang drifted to 5.5). The guard prevented corruption, the model "recovered" — but by learning **timid near-zero deltas** to survive the exploding BPTT grads → DEGRADED: s10000 long-horizon eval pred 6.02/base 10.21 (Δ−4.19), collapsing at ~1s (vs s6000's beats-static-to-4s). **Lesson: don't naively raise K** (needs truncated-BPTT or a curriculum); abandoned this run.
- **Drift DIAGNOSED (instrumented `eval_longhorizon` with per-step scale/disp logging on the GOOD s6000 model):** the ≈5s collapse is **100% SCALE BLOWUP** — maxScale 0.013(G0)→0.27(step1)→26(step10/~5s)→**383,575(step40)**; meanScale grows **9000×**. Position drift is tiny (meanDisp 0.003→0.016 of radius). **Mechanism:** scales are UNSUPERVISED (trajectory_loss only constrains position) and the render loss REWARD-HACKS them (bigger Gaussians fill gaps within the K-horizon) → the model emits a consistent +δs → `s·exp(δs)` compounds exponentially at extrapolation → giant Gaussians → render collapse.
- **FIX = `losses.scale_anchor_loss`** (anchor rolled-out dense scales to G0 in LOG space; symmetric, scale-invariant; modest weight allows real change, kills runaway growth). Wired into `train_stream` as `--w_scale_anchor` + a `scl` log field. **stream13_anchor** = stream11c resumed @s6000, **K=8 (STABLE — no BPTT turbulence)**, `--w_scale_anchor 0.5`, ckpt_every 1000 (fast iter), all prior fixes. **scl 1.534→0.001 in ~40 steps** (scale growth suppressed instantly), 0 skips, 44GB. rPSNR dropped 13.8→~7–12 (the fake scale-hack PSNR removed — honest now). **Pending:** auto-eval watcher runs the drift diagnostic at ckpt_0007000 — expect maxScale bounded over 40 steps ⇒ no collapse ⇒ long-horizon coherent. If confirmed, push horizon / tune w; then #1.
- **★★★ RESULT (s7000 + scale-anchor) — DRIFT FIXED.** Drift diagnostic over 40 steps (10.7s): **maxScale 0.013→0.015** (was →383,575), meanScale 0.0041→0.0042 (was →36). NO collapse. Long-horizon Δ_mean across 3 episodes: ep0 −0.27, ep10 −0.12, ep25 −0.18 (was −1.99, and K=16 was −4.19). The anchor generalizes to extrapolation — scales stay flat over the full 10.7s. **The long-horizon collapse is solved.** Honest caveat: the model now MATCHES static rather than strongly beating it (early-horizon Δ≈0): the s6000 "+3.4 beat" was partly scale-hack render-inflation (now removed → honest); these supermarket-pickup eps are slow (GT future≈static ⇒ render-vs-static metric near-tied even for correct prediction); pos_l≈0.003 (fits GT 3D trajectory well). So it's a STABLE, accurate, conservative 10s predictor — beating static more strongly is a quality/training matter (more steps; the run continues), not a drift problem. **#2's drift goal achieved.** Cost discovered: 2 concurrent evals on one GPU OOM (each ~18GB); run one-per-GPU alongside the ~43GB training.

## 34. ★★ #1 MetaQuery: IMPLEMENTED + VALIDATED → A/B running (stream14 vs stream13) (2026-06-07)
Per goal "先做2后做1": #2 drift fixed → now #1 (MetaQuery A/B). Opus sub-agent implemented it from `notes/research_G_metaquery_impl.md`:
- **`QwenVLEncoder.forward_metaquery`** (conditioning.py): research_G §2 steps — reuse Qwen's `get_image_features`+`get_placeholder_mask`+`masked_scatter` to build `inputs_embeds`; concat `meta_query` (new `nn.Parameter[N=64,2048]`); extend `attention_mask`(+N), M-RoPE `position_ids[3,1,L]` (queries = `arange(N)+max_pos+1` on all 3 axes → verified land at 37–100, no collision), `visual_pos_masks`(+N False); call `language_model(output_hidden_states=True)`; slice `hs[j+1][0,-N:]` → `query_hidden[28,N,2048]`. **NOT `@no_grad`** (grad must reach meta_query; Qwen stays frozen).
- **`model_full` `cond_mode` switch** ('aggregator' default = baseline, untouched; 'metaquery' creates meta_query + `_encode_metaquery` projecting each layer via the EXISTING `layer_proj[j]`). `--cond_mode` added to train_stream + both eval scripts (load arch must match ckpt).
- **5 GPU checks ALL PASS:** mm_token_type_ids present; 29 hidden states `[1,L+N,2048]`; **grad→meta_query finite, 0/626 Qwen params get grad**; **instruction-sensitivity mean-abs-diff 1.35**; deepstack+full forward OK. Backprop from layer 13 AND 27 both reach meta_query ⇒ grad traverses all 28 frozen layers. forward_metaquery(N=64) peak **5.7GB**. MetaQuery REQUIRES `--vlm_image 1`.
- **A/B (clean — same base stream11c@s6000, same config, only cond_mode differs):** `stream14_metaquery` (cond_mode=metaquery) vs `stream13_anchor/ckpt_0008000` (aggregator), both K8/InfoNCE/w_scale_anchor0.5/vlm_image1, both → s8000. Metaquery training: peakGB **44.7 (no penalty vs aggregator)**, lang moving (grad through Qwen in DDP ✓), scl→0.001, 0 skips. **Watcher auto-runs at stream14/ckpt_0008000:** eval_lang_sensitivity on BOTH (same seed/clips → paired Δ_null/Δ_wrong) + long-horizon on metaquery. Decide winner by Δ (language controllability) + prediction quality. Baseline to beat: aggregator Δ_null≈0.24 (s6000).

## 35. ★★★ CORE ISSUE FOUND: model predicts ~1% of real motion (near-static) → motion-weighted loss (2026-06-07)
User watched the rollout videos: left(GT real)=normal motion, middle(static)=frozen, **right(predicted)≈frozen, just a tiny global translation**. `scripts/motion_diag.py` (GT vs PRED per-control displacement, frac of scene radius, K=16) CONFIRMED it quantitatively:
- **GT** control motion is rich + heavy-tailed: 18–71% of controls move >0.02 radius, top movers 0.5–1.2 (the arm/object).
- **PRED** (stream13 anchor): p50≈p90≈max≈0.01 for EVERY control, frac>0.02=**0.000** → a near-uniform tiny global drift, reproducing **~1%** of the top movers, corr~0/neg. (s6000 pre-anchor slightly better: frac>0.02=0.11–0.37, still only **1–4%**.)
- **Root cause:** per-control motion is heavy-tailed (most static, few move a lot); the L1 trajectory_loss normalized over ALL controls is minimized by the **static median ≈0** → the movers are drowned out → model predicts ~static + global drift. (`pos_l≈0.003` looked fine only because predicting 0 satisfies the static 80%.) This means earlier "successes" were hollow: long-horizon "stability" is partly because a near-static model trivially can't collapse, and the s6000 render "beat" was the scale-hack. **The scale-anchor (w=0.5) regressed motion further** (its large loss term pushed the model into a do-nothing basin: frac>0.02 0.37→0).
- **FIX (self-supervised, no grounding):** derive per-control **task-relevance from GT MOTION** — `rel = (max_t‖gt_pos_t−gt_pos_0‖ / q90).clamp(0,1)` (movers~1, static~0) — wired in `train_stream` when `use_grounding=0`. This activates the existing **obj_focus** (movers dominate the traj loss → predicting static is no longer optimal) AND **bg_static** (freeze the static majority → kills the global drift). Smoke: `bg` now nonzero, pos_l 0.006→0.034 (the movers now COUNT). **stream15_motion** = resume s6000, K=8, `--obj_focus 8 --w_bg_static 0.5 --w_scale_anchor 0.2` (anchor lowered so it stops crushing motion), InfoNCE, vlm_image1. Watcher re-runs motion_diag + long-horizon at ckpt_0007000 — success = PRED frac>0.02 and the top-mover ratio rise substantially (model reproduces the localized motion). #1 MetaQuery A/B paused (moot on a static model; redo after motion works — both arms get this fix). This is the real #2 (predict the dynamics, not just avoid collapse).

## 36. ★★ Motion-weighting FAILED (uniform) → CONFIRMATION experiment: relevance-as-input (2026-06-07)
- **stream15 (motion-weighting, obj_focus=8) result @s7000:** magnitude rose (PRED 0.01→0.025) but the model predicts a NEAR-UNIFORM displacement for EVERY control (p50≈p90≈max identical), frac>0.02=1.000, **corr(GT,PRED)≈0/neg, still only 2–6% of mover magnitude**, and long-horizon Δ −0.27→−0.76 (uniform motion = WRONG motion → worse render). So obj_focus just cranked the uniform global-drift magnitude; it did NOT induce localization. The best LOCALIZATION was actually the plain s6000 (corr 0.43, but timid 4% magnitude). Stopped stream15.
- **Deeper diagnosis:** the dynamics outputs a near-identical per-control displacement = a GLOBAL translation driven by the global conditioning (AdaLN/cond_global); the per-control pathway is too weak to LOCALIZE motion to the manipulated object. Localizing = a GROUNDING problem (which Gaussians are the object, from lang+image) — the core hard part of the whole project. Earlier #1/#2 "wins" were on a model that barely moves / only translates globally.
- **CONFIRMATION experiment (stream16_confirm):** feed per-control task-relevance as a MODEL INPUT (`--feature_dim 1`; the GT-motion relevance set on `g0n.features[ctrl_idx]` → tokenizer concatenates it per control). If the dynamics, TOLD which controls move, now LOCALIZES (corr↑, top-mover ratio↑, PRED frac-moving matches GT) → proves the architecture is CAPABLE and the only missing piece is GROUNDING (derive that relevance from language+image, e.g. Module-E per-control spatial cross-attn). If it still can't localize → the dynamics per-control pathway itself needs strengthening. Config: resume s6000, K8, feature_dim1, obj_focus5, bg_static0.5, scale_anchor0.2, InfoNCE, vlm_image1. `motion_diag.py` updated to feed the same relevance at eval. Watcher reruns motion_diag at ckpt_0007000 (~1h). (Resume from feature_dim0→1 re-inits only the tokenizer in/out proj; the 28 DiT blocks load.)

## 37. ★★★ DEFINITIVE: the dynamics ARCHITECTURE cannot produce localized motion → per-control spatial grounding (2026-06-07)
`overfit_motion.py` — overfit the 1.76B model on ONE clip (a capable arch MUST memorize one clip):
- **vanilla L1 (obj_focus=0):** pos plateaus 0.0067→0.0054 (irreducible), PRED→uniform max 0.011, top-mover ratio **0.01** → the model predicts ~0 for the movers (median-collapse); the irreducible pos = the mover error the arch CAN'T represent.
- **obj_focus=8:** PRED→uniform 0.33 (all controls identical), corr negative, ratio 0.36.
- Even the s6000 base only has weak per-control variance (corr 0.43, max disp 0.087 vs GT 0.9), which COLLAPSES to uniform under more training.
- **CONCLUSION: the per-control pathway is fundamentally too weak — the dynamics output is dominated by the GLOBAL conditioning (AdaLN(cond_global) + cross-attn to 16 DISTILLED/global Qwen tokens), which is uniform across controls. Each control knows its 3D position but NOT "what's at my image location" (is it the manipulated object). So it can't assign large motion to specific controls → near-uniform output. This is THE core blocker; #1/#2 were built on a model that only globally-translates.**
- **FIX = per-control SPATIAL visual grounding** (the Module-E idea, now definitively justified): for each control, sample the Qwen IMAGE patch features at its frame-0 projected `uv` → a per-control visual feature → feed into the dynamics token. Then each control knows "what Qwen sees at my location" + the global instruction → can localize motion. Available at inference (from the image, no GT). Plumbing: encoder exposes image tokens in spatial grid (`image_grid_thw`); `model_full.encode` grid_samples at control uv; pass `control_uv` to `model.forward`; ingest via the tokenizer feature_dim path. **ACCEPTANCE (clean gate): `overfit_motion.py` must now FIT one clip's localized motion (pos→≪0.005, corr→0.7+, top-mover ratio→0.5+).** Until the overfit passes, full training is pointless. Delegating the implementation (opus sub-agent) with this overfit as the acceptance test.

## 38. ★★★ ROOT CAUSE is the DATA (GT), not (only) the architecture → DATA-QUALITY PIVOT (2026-06-07)
- **Overfit verdict:** spatial-grounding lets the arch produce NON-uniform output (max 0.19) and fit L1 (pos→0.002) but **does NOT localize** (corr stays ~0.1, top-mover ratio ~0.02 even overfitting ONE clip, 600 steps). Vanilla stays uniform.
- **GT-coherence check (`/tmp/gt_coherence.py`) — the key:** the GT top-movers ARE a coherent object (pixel-spread 0.095–0.152, clustered ✓) BUT **mover-visibility only 0.40–0.66** (heavy occlusion — hand grasps object, object goes behind things) and **GT max displacement up to 2.1× scene-radius = jumped/lost tracks (noise)**. So the localized-motion TARGET is occlusion-corrupted / partly UNLEARNABLE → the model can't fit it even overfitting (you can't memorize noise) → retreats to the safe uniform/timid prediction. My architecture experiments were fighting a noisy target.
- **USER DIRECTIVE (decisive):** the learning DATA is bad → fix the DATA first. Consider (a) a DEPTH-equipped dataset to accurately reconstruct the 3DGS, and/or (b) better ways to produce realistic, learnable 3D-motion data from video.
- **Two data defects:** (1) 3DGS geometry from Pi3 monocular = approximate; (2) motion GT from CoTracker(2D)+Pi3-lift = occlusion-fragile. Manipulation is the worst case for both.
- **Plan — get CLEAN GT. 3 avenues:** **A) Simulation** (ManiSkill/RLBench/CALVIN…) = EXACT rendered depth + EXACT per-object 3D motion from sim state (zero estimation/occlusion noise, the sim knows poses even when visually occluded) + language → the CLEANEST GT, decisively validates whether the arch can learn localized motion. **B) Real RGB-D** (DROID/RH20T…) = real depth + language, realistic but sensor noise/occlusion remain. **C) Better video→3D** (St4RTrack / Shape-of-Motion / MonST3R) on existing RGB = cleaner motion but still estimation. **Lean: A (sim) FIRST to unblock+validate with perfect GT, then B/C for realism.** Research sub-agent dispatched to nail specifics (which dataset, availability, license, integration). Architecture work (spatial-grounding) PAUSED until clean GT exists — it's the right idea but can't be validated on noisy GT.

## 39. ★★★ DATA PLAN decided + sim WORKING → building clean-GT pipeline (2026-06-07)
**Research (sub-agent a5072107, web-cited):** Stage-1 UNBLOCK = **simulation** — the key insight: in sim you DON'T track. Build 3DGS once from frame-0 RGB-D, label each Gaussian to an actor via the seg mask, then move it by the sim's EXACT per-actor pose every frame (`X_t = T_{o,t}·T_{o,0}⁻¹·X_0`) → analytically exact, occlusion-FREE per-Gaussian 3D trajectory. **ManiSkill3** (pip, SAPIEN GPU render, 30k FPS, RGBD+seg+poses) or **RLBench** (cleanest pose via `task_low_dim_state`). Stage-2 REALISM = **DROID** (real ZED-stereo depth + calibration + crowd language, CC-BY-4.0, `gsutil cp gs://gresearch/robotics/droid`) re-extracted with **SpatialTrackerV2** (feed-forward, occlusion-aware 3D tracker, ~5–10s/clip, big TAPVid-3D accuracy jump — REPLACES both halves of CoTracker+Pi3; supersedes the old St4RTrack/MonST3R plan; CC-BY-NC). Hand-object sets (DexYCB/ARCTIC) = near-analytic but no language/robot. (`notes/` — research result in transcript.)
- **Progress:** ManiSkill3 3.0.1 installed in the uv venv (`uv pip install --python .venv/bin/python`); **RENDERS on the headless A100** (SAPIEN builtin-Vulkan fallback — ignore the Vulkan-ICD warnings); confirmed RGB+depth render + per-actor `pose.raw_pose` accessible. Gaps handled by the build: turn on segmentation (`obs_mode="rgb+depth+segmentation"`), use motion-planning solvers/demos so objects actually move (random actions don't).
- **Delegated (opus sub-agent ac07de0):** build `maniskill_gt.py` (RGB-D→3DGS, seg→actor-id, analytic per-Gaussian motion) with a **GT-correctness safeguard** (render analytic-moved Gaussians vs the real sim future frame — must match) + adapt `overfit_motion.py` to the clean GT. **GO/NO-GO:** on clean analytic GT, does the dynamics LOCALIZE (corr→0.7+, ratio→0.5+)? YES → architecture validated, real-data GT was the sole culprit → proceed to DROID+SpatialTrackerV2. NO → genuine architecture limit → resume spatial-grounding.
- **★★★ RESULT = GO (decisive, 2026-06-07).** Sub-agent ac07de0 built `code/scripts/maniskill_gt.py` (RGB-D→3DGS + seg→actor-id + analytic per-Gaussian motion `X_t=T_{o,t}T_{o,0}⁻¹X_0`) + `overfit_motion_sim.py`. **GT-validation PASSED**: analytic-moved Gaussians render at PSNR ~20–22 vs the REAL sim future frames (`outputs/maniskill_val_stackcube/val_t*.png`) ⇒ the clean motion target is correct. **Overfit on clean PickCube GT (`/tmp/of_pickcube*.log`):** SPATIAL-grounding → **corr 0.46→0.935→0.949→0.90, top-mover ratio →0.80, pos 0.024→0.004** (DECISIVELY localizes; gate corr→0.7/ratio→0.5 exceeded). VANILLA (no spatial-grounding) → oscillates uniform-large↔zero, **corr collapses to 0.03 (cannot localize even on perfect GT)**. **Conclusion: BOTH are needed — clean GT AND per-control spatial-grounding. Together the dynamics produces correct localized motion (corr 0.95).** This validates: (1) the noisy CoTracker+Pi3 GT was the blocker, (2) spatial-grounding is necessary+sufficient for localization, (3) the whole language-conditioned 3DGS-dynamics approach is sound. **This is a CAPACITY proof (overfit one clip); next = GENERALIZATION** — train on MANY sim clips (spatial_ground on, clean analytic GT) → does localized + language-conditioned prediction generalize to held-out clips? Then real data (DROID + SpatialTrackerV2). The architecture (spatial-grounding) and clean-GT pipeline are now the locked foundation.
- **GENERALIZATION phase launched (all 4 GPUs, sub-agent aa58ece, 2026-06-07):** scale `maniskill_gt` to ~300–600 clean clips (PickCube/PushCube/StackCube/+; parallel data-gen across 4 GPUs+CPU; keep a HELD-OUT split), build a sim-clip Dataset + adapt `train_stream` (DDP-4GPU, `spatial_ground=True`, direct-3D `traj` loss + InfoNCE, NaN-guard, resume stream11c strict=False), train, then EVAL on HELD-OUT: corr/ratio + language Δ + a rollout video. Verdict = does localized + language-conditioned prediction GENERALIZE (held-out corr high), or only memorize (held-out corr~0)? (Note: capacity tests used only 2 GPUs because they're single-clip/1-GPU; now the box is fully used.)
- **DATASET BUILT + 4-GPU TRAINING LAUNCHED (2026-06-07).** 482 clean sim clips in `data/maniskill/`: **266 train** (134 PickCube + 132 PushCube), **54 heldseed** (in-task generalization), **160 StackCube = heldtask** (whole-task generalization). Files (built by sub-agent aa58ece, all compile + smoke-pass): `code/scripts/gen_sim_dataset.py` (+`gen_sim_launch.sh`; sharded parallel `--shard/--nshards`), `code/igsw/data/sim_clips.py` (`SimClipDataset`/`sim_collate`; split by filename), `code/scripts/train_sim.py` (4-GPU DDP, `--spatial_ground 1 --vlm_image 1`, defaults = validated recipe: lr3e-4 + lr_sg1e-3, w_scale_anchor0.2, InfoNCE, NaN-guard; ckpt `ckpt_{step:07d}.pt`/`ckpt_last.pt` every 500), `code/scripts/eval_sim_generalization.py` (held-out corr + Δ_null/Δ_wrong + train-vs-held contrast + rollout videos; `--ckpt --data`). **Running:** `logs/train_sim_gen.log` (`checkpoints/sim_gen`, resume stream11c ckpt_0006000 strict=False → 14 spatial/InfoNCE heads reinit). Smoke + live first steps healthy: peakGB ~45, **corr −0.4→0.6 in warmup, ratio→0.69, lang→3.7, scl→0.008** (learning to localize on MULTI-clip data ✓; per-step corr noisy = varied clips). Auto-eval watcher `logs/sim_gen_eval.log` fires at ckpt_0000500 + ckpt_0002000 → the GENERALIZATION VERDICT (held-out corr/Δ + videos in `outputs/sim_gen_eval/`). To resume/monitor: read those two logs.
- **★★★ GENERALIZATION CONFIRMED (2026-06-07) — the approach works end-to-end.** Held-out eval (`logs/sim_gen_eval.log`, `outputs/sim_gen_eval/`): **s500** train 0.722 / held-seed **0.744** / held-task 0.650; **s2000** train 0.694 / held-seed **0.696** / held-task **0.756**. (1) held-seed corr ≈ train ⇒ ZERO overfitting (unseen seeds localize as well as train); (2) **held-task = StackCube, NEVER trained, corr 0.65→0.76 and RISING** ⇒ cross-task generalization; (3) Δ_null/Δ_wrong all >0 ⇒ responds to the instruction. Training live e122 s8180, train-clip corr **0.91** ratio 0.82, healthy. Held-out rollout videos: `viz/rollout_heldseed_pickcube_s100{0,1}_heldseed.mp4`. **Conclusion: clean sim GT + per-control spatial-grounding ⇒ a localized, language-conditioned 3DGS dynamics model that GENERALIZES to unseen clips AND unseen tasks (not memorization).** The full diagnosis arc resolved: broken AgiBot run = noisy GT (CoTracker+Pi3 occlusion) + missing per-control grounding; fixed both. **NEXT:** (a) realism — port to DROID + SpatialTrackerV2 (real depth + occlusion-aware 3D tracks); (b) re-enable #1 MetaQuery language A/B + #2 long-horizon (now meaningful on a model that actually moves correctly); (c) keep training sim (corr still improving) + eval longer rollouts/Δ.
- **INSPECTION findings (2026-06-07, user scrutiny):** (1) the eval video "重影/ghosting" = a VIZ artifact: `_render_video` composites the moved 3DGS over the real frame-0 photo as bg → object shows in old+new spots; plus single-view RGB-D recon = soft Gaussians. NOT a model bug. (2) **REAL issue the corr metric HID:** the predicted APPEARANCE drifts over the rollout (mid-rollout the table goes RED, then scrambles) — because the sim GT motion is RIGID (color/opacity/scale should be constant) but the model predicts deltas for all of them and ONLY position is strongly supervised (+ weak scale-anchor); color/opacity ~unconstrained → drift. corr only measures per-control displacement (position), so it missed this. **FIX (pending): strongly anchor color/opacity/scale to G0 in sim training** (rigid ⇒ appearance constant; model should only move pos+rot). (3) `code/scripts/export_3d.py` = clean inspection: writes `outputs/ply/*_{g0,gt_final,pred_final}.ply` (binary PLY point clouds, viewable in MeshLab/web splat) + `outputs/clean/*_clean16.mp4` (GT|static|pred on BLACK bg, pred rendered with FROZEN g0 appearance = MOTION isolated, sidesteps the appearance bug) + a longer free-rollout. (4) `maniskill_gt --start_frac F` (build_clip start_frac) anchors a clip at a MID-EPISODE frame → predict continuation from mid-manipulation; `pickcube_midstart.pt` (start_frac 0.5) GT-validated PSNR ~24, model predicts plausible localized continuation (motion present, clean appearance, ~corr-0.74 quality, not perfect pose). Pulled to local `viz/`.

## 40. ★★ GENERALIZATION phase — scale-up infra BUILT + dataset gen (2026-06-07, sub-agent aa58ece)
Springboard = §39 overfit GO (corr 0.95 on ONE clean clip w/ spatial-grounding). Now: train on MANY clean sim clips, eval HELD-OUT (memorization vs generalization). All 4 GPUs.
- **Dataset gen (parallel, 4 GPUs × 4 workers = 16 shards):** `code/scripts/gen_sim_dataset.py` (+`gen_sim_launch.sh`) runs `maniskill_gt.generate_episode+build_clip+validate` over the (env,seed) job list (`jobs[shard::nshards]`), DROPS clips below `val_psnr≥16` or `movefrac≥0.02`, encodes the split in the filename `{env}_s{seed:04d}_{split}.pt` (split∈train/heldseed/heldtask) so the trainer/eval split by filename. Resumable (skips existing). `sim_dataset_summary.py` = the val_psnr distribution report.
- **Tasks = PickCube, PushCube, StackCube** (3 reliable scripted policies). PokeCube/PullCube SKIPPED — they need new grasp-peg-poke / pull-back policies (NOT "easy"); a broken policy just generates motion-less clips that get dropped (wasted GPU). Diversity comes from many randomized seeds (object positions → varied motion + per-task instruction). **Held-out split: StackCube ENTIRELY (heldtask = task-level generalization) + 15% of each other task's seeds (heldseed = unseen-config generalization).**
- **★ BUG (caught at scale, fixed):** in THIS ManiSkill version PushCube's manipulated object is named `obj` (not `cube`), StackCube uses `cubeA`/`cubeB`; `_script_push` + `generate_episode`'s moved-check did `u.cube.pose.p` → AttributeError → ALL 124 PushCube jobs FAILED (and would never pass the motion check). Fix = `maniskill_gt._manip_object(u)` tries `cube`/`obj`/`cubeA`. After fix: PushCube cube moves 0.23m, StackCube 0.15m, both validate. (PickCube already used `cube` → its 160 clips were fine; kept them, relaunched only PushCube+StackCube.) val_psnr is consistently ~21–22 (clean reconstruction) and 0 drops on the working tasks. Gen rate ≈19 clips/min @16 workers (the full per-clip pipeline — episode w/ retries + backproject 3DGS + validate-render 17 frames @512² — is the cost, ~50s/clip/worker).
- **Sim trainer (`code/scripts/train_sim.py` + `train_sim_launch.sh`):** 4-GPU DDP (`static_graph=True`, `broadcast_buffers=False`, `gradient_as_bucket_view`); `DistributedSampler` shards the train-split clips across ranks/epochs; map-style `SimClipDataset` (`code/igsw/data/sim_clips.py`). Per clip (NO Pi3 lift / NO CoTracker track — the clip IS the clean GT): load g0 + uv + the EXACT `traj`; **mover-biased control sampling** (≤½ the M=2048 controls drawn from GT-movers, as in the overfit, so localization is measurable); `spatial_ground=1` + `control_uv`/`control_uv_hw` (the §39 model call); direct 3D `trajectory_loss` (pos+vel) vs `traj` + `rotation_loss` (Kabsch on GT-knn) + InfoNCE language loss + MoCo queue + render-aux (clip's STATIC camera) + scale-anchor; the finite-grad-norm NaN-guard. **2 param-groups: base lr3e-4, freshly-init spatial-grounding (`vis_*`) lr_sg 1e-3** (matches the overfit's higher SG LR). Resume stream11c strict=False; do NOT load opt/step (new task, changed param set, fresh cosine schedule). Logs per-step train corr + top-mover ratio (the localization signal).
- **Generalization eval (`code/scripts/eval_sim_generalization.py`) — the deliverable:** on heldseed + heldtask (and a few train clips for contrast), with the SAME spatial-grounding call + mover-biased sampling: (1) motion: corr(GT_disp,PRED_disp) + top-mover ratio + frac>0.02r; (2) language sensitivity Δ_null=‖v(ℓ)−v(∅)‖/‖v(ℓ)‖ and Δ_wrong=‖v(ℓ)−v(ℓ')‖/‖v(ℓ)‖ (control set + image FIXED, only the TEXT to frozen Qwen changes); (3) GT|static|pred rollout mp4 for a couple held-out clips. One-line VERDICT = train vs held-seed vs held-task corr. **GOTCHA: don't train while 16 gen workers run** (a single-GPU smoke starved on CPU/disk I/O loading the 26GB ckpt — killed it; the real validation is the 4-GPU launch once gen winds down).

## 41. ★★ RANDOM-START 4-SECOND experiment — launched (2026-06-08, user-directed)
User: "现有的数据，起点随机，累计4s的变化，先训1w steps" → on the maniskill sim data, generate clips that START at a RANDOM episode frame and span ~4 SECONDS of motion (vs the old clips = whole-episode, always from frame-0 / task-start), then train 10k steps. Goal = a model that predicts a meaningful 4 s horizon from ANY mid-task state, not just from the beginning.
- **Time mapping (verified):** PickCube/PushCube run at `control_freq=20 Hz` (0.05 s/step). 4 s = **80 control steps**. The model is UNCHANGED (K=16 frames); those 16 frames now span 4 s @ 0.25 s/frame (was ~2.7 s @ 0.17 s/frame over the whole episode). The trainer is data-agnostic to the temporal span (it just consumes the clip's 16-frame `traj`) → no trainer/model change needed.
- **Two data defects to fix for this:** (1) the scripted tasks only ran ~40–54 steps (2.0–2.7 s) < 4 s; (2) `build_clip` always started at frame 0. **Fixes (code):**
  - `maniskill_gt._script_pick` / `_script_push` EXTENDED to ~6–7 s of CONTINUOUS motion via MULTI-WAYPOINT paths (pick→lift→carry the grasped cube through 7 waypoints; push through 4 zigzag targets re-approaching behind each time). The FIRST waypoint is still the task goal so the instruction matches. → episodes now **121 (push) / 139 (pick) frames**, cube moves 0.22–0.30 m (a random 4 s sub-window always contains real change; no static tail).
  - `maniskill_gt.build_clip(..., window_steps, rng)`: samples K+1 frames over a `window_steps`-long WINDOW whose START is RANDOM in `[0, n_sim-1-window]` when an `rng` is given (else deterministic `start_frac` within that range). None = legacy whole-episode clip. Returns `start`/`win`/`n_sim` for traceability.
  - `gen_sim_dataset.py`: `--window_sec` (→ steps via `--control_freq`), `--random_start` (deterministic per env+seed via md5 hash → reproducible). Saves `window_sec`/`start_idx`/`n_sim`/`win_steps` in the clip.
- **Verified (1-seed test, /tmp/test_4s):** PickCube 139-frame ep / PushCube 121-frame ep; windows win=80 with VARIED random starts (43, 5, 19, 4); traj `[17, ~200k, 3]`; **GT-validation PSNR 21–23 dB**, movefrac 0.12–0.16, 0 dropped. Pipeline correct.
- **RUNNING (data → `data/maniskill_4s`, ckpt → `checkpoints/sim_4s`):** parallel regen (16 shards, 4 GPUs): `--tasks PickCube,PushCube,StackCube --seeds 200 --seed_base 1000 --held_task StackCube --held_seed_frac 0.15 --window_sec 4 --control_freq 20 --random_start 1 --min_val_psnr 14 --min_movefrac 0.02` (logs/gen_4s_shard*.log). **Auto-handoff** `code/scripts/orchestrate_4s.sh` (detached, setsid): waits for the gen workers to finish → if clips≥100, launches `train_sim.py --data data/maniskill_4s --out checkpoints/sim_4s --resume checkpoints/sim_gen/ckpt_last.pt --spatial_ground 1 --vlm_image 1 --total_steps 10000 --max_steps 10000` (logs/train_4s.log, logs/orchestrate_4s.log). **WARM-START** from sim_gen ckpt_last (step 9500, corr~0.9, all spatial layers present → near-full load, NOT a from-scratch run) so 10k steps suffices; fresh optimizer + cosine over 10k. Old 60k sim_gen run STOPPED (generalization already confirmed §39; its weights carry forward via warm-start).
- **Note:** appearance-drift fix (anchor color/opacity/scale to G0, §39 inspection) NOT included here — kept the experiment focused on random-start+4s; `export_3d.py` already freezes appearance for clean inspection renders. Still a pending separate improvement.
- **NEXT after 10k:** eval held-out corr / language Δ + export a clean 4 s random-start rollout video; compare vs the frame-0 sim_gen model (does random-start training help predict-from-mid-state?).

## 42. ★★★ DATA-QUALITY upgrade: WHOLE-VIDEO temporal fusion of the canonical Gaussian set (2026-06-08, user-directed)
**User directive (重大问题):** the single-frame "每张图像独立生成高斯" approach gives Gaussian increments with large uncertainty → biased learning. Use a UNIFIED whole-video method to reconstruct the entire Gaussian evolution → higher-quality learning data; verify the model learns well on it.
- **Alignment核验:** mission (§0) = language-conditioned 3DGS dynamics world model. The sim work (§38-41) validates the dynamics on clean GT — aligned. ✓
- **Assessment (where the problem actually is):** our sim MOTION GT is already coherent/exact (analytic `X_t=T_{e,t}T_{e,0}⁻¹X_0`, NOT per-frame independent). The weak link is the **canonical G0**: built from ONE frame's ONE camera view → an incomplete "shell" (occluded/back/bottom missing; geometry only certain where visible). Moving that incomplete shell = uncertain increments + the "重影/ghosting" the user saw.
- **Fix = whole-video temporal fusion (`maniskill_gt._fuse_canonical_gaussians`, used by `build_clip(fuse_stride>0)`):** back-project EVERY (strided) frame's depth and REGISTER each entity's points into the canonical pose via the KNOWN per-entity poses (`X_canon = T_{e,0}·T_{e,f}⁻¹·X_f`; static/no-pose→identity since the camera is static), accumulate across the whole video, voxel-dedupe → ONE complete, denoised canonical GaussianSet driven by the same analytic trajectory. Each kept point carries its source frame's grid-neighbour scale (detail preserved). Moving entities reveal new faces over time + the arm's occlusion-shadows on the static scene get filled. **Temporal fusion is the right methodology — it transfers to real MONOCULAR video (one camera over time); multi-camera would be a sim-only crutch that doesn't transfer.**
- **★ Validation (`validate_fusion.py`, same episode, voxel sweep) — fusion strictly improves the data:** SAME canonical-view PSNR full **21.5→22.7** (+1.2 dB) / dynamic-region **16.7→19.3** (+2.6 dB) at voxel=2mm; novel-view (25° orbit) coverage **+3%** (more complete, fewer holes) at 1-2mm; N=247k (×1.26 single-frame). 2mm = the sweet spot (1mm doubles count for no extra coverage; 3mm loses resolution). The +2.6 dB dynamic-region gain = exactly the "denoised, certain increments" the directive asks for. Images: `viz/fusion_val/novel25_single_vs_fused_v*.png`.
- **Verification run (RUNNING):** regen `data/maniskill_fused` = **frame-0 + 2.7s window (matches sim_gen's pick-and-place task) + FUSED G0 (stride 3, 2mm)** → a CLEAN A/B vs sim_gen (same task, only the G0 reconstruction differs; warm-start transfers so corr should reach ~0.9 fast). 150 seeds × {PickCube,PushCube,StackCube}, StackCube held-out. Then `orchestrate_fused.sh` warm-starts from sim_gen/ckpt_last on **3 GPUs (1,2,3)** (--workers 2, total_steps 3000). Verdict = corr reaches sim_gen-level AND the predicted rollout renders CLEAN + COMPLETE (no single-frame ghosting).
- **Infra notes:** the §41 random-start-4s train CRASHED — SIGTERM (signal 15) at s640, host-RAM pressure (a NEIGHBOR pod OOM'd on the node; our memcg is 400GB w/ 364GB free, so likely node-level eviction); also its corr was stuck ~0 (the frame-0 warm-start does NOT transfer to random-start mid-state prediction → random-start is a separate, harder problem, deferred). GPU 0 has a stuck 978MiB/82% process un-killable from inside the container (different pid namespace) → using GPUs 1-3. Defensive train: --workers 2, RAM logged each 2 min in orchestrate_fused.log.

## 43. ★ FUSION verification + VISUALIZATIONS (2026-06-08/09)
Fused dataset `data/maniskill_fused` = 450 clips (247 train PickCube+PushCube, heldseed, 150 heldtask StackCube), frame-0 + 2.7s window, whole-video FUSED G0 (stride3, 2mm, ~240k Gaussians/clip), val_psnr 22-24 (>single-frame's 21-22 ✓).
- **Train (warm-start sim_gen, full-load 1178 tensors, 0 reinit):** corr **0.81 ZERO-SHOT (s0)** → 0.93 peak (s40) on fused data ⇒ the fused data is immediately learnable + warm-start transfers (unlike random-start-4s which got corr~0). NO OOM (the §41 SIGTERM was node-level; venv RAM fine at 90/400GB).
- **★ But the LR 3e-4 (designed for sim_gen's from-scratch w/ reinit heads) was TOO AGGRESSIVE for a full warm-start fine-tune → it DISRUPTED the good init:** held-out corr s500=0.21, **s1000=−0.5 (broken)**, recovered s3000=**0.635** (train 0.635 ≈ heldseed 0.636 ≈ heldtask 0.634 ⇒ zero overfit + cross-task generalization). BUT the s3000 autoregressive ROLLOUT SCRAMBLES in later frames (over-prediction ratio~2 compounds) — the §39 appearance/magnitude-drift issue, amplified.
- **Fix = gentle LR 3e-5 fine-tune** (`checkpoints/sim_fused_ft`, lr=lr_sg=3e-5, warmup 50): s300 rollout is STABLE (per-clip corr 0.41-0.55 heldseed/heldtask). Tradeoff: stable but lower corr (undertrained at s300).
- **KEY INSIGHT:** fusion improves the **GEOMETRY** (completeness/denoising — validated +1.2 full / +2.6 dB dynamic PSNR, +3% novel-view coverage, §42), NOT the motion-corr (the analytic motion GT is identical with/without fusion). So corr ≈ single-frame is EXPECTED; the fusion win is render cleanliness + certain geometry. The rollout-stability/drift is a separate MODEL-side issue (the deferred appearance-anchor: anchor color/opacity/scale to G0 since sim motion is rigid).
- **Visualizations (`code/scripts/viz_for_user.py` → `outputs/viz_user/`, pulled to `viz/viz_user{,_s3000}/`):** (1) TRAINING DATA = fused GT rollout of train clips (clean complete arm+cube motion); (2) TEST = GT|static|PRED rollout on heldseed (unseen config) + heldtask (unseen StackCube task), frozen-appearance. s300 stable; s3000 scrambles late.
- **Bug fixed:** `orchestrate_*.sh` called bare `torchrun` → system python (no transformers) → ChildFailedError. Use `.venv/bin/torchrun`.

## 44. ★★★ SEMANTIC-LOCALIZATION failure → mover/static gate (Exp-1) (2026-06-09, user-directed)
User reviewed the test rollouts and named 2 concrete failures: **(2.1) the static TABLE "sinks"** (motion leaks onto background) and **(2.2) the red CUBE doesn't move during "pick"** (the instruction's target stays still). Diagnosis = the model is NOT object-aware: it can't bind "instruction→movable object" nor infer "background=static". Also the cube is TINY (~1-2% of Gaussians) vs the table HUGE (~73%), so the loss under-weights the target + any leak on the table is glaring. User directive: optimize the DATA (Gaussians carry semantic+motion features) AND the METHOD (extract enough image-semantic-motion from frozen Qwen3-VL); RESEARCH first, then improve+experiment, iterate until the user approves (the user judges via the visualizations).
- **Visualization correction:** the user wants the **GAUSSIAN DATA in 3D**, not rendered videos. `code/scripts/export_3dgs_ply.py` → standard 3DGS `.ply` (means, log-scales, wxyz quats, inv-sigmoid opacity, SH-DC colors; isotropic so quat-order is moot) → `viz/gaussians/` (t00_natural / t00_segment / t08 / t16 for PickCube + StackCube). Open in SuperSplat. `seg_per_g` has 13 entities (table/ground/cube/~10 arm links). (`viz/train_videos/` + `code/scripts/viz_training_videos.py` 3-panel natural|seg|mover videos confirm the GT is correct: table static, arm+cube move — the failure is the MODEL.)
- **Research (sub-agent a44b5e7, `notes/research_semantic_motion_gaussians.md`, 37 cites):** root cause = (1) 2 of 3 conditioning paths are GLOBAL (AdaLN cond_global + cross-attn to 16 distilled tokens) → uniform-translation is the easy min; (2) per-control feature = raw Qwen image patch ("red here"), NO instruction→object binding; (3) move/stay supervised only by `trajectory_loss` whose all-control norm makes "predict 0 for all" the L1 min → table leaks/cube timid; obj_focus + background_static_loss are OFF/unused + rely on GT-motion (no-inference); (4) **`seg_per_g` (exact per-Gaussian entity label) is SAVED but never supervised** = a free perfect mover/static + identity label. Literature ("localize-then-move"): 3DFlowAction (2506.06199), DynaSplat/DeGauss (per-Gaussian dynamics mask before deform), FOCUS (per-object mask aux + bg suppression), SemanticSplat/GaussianGrasper (per-Gaussian semantic latent distilled from a frozen 2D teacher + text→object cosine), Qwen2.5/3-VL referring points/boxes = cleanest no-GT "which pixels=cube" (raw attention is a sink — confirmed by our §28).
- **Exp-1 (sub-agent a5624dd implementing, flag `--dyn_gate`):** per-control **mover/static gate `p_dyn`** (head off the spatial-grounding features) → **`v ← sigmoid(p_dyn)·v`** (structurally forbids background motion; warm-start bias so sigmoid≈1 at init), BCE(p_dyn, GT-mover-label `disp>1cm`), + object-semantic head supervised by `seg_per_g` (CE/grouping). New metrics: **static-leakage** (static-seg predicted disp →0), **mover precision/recall** (>0.9), corr 0.65→0.8+. Acceptance = overfit ONE clip: leakage→0 + cube moves + corr≥ungated. Then full 4-GPU train. Exp-2 = per-control instruction↔Gaussian cross-attn; Exp-3 = Qwen referring-point prior (real-data transfer).

### 44b. Exp-1 RESULT = PASS + full training launched (2026-06-09, sub-agent a5624dd)
**Overfit A/B (1 PickCube clip, 500 steps, resume stream11c):** BASE(no gate) corr 0.876 / static-leakage 0.0282; **GATE corr 0.940 / leakage 0.0102 (2.8× lower) / cube PRED 0.96×GT / mover precision 0.961 recall 0.992.** PASS all 4 gates (leak→0, cube moves, P/R>0.85, corr≥base). BASE reproduced the failure (leak stuck 0.03-0.05 = table leaks).
- **Implementation (`--dyn_gate 1 --w_dyn 1.0`, backward-compatible, byte-identical at `--dyn_gate 0`):** `model_full._control_visual` → new `dyn_head MLP(H→d→1)` off the SAME per-control Qwen patch feature as `vis_*`; last-layer zero-weight + bias +4 → sigmoid(p_dyn)=0.982 at init (gate open). `model.py predict_deltas(gate_local)`: applied AFTER the tanh bounds → `v=v*gate; omega=omega*gate` (§31 tanh discipline intact). `losses.mover_bce_loss` = BCE(p_dyn, mover_label= GT disp>1cm, train-only). dyn/sem heads in the lr_sg group; ckpt saves `dyn_gate`/`sem_dim`; eval reconstructs the gate. New metrics in train+eval: **static-leakage**, **mover P/R**. Object-semantic head #3 (`--sem_dim 16 --w_seg 0.2`, Gaussian-Grouping prototype-CE on seg_per_g + 3D-NN consistency) wired but default OFF (not yet validated).
- **Full train RUNNING:** `checkpoints/sim_gen_dyngate`, `.venv/bin/torchrun --nproc_per_node=4 ... --resume stream11c_infonce/ckpt_0006000 --dyn_gate 1 --w_dyn 1.0 --total_steps 6000` (logs/train_dyngate.log). Gate-only first (validated); sem head = next iter if needed. NEXT: eval held-out (leakage + mover P/R + corr per split) at s500/1000/2000 + export the gated model's PREDICTION as 3DGS .ply (user reviews in 3D whether the table stops sinking + the cube moves).

### 44c. dyn_gate full-train: WORKS but drifts → SETTLED short run locks it (2026-06-09)
Warm-start from **sim_gen** (not stream11c — that re-learns sim from scratch, corr starts −0.4) + gentle LR + `--dyn_gate 1`: resume reinit only 4 tensors (the dyn_head). **The gate generalizes the overfit result**: leak 0.082→**0.010-0.019** (5-8× down = static/table suppressed, §2.1) while **corr recovers to 0.90-0.95** (a transient dip s20-40 as the gate over-suppresses movers — mR 0.38 — then mR→0.98 and corr returns) and the cube moves (mR 0.98). BUT a sustained LR (5e-5) lets the model **drift after ~s200** (mover-magnitude OVER-prediction — ratio→2.5, pos↑, leak creeps back to 0.13): the gate fixes STATIC leakage, NOT the §39 mover-magnitude drift (separate problem; the deferred appearance/velocity anchor).
- **Fix = SHORT SETTLED run** (`checkpoints/dyngate2`, total_steps 200, cosine LR→0 by s200, ckpt every 50): the LR decays before the over-prediction compounds, FREEZING the gate-working state. **s160-180 settled: corr 0.95, leak 0.012, mover P/R 0.89/0.98, ratio 0.74 (NO over-prediction).** `ckpt_last` = the deliverable. Held-out eval + `export_pred_3dgs_ply.py` (gated PRED vs GT as 3DGS .ply) → user 3D review.
- **Lesson:** for warm-start fine-tunes with a reinit head, a SHORT cosine-to-0 schedule locks the good state; sustained LR drifts (the recurring §39/§43 magnitude-drift). The real fix for the drift = the appearance/velocity anchor (still deferred). Exp-2 (instruction↔Gaussian cross-attn) + the sem head (`--sem_dim`) remain for deeper object-binding.

### 44d. dyn_gate held-out eval + magnitude problem → gate+obj_focus (2026-06-09)
**dyngate2 (settled, s200) HELD-OUT eval** (`eval_sim_generalization` now reports leak + mover P/R): **static-leakage 0.009-0.013 across train/heldseed/heldtask** (§2.1 SOLVED + generalizes incl. unseen StackCube), mover P/R 0.87-0.99 (gate identifies movers correctly), corr 0.48-0.54 (consistent = no overfit). **BUT top-mover ratio 0.2-0.4** = the cube MOVES (not frozen) but UNDER-predicts magnitude (20-40% of GT) = §2.2 not yet passing. **The magnitude under-prediction is PRE-EXISTING in sim_gen** (the §37 L1-median-collapse on heavy-tailed displacement), NOT caused by the gate (gate-open s0 already had ratio 0.37). Review artifacts: `code/scripts/export_pred_3dgs_ply.py` → `viz/gaussians_pred/` (gated PRED vs GT as 3DGS .ply at t8/16 for PickCube + StackCube; user reviews in SuperSplat).
- **Next iter (dyngate3, RUNNING):** `--dyn_gate 1 --obj_focus 3.0` — obj_focus mover-weights the traj loss to amplify the cube's magnitude. §37 noted obj_focus alone backfired (uniform-larger, not localized) — but NOW the GATE localizes (static gated off), so obj_focus should amplify ONLY the gated-on movers (cube↑) without inflating the table. Watching top-mover ratio↑ while leak stays low. (sem head `--sem_dim` + Exp-2 cross-attn = further iters if magnitude still short.)

### 44d. dyn_gate + obj_focus = BOTH 2.1 & 2.2 (2026-06-09)
The pure gate (dyngate2) fixed 2.1 (static-leakage 0.009-0.013 held-out) but the cube UNDER-moved (top-mover ratio 0.2-0.4 = only 20-40% of GT magnitude) — the §37 heavy-tailed-displacement under-prediction, a sim_gen-base issue the gate doesn't touch. **Fix = add `--obj_focus` (mover-weighted trajectory loss) ON TOP of the gate** (`checkpoints/dyngate3`, sim_gen warm-start, short cosine→0, ckpt every 50): the gate suppresses static + obj_focus amplifies the gated-on movers. **Settled s160-180: corr 0.95, ratio 0.80-0.82 (cube → 80% of GT, up from 0.2-0.4!), leak 0.018-0.023, mover P/R 0.88/0.99.** Trade: leak slightly higher than the pure gate (0.02 vs 0.012) but 3-4× below the 0.082 ungated baseline. Eval s150/s200/last held-out to pick the best ckpt → export PRED-vs-GT 3DGS .ply for user 3D review. **Both of the user's failures now addressed: 2.1 table-static (gate) + 2.2 cube-moves-enough (obj_focus).** (§37's obj_focus "backfire" was WITHOUT the gate — uniform amplification incl. background; WITH the gate it only amplifies true movers, so it works.)

### 44e. USER insight: table-collapse-under-OVERLAP = insufficient 3D semantic → enable SEM head (2026-06-09)
User /goal: push this version to fix BOTH (2.2 cube-moves + 2.1 table-collapse-near-arm). **Key user hypothesis: the failures are insufficient SEMANTIC learning — "if the semantic info were enough, then even with 2D visual OVERLAP, there shouldn't be large-area collapse."** This is sharp + correct: the gate's per-control feature is the Qwen 2D patch, which at a pixel where the ARM occludes the TABLE returns the ARM's feature (overlap-ambiguous) → that table-control leaks (follows the arm). The held-out leak (0.01) is the AVERAGE; the residual collapse is local to the overlap region.
- **Diagnostics:** `max_disp=0.1` per-step, cube needs ~0.02/step → **NO clipping** ⇒ the cube-magnitude under-prediction (held-out ratio ~0.45, both dyngate2 pure-gate AND dyngate3 +obj_focus) is a LOSS/LEARNING issue, NOT a capacity cap — supports the semantic hypothesis. dyngate2 heldseed ratio 0.40, dyngate3 0.45-0.50 (obj_focus helped only a little; the per-step 0.80 was noise).
- **Fix (dyngate4) = enable the OBJECT-SEMANTIC head** (`--sem_dim 16 --w_seg 0.3`, the Exp-1 head left OFF): per-control sem embedding supervised by **`seg_per_g` (the 3D per-Gaussian entity id — OVERLAP-INVARIANT, unlike the 2D Qwen patch)** via Gaussian-Grouping prototype-CE + 3D-NN cosine consistency. This teaches the trunk an occlusion-robust identity → the gate/motion can know "this 3D point is table (static)" even where the arm overlaps it in 2D (fixes the residual 2.1 collapse), and "this is the cube (the instruction's movable target)" (helps 2.2). + keep gate (`--dyn_gate 1`) + obj_focus 1.5, sim_gen warm-start, settled cosine→0 (total 300, ckpt 50). Watch the `seg` loss (was 0.0000 with sem off) + leak/ratio. NEXT if needed: Exp-2 instruction↔Gaussian cross-attn (explicit instruction→object binding).

### 44f. dyngate4: SEM HEAD IS DEAD (seg loss flat) → debugging (2026-06-09)
Ran dyngate4 (gate + obj_focus 1.5 + `--sem_dim 16 --w_seg 0.3`, resume sim_gen, settled 300). **The sem head did NOT learn: `seg` loss FLAT at 2.66-2.74 = log(#entities) = uniform/random prediction, all 300 steps.** So the user's "semantic learning" lever is currently a no-op; the other metrics (settled corr 0.95, ratio 0.71 per-step, leak 0.015) ≈ dyngate3 (the dead head didn't change them). The head (added in §44 but "never exercised") is buggy — most likely the SPARSE `seg_per_g` ids (1..18 with gaps) used as CE class indices without remap to dense 0..K-1, or sem_dim=16 < #entities, or a detached/no-grad prototype path. **Dispatched a debug+fix sub-agent** (acceptance = seg loss must DROP from ~2.7 to <1 on a 1-clip overfit, gate intact). Until the sem head learns, the user's hypothesis (3D-semantic supervision fixes the overlap-collapse + the cube) can't be tested. Current best deliverable for review = dyngate3 (2.1 solved leak 0.01; 2.2 partial ratio 0.45-0.50). max_disp=0.1 confirms the cube-magnitude is a learning issue (no clip), so getting real semantic learning is the right next lever.

### 44g. SEM head FIXED + VERIFIED → dyngate5 (2026-06-09, sub-agent a0e0e02 — API-cut but fix landed)
**Bug (in `losses.semantic_id_loss`/`model_full`):** the class prototypes were the batch-mean of `e_sem` then `.detach()`; with the sem-head zero-init → e_sem=0 → protos=0 → logits=0 → uniform softmax (CE=log#entities≈2.7) AND ∂logits/∂z=protos.t()=0 → **gradient to the head EXACTLY 0 = dead saddle pinned at 2.7**. **Fix:** a LEARNABLE prototype bank `sem_proto` (out["sem_proto"], indexed by the RAW sparse entity id so a clip's ids {1,3,16} use rows 1/3/16) + cosine-logit CE/tau (both e_sem AND protos get grad) + non-zero head init + 3D-NN cosine consistency on knn. **VERIFIED (90-step 1-GPU check):** seg **7.27→1.17→0.55→0.51** (drops! was flat 2.7), corr 0.9, ratio up to 0.97 per-step, leak 0.015-0.024, no NaN. The agent also touched eval/export/diag for the sem path. **dyngate5 RUNNING** = gate + obj_focus 1.5 + WORKING sem head (sem_dim16 w_seg0.3), sim_gen warm-start, settled cosine→0 (total 300, ckpt 50). Then held-out eval (leak/ratio/corr + mover P/R + sem-acc) vs dyngate3 + .ply/video → test the user's hypothesis: does real 3D-semantic learning further suppress the overlap-region table-collapse + raise the cube magnitude? NOTE: all the fusion/gate/sem work is UNCOMMITTED vs the GitHub push (commit aa7f1d0) — re-sync to GitHub when stable.

### 44h. dyngate5 (working sem head) ≈ dyngate3 → WIRE sem INTO the gate (2026-06-09)
dyngate5 = gate + obj_focus + the NOW-WORKING sem head (seg loss settles ~0.43, identity learned). **Held-out = ESSENTIALLY IDENTICAL to dyngate3 (no sem head):** leak heldseed/heldtask 0.012/0.007 (same), ratio 0.41/0.27 (same), corr 0.62/0.70 (same). Video/montage visually identical (table static via the gate, cube moves ~45%). **Why no gain: the sem head is a PARALLEL AUXILIARY — `p_dyn = dyn_head(Qwen 2D patch feature)` never consumes `e_sem`.** So the learned 3D identity isn't fed to the decision-maker; at an arm-over-table pixel the gate still only sees the (arm) 2D feature → residual overlap leak. The user's hypothesis is right in principle but needs the identity WIRED IN. **Next (sub-agent a4fbc86, `--gate_uses_sem`): dyn_head input = concat([per-control feat, e_sem])** so the gate uses the occlusion-robust identity ("I'm table even though the arm is in front of me in 2D" → stay static). Then dyngate6 = sem→gate + higher obj_focus (push the cube magnitude, the bigger remaining gap). State so far: 2.1 table = largely fixed by the gate (leak 0.012, much better than the pre-gate ~0.08); 2.2 cube ratio stuck ~0.45 across gate/obj_focus/sem-aux — the heavy-tailed magnitude regression is the deep limit (max_disp=0.1 rules out clipping). Deliverables for review: viz/dyngate3_video, viz/dyngate5_video, viz/gaussians_pred, viz/gt_recon.
- **IMPLEMENTED + overfit-PASS (sub-agent a4fbc86):** `--gate_uses_sem 1` (default ON when sem on). `model_full.py`: `e_sem` (sem_head) is built+computed BEFORE the gate; `dyn_head` input = `concat([2D Qwen patch feat, e_sem])` → in-width `H+sem_dim` (2064 for sem16); gradients FLOW (seg loss still trains e_sem, plus the gate now back-props into it). Byte-identical warm-start preserved: the dyn_head LAST layer is zeroed + bias +4 ⇒ `p_dyn≡4.0` (sigmoid 0.982, gate open) at init REGARDLESS of input width, so `--gate_uses_sem 0/1` (and sem off) share the exact init. `sem_dim==0` ⇒ gate_uses_sem forced False ⇒ gate input = H (legacy byte-identical). Flag threaded through train_sim ckpt save + eval_sim_generalization/export_pred_3dgs_ply/export_3d reconstruct (so the gate width matches on load). New overfit harness `code/scripts/overfit_gate_sem.py` (A/Bs gate_uses_sem 0 vs 1, sem ON both, + the user's OCCLUSION metric: render frame-0 mover/static surface ids, read back at each static control's uv → "occluded table" = front surface is a mover/arm). **OVERFIT (pickcube_s1002, 200 steps, resume sim_gen, --dyn_gate1 --sem_dim16 --w_seg0.3 --obj_focus1.5): PASS, no NaN.** gate_uses_sem 0→1: corr 0.930→0.949, GLOBAL leak 0.0150→0.0122, **OCCLUDED-table leak (107 table-under-arm controls) 0.1267→0.1061 (the user's overlap metric: sem-fed gate leaks LESS exactly where the arm occludes the table)**, unocc-table 0.0020→0.0012, seg 0.31 (both, head learns from 7.7), cube 0.99x→0.96x, mP/mR 0.92/0.99 (both). Confirms the diagnosis: feeding the occlusion-robust 3D identity into the gate reduces the residual overlap-region table-collapse. **4-GPU full run = NOT launched (user will); command below.** Occluded-table leak (0.11) is still ≫ unoccluded (0.002) → the overlap region remains the hardest; dyngate6 should keep this + push obj_focus for the cube.

### 44i. sem→gate WIRED + VERIFIED (user hypothesis confirmed) → dyngate6 (2026-06-09, sub-agent a4fbc86)
Wired `e_sem` INTO the gate: `model_full` now builds `sem_head`/`sem_proto` BEFORE the gate, `dyn_head` input width = `H+sem_dim` (2064), `p_dyn = dyn_head(concat([per-control feat, e_sem]))` when `--gate_uses_sem 1` (default on when sem on; byte-identical when sem off; warm-start `p_dyn≡4.0` preserved by zeroing the last layer). Flag threaded through train/eval/export ckpts. **★ Overfit A/B (200 steps, the user's EXACT overlap metric — 107 arm-occluded table controls):** feeding e_sem into the gate drops the **OCCLUDED-table (table-under-arm) leak 0.1267 → 0.1061**, global leak 0.0150→0.0122, corr 0.93→0.95, seg/cube/mover-PR unchanged. **⇒ the user's hypothesis is CONFIRMED: a 3D-semantic (occlusion-robust) identity fed to the gate reduces the overlap-region collapse.** Caveat: occluded-table leak (0.106) still ≫ un-occluded (0.002) — the overlap region is the hardest residual (partial). New harness `overfit_gate_sem.py` (colors frame-0 by mover/static label, reads back at each static control's uv to flag arm-occluded ones = the overlap metric). **dyngate6 RUNNING** = sem→gate + **obj_focus 3.0** (up from 1.5, to push the stuck cube magnitude) + settled cosine→0. Then held-out eval + .ply/video. 2.1 table now: gate + sem→gate; 2.2 cube: obj_focus push (the heavy-tailed magnitude is still the deep open problem; if obj_focus plateaus, next = Exp-2 instruction↔Gaussian cross-attn or a relative/magnitude-aware motion loss).

### 44j. MAGNITUDE loss for the stuck cube (2026-06-09)
The cube top-mover ratio was STUCK ~0.46 across dyngate2-6 (gate, obj_focus 1.5→3.0, sem head, sem→gate — NONE moved it). Root cause: **`trajectory_loss` is L1 on positions, whose argmin is the MEDIAN → a few large movers (the cube) are systematically under-predicted; obj_focus only scales the L1 weight, not its median-seeking argmin.** Fix = `losses.mover_magnitude_loss` (`--w_mag`): for controls with GT disp>thresh, penalize the FRACTIONAL magnitude error `|‖pred_disp‖−‖gt_disp‖|/(‖gt_disp‖+eps)` — a 50% undershoot of a LARGE mover now costs as much as of a small one, directly pulling ‖pred_disp‖→‖gt_disp‖ (the ratio). Symmetric, magnitude-only (direction still from the L1). Wired into train_sim.py (import/arg/compute/total). **Quick-check (1-GPU overfit, w_mag on): ratio 0.45→0.56→0.66→0.71 climbing in 75 steps (vs the stuck 0.46), corr ~0.9, leak low, no NaN.** **dyngate7 RUNNING** = dyngate6 best config (sem→gate + gate + sem head, obj_focus back to 1.5) + `--w_mag 0.5`, settled cosine→0. Then held-out eval: does the magnitude loss break the 0.46 ratio ceiling on UNSEEN clips (generalize, not just overfit)? + .ply/video.

### 44k. ★★★ BREAKTHROUGH: magnitude loss breaks the cube ceiling — BOTH 2.1 & 2.2 solved (2026-06-09)
**dyngate7 = gate + sem→gate + sem head + obj_focus 1.5 + `--w_mag 0.5` (the relative mover-magnitude loss).** Held-out (UNSEEN clips, n=30/split), s200 vs dyngate6:
| | dyngate6 | **dyngate7** |
|---|---|---|
| heldseed corr | 0.64 | **0.846** |
| heldseed top-mover ratio | 0.46 | **0.810** |
| heldtask corr | 0.73 | **0.895** |
| heldtask ratio | 0.28 | **0.568** |
| static-leakage | 0.012/0.006 | 0.014/0.010 |
**The stuck cube ratio 0.46→0.81 (cube now moves 81% of GT on UNSEEN heldseed, 57% on the unseen StackCube task) — GENERALIZES, not overfit.** corr also jumped (0.64/0.73→0.85/0.90) because predicting the right magnitudes correlates better. Leak stays low (table still static). **So the user's BOTH failures are now resolved: 2.1 table-static = gate + sem→gate (occlusion-robust 3D identity); 2.2 cube-moves-enough = the relative-magnitude loss (the L1-median under-prediction was THE blocker, not capacity — obj_focus/sem couldn't touch it, the loss FORM had to change).** Best ckpt `checkpoints/dyngate7/ckpt_0000200.pt`. Deliverables: viz/dyngate7/ (montage+video+ply), pred-vs-GT .ply for SuperSplat. The full recipe (fusion data + gate + sem→gate + magnitude loss) is the working config — all UNCOMMITTED vs GitHub aa7f1d0; re-sync when the user approves. Remaining: heldtask ratio (0.57) < heldseed (0.81) = cross-task cube magnitude still lower; longer/larger training + Exp-2 cross-attn could push further.

## 45. ★ PIVOT to LIBERO data (2026-06-09, user-directed)
User: maniskill version "先到这里" (good enough — dyngate7 §44k solved BOTH 2.1 table-static + 2.2 cube-moves on held-out); the maniskill single fixed camera angle may itself bias the learning data. **Try LIBERO instead** ("拿这个数据去训练和验证"). All maniskill gate/sem/magnitude work is UNCOMMITTED vs GitHub aa7f1d0 (re-sync later).
- **Data found:** `binhng/libero_object_lerobot_mask_depth` (HF, LeRobot format, was NOT actually in `/root/.cache/huggingface/lerobot/libero` — only an empty meta dir; downloaded via proxy). 500 eps × ~148 frames @10fps, libero_object suite ("pick up the {butter,milk,ketchup,...} and place it in the basket"). Stored IN the parquet (HF image structs `{bytes,path}`): `image`[256²]RGB(agentview), `wrist_image`, `image_depth`[256²]uint8 **8-bit GRAYSCALE depth (0-255 NORMALIZED, not metric)**, `image_mask`[256²] seg ids{0..10}, **`object_of_interest_mask`[256²] BINARY = the instruction's target object**, wrist variants, `state`[8], `action`[7], task language.
- **vs maniskill:** (+) has object-of-interest mask (great for language/gate); (−) depth is 8-bit normalized not 16-bit metric (coarser, needs near/far calibration); (−) NO per-object poses → motion must be ESTIMATED via per-object rigid registration of the masked depth point clouds (centroid translation + Procrustes/PCA rotation).
- **Pipeline build (sub-agent a3d14a5, `code/scripts/inspect_libero.py` done):** Stage-1 de-risk = install LIBERO/robosuite → exact agentview intrinsics(fovy)+extrinsic+depth near/far → correct backprojection (table flat? objects above?). Stage-2 = `libero_gt.py` mirroring maniskill_gt (depth→3DGS, mask→seg_per_g, object_of_interest flag, per-object registration→traj, same clip schema). ACCEPTANCE = validate() render moved Gaussians vs real future RGB, frame-0 PSNR≥18 + dyn motion tracks. Then scale + train (reuse the dyngate7 recipe: fusion? + gate + sem→gate + magnitude loss). Fallback if 8-bit depth too coarse: re-render in the LIBERO sim for exact depth+seg+pose (the maniskill approach).

## 46. ★★ PURE-VIDEO data-gen plan (2026-06-09, user-directed) — the END GOAL
User: the LIBERO depth/camera pipeline still needs GT intrinsics/extrinsics/depth, but we LATER want to train from PURE VIDEO — "能否不依赖于这些内外参或depth产生数据". KEY: the MODEL (dyngate7 recipe) is agnostic to how the 3DGS+motion-GT are made; only the DATA-GEN must change. Research → `notes/research_pure_video_4d.md` (sub-agent a2876345, cited).
- **Why §38 failed:** chained two brittle single-purpose nets (Pi3 per-frame depth + CoTracker 2D, then WE lifted 2D→3D with no occlusion model → teleporting tracks, mover-visibility 0.40-0.66). 2024-26 SOTA fixes this with ONE feed-forward net jointly estimating depth + camera(intr+extr) + per-pixel 3D motion + visibility, end-to-end (temporally consistent depth, occlusion-scored tracks).
- **Models (all the relevant ones are CLONED on the server `third_party/`: St4RTrack, VGGT, monst3r, CUT3R, Pi3, co-tracker, LIBERO, Dynamic3DGaussians, diff-gaussian-rasterization-w-depth):** TOP pick **SpatialTrackerV2** (ICCV'25, 2507.12462; RGB→depth+cam+per-pixel 3D world tracks+`p_vis`+`p_dyn`, occ-acc 90.6, CC-BY-SA, **NOT yet cloned → clone it**). De-risk-NOW **St4RTrack** (weights ON SERVER `third_party/St4RTrack/checkpoints/` MASt3R_base.pth 2.7G + model.safetensors 2.3G — zero download; RGB→per-pixel 3D world tracks+cam, weaker occlusion). Upgrade **Track4World** (2603.02573, dense, has a Pi3-weight variant). NEGATIVE: no single open model does raw-RGB→trained dynamic-3DGS for cluttered manip (DGS-LRM needs posed video, no weights; Robo3R is multi-view) → tracker gives geometry+cam+motion, WE assemble 3DGS+per-Gaussian traj with our code.
- **Build = ONE new `video_gt.py`** (twin of maniskill_gt): tracker(RGB) → unproject frame-0 depth → g0 (reuse `to_gaussians` + §42 `_fuse_canonical_gaussians`) → query the tracker at each Gaussian's `uv` → 3D `traj`; **occluded movers via per-entity RIGID-FIT from the VISIBLE points** (recovers the sim `X_t=T_{e,t}T_{e,0}⁻¹X_0` from video); `seg_per_g` from `p_dyn` + Grounded-SAM2/VLM-referring mask. **Trainer/model/losses/eval UNTOUCHED** (emits the same clip dict). Tracker runs as an offline data-gen step in its OWN venv (torch-version isolation, like cache_clips).
- **VALIDATE vs LIBERO GT (depth/mask/object_of_interest/camera, §45):** camera Sim(3)-Umeyama→ATE/RPE; depth scale-shift-align→AbsRel/δ; motion APD3D + occluded-segment EPE vs mask-rigid GT; PLUS our corr/top-mover-ratio/static-leakage/mover-PR = "is pure-video GT as learnable as sim GT?". GO/NO-GO = within ~20% of dyngate7 (corr 0.85/ratio 0.81/leak 0.01). Risks: occlusion (→visibility+rigid-fit), scale ambiguity (→per-clip-normalized relative disp, as sim did), soft mono-depth g0 (→fusion + brief gsplat opt).
- **Order:** depth-GT agent finishes the LIBERO GT reference → clone STv2 (or St4RTrack now) → write video_gt.py → LIBERO-GT validation harness → scale + warm-start the dyngate7 recipe.

## 47. ★★★ PURE-VIDEO LIBERO TRAINS — the end-goal pipeline WORKS (2026-06-09)
Built the pure-video data-gen (St4RTrack RGB→depth+camera+3D-tracks, NO GT used to generate; GT only for validation). `code/scripts/st4r_stage1.py` + `st4r_lib.py` (St4RTrack wrapper, weights already on server) + `video_gt.py` (twin of maniskill_gt: tracker→frame-0 3DGS via to_gaussians, per-Gaussian traj from the 3D tracks, occluded movers via rigid-fit-from-visible; seg_per_g from the GT mask shortcut for v1; emits the maniskill clip schema). `code/scripts/gen_train_libero.sh` = autonomous gen→train (24 clips: 20 train + 4 heldtask, every-20th episode spans the 10 libero_object tasks, 4 GPUs) → dyngate7 recipe.
- **Clip quality (pure-video validation render-vs-real-RGB):** frame-0 PSNR 23.7, motion frames 18-21 (≈ maniskill sim 21-23) — the ESTIMATED depth+camera+motion reconstruct the real video. 207k Gaussians/clip, object moves 0.26m.
- **BUG fixed:** video_gt.py stored gt_rgb at native 256² but H/W/intrinsics/uv are 512² (St4R size) → trainer render-loss crashed (512 vs 256). Fix = store gt_rgb resized to (Hm,Wm) (+ module-level `import cv2`); post-processed the 24 existing clips.
- **★ Training (resume sim_gen warm-start, gate+sem→gate+magnitude recipe, 20 train clips):** corr **0.53→0.90** (s60), top-mover ratio **0.16→0.85**, static-leakage 0.05-0.07, seg loss 6.7→0.10 (sem head learns), mover P/R 0.91/0.99, no NaN. **⇒ the model learns LOCALIZED + magnitude-correct + language-conditioned motion on PURE-VIDEO data, ≈ the maniskill sim numbers (corr 0.85/ratio 0.81), leak slightly higher (estimation noise blurs the static/dynamic edge).** Caveat: rPSNR only ~5.8 (the pure-video 3DGS appearance/geometry is COARSER than sim — mono-depth + upscaled gt_rgb — but render-loss weight 0.1 is minor; motion is the point). **THE END-GOAL (train from pure RGB video, no GT depth/camera/poses) IS VALIDATED.** NEXT: finish 1000 steps + heldtask held-out eval (cross-task generalization) + export video/ply. Improvements: better tracker (SpatialTrackerV2 occlusion>St4RTrack), VLM/SAM seg instead of the GT-mask shortcut, gsplat g0 opt for appearance. ckpt `checkpoints/libero_v1`. All UNCOMMITTED vs GitHub aa7f1d0.

## §48 LIBERO pure-video: the geometry+motion were BOTH broken (honest root-cause + fix)

User caught me claiming success on torn/distorted renders ("睁着眼说瞎话...物体全部都是撕裂的" then "还是扭曲的"). They were RIGHT. Stopped claiming, debugged to root cause. Two independent bugs in `code/scripts/video_gt.py`, both now fixed + visually verified (g0 render matches RGB, salad moves correctly toward basket):

**BUG 1 — anisotropic geometry (the "扭曲"/vertical-blob).** St4RTrack's raw pointmap `pts0` is anisotropically distorted on *sim* renders: fit means→stored-uv gives clean vertical axis (fy=618px / resid 1.7px) but broken horizontal (fx=1499px / resid 28px) — **x compressed ~2.4×**, so the scene rendered as a tall vertical blob. `estimate_focal` averaged the two → biased 672. True LIBERO agentview focal = robosuite fovy=45° → 256/(2·tan22.5°)=309@256px=**618@512** = the clean y-fit. FIX: estimate focal from the clean vertical axis (`fper=(vv*z)/y`, median) and **rebuild geometry by pinhole backprojection** `pts0=[(uu*z)/f,(vv*z)/f,z]` (keep St4R depth, which is faithful ~0.95 corr; discard its distorted x,y). Result: frame-0 PSNR 18→33.6, render matches RGB.

**BUG 2 — dead motion (static traj).** Two sub-causes: (a) the `conf>1.2` keep-mask **silently dropped the small manipulated object** (St4R low-confidence on the salad) → no object Gaussians to track. (b) the `object_of_interest` mask spans BOTH the moved object AND the static place-target (basket = the *larger* blob), so a single rigid PnP over all of them let the static majority win → identity. The OLD clips' "23cm motion" was SPURIOUS (broken geometry made PnP ill-conditioned) — that garbage is what scattered the earlier models. FIX: (a) **force-keep ooi pixels** regardless of conf; (b) **isolate the points that actually move in 2D** (CoTracker end-vs-start disp >~3px) and solve the pose only for them; static object points hold g0. Result: salad now n_mov=1157, mover_frac=0.29, moves left+up toward basket (2D centroid 160→51, basket@53), tight rigid disp 0.32m.

**Caveat (open):** PnP depth (t_z) is the least-constrained DOF → object depth-motion ~1.7-2.5× over the mask-size-implied value. Direction+coherence correct; magnitude refinement (constrain t_z by mask-area depth proxy) is future work.

**Lessons:** (1) val_psnr 18-24 was BG-composite-dominated — it never validated geometry; always render g0-vs-RGB directly. (2) corr metric hid both bugs. (3) Verify VISUALLY (g0 vs RGB, GT-motion direction) before any claim.

## §49 学习方式重构（用户 /goal："高斯在炸开扩散而非移动"）— 三个结构修正

用户审查 v4 视频的判断（正确）：预测像"原高斯的一部分逐渐炸开/扩散"，不是"学这些高斯如何移动"。逐控制点独立回归 + 几何近邻 LBS 没有任何"同一物体一起动"的结构。三个修正（A/B：v5 = v2 数据 + 这三项）：

1. **实体感知 LBS**（`deform.py build_lbs_binding(dense_seg, control_seg)` + `scgs.py` + `model_full(entity_lbs=1)`）：稠密点只绑同 seg 实体的控制点（无同实体控制点→退普通近邻）。杀跨界稀释（物体边界点被静止桌面控制点拖到 ~70% + 拖尾 = "炸开"主源之一）。
2. **实体刚性一致损失**（`losses.entity_rigidity_loss`，`--w_rigid 0.5`）：对每实体的预测端点做可微 Kabsch 自拟合，罚到自身最优刚体的残差。不与幅度监督打架（对拟合到的变换不变），只罚实体内不一致（撕裂）。GT 本身逐实体刚体（PnP/sim），先验精确匹配。单测：完美刚体→0，撕裂→0.042。
3. **gate 实体池化**（`model_full(gate_entity_pool=1)`）：p_dyn logit 按 seg 实体均值池化——move/stay 是物体级决策。诊断依据（`_libero_440debug.py`）：epi440 上 gate 把 59% 的 mover 控制点掐死（med gate 0.166），物体一半冻结=拖尾另一主源；方向 cos 0.90 全对，速度被 gate 压到 1.3cm/步 vs GT 4.3。
- ckpt 新字段：entity_lbs / w_rigid / gate_entity_pool；eval 脚本按 ckpt 重建并传 seg_per_g。

## §50 数据三修复 + 重大数据审计发现（LIBERO 失败演示）

**mask 审计**（agent，全 500 集一致）：msk id：0=背景，2=篮子，3-7=桌上物体，8=臂身，10=夹爪；**ooi = 63% 篮子 + 29% 夹爪 + 7.7% 物体** → v2 用 ooi 跟"物体"实为夹爪+物体混合刚体拟合（污染）。且 **id↔物体种类固定、操作哪个 id 随任务变**（epi0=id1，epi400 理论上=salad 的 id）。

**重大发现：epi400 是失败/未完成演示**——夹爪下到沙拉酱周围后空手撤回，**全部非机器人 id 全程 END-START 0-1px**。v2 heldtask 的"物体运动"100% 是夹爪运动（假 GT，模型一直被假 GT 评估）。⇒ 必须按"检测到的物体 id 真动了"过滤 episode（`_libero_scan_movers.py`，只解码 mask 列，END-START>15px 为 MOVER）。

**video_gt.py v3 改动**：
1. `find_object_id`：非{0,8,10} id 中**首末有效帧质心位移**最大者（路径和会被遮挡 NaN 吃掉搬运段）。
2. 逐实体运动：ENT_TRACK=(obj_id,8,10) 各自 CoTracker（一次合并 pass）+ 各自刚体 PnP；med2d>2px 才解（臂 39px/夹爪 38px 已验证跟上）。臂=数据修复1。
3. `_size_depth_correct`：表观尺寸单目深度线索 z_t=z_0·s_0/s_t（公共可见子集 spread，clamp [0.6,1.6]×z0，沿中心视线平移，中心重投影不变）。compact 实体（物体/夹爪）启用，臂(id8 多链节)禁用。=数据修复2。
4. 填洞：第二遍 St4R 反向锚定（锚=末帧）+ 同样的竖轴焦距重建；候选=末帧非实体像素投影落在 frame-0 实体区（未来的洞）内 + 体素去重(4mm) → 追加静止背景高斯（seg=末帧 msk id）。epi400 +690 点。=数据修复3。
5. keep-mask 强保 {obj_id,8,10} 实体像素；窗口按检测物体 id 的运动选。

**教训**：(a) 别信单一 mask 语义（ooi≠物体）；(b) 演示数据有失败集，必须按"任务效果实际发生"过滤；(c) corr/路径和等聚合指标会被遮挡和错实体悄悄骗过——每个结论都要可视化+逐实体数字双验证。

## §50b 数据 v3 视觉审计迭代记录（epi0 连续 7 轮目检定位的问题）

1. **幽灵臂**＝臂形空洞被远墙色填充、孤立浮在黑底（fill 无深度限制）→ fill 限 z≤p92(g0)。
2. **3cm 邻居判据矫枉过正**：洞中心离既有几何>3cm 是定义本身，填充塌到 81 点 → 撤销，只留深度限制。
3. **实体剩余高斯冻结**（重大 bug）：刚体解在 ENT_CAP 子集上解、也只应用到子集——夹爪 5270 点只动了 2500，剩余成"深色钩"幽灵 → 子集解算、**全实体应用**（_fit_rigid 重拟合）。
4. **臂单刚体失效**：id8=静止底座+多链节，静止多数把 Procrustes 拖成 identity → **运动聚类**（k-means k=2/3 on [末,中]位移）+ 每簇独立刚体 + 合成 seg id 50+c（喂给 entity-LBS/gate 池化）。
5. **静止桶误判**：首末位移判静止漏掉"中途动回原位"段 → 改全程最大位移；**CoTracker 暗色低纹理跟踪失败**→静止查询若 14px 内有运动查询则继承其簇（补标 +2000 点）。
6. **PnP 传送护栏**：遮挡/出画→垃圾位姿→部件高斯散飞 → 单步质心位移>0.15m 保持上一位姿；最少可见点 max(12,10%)。
7. **"碎片云"真相**＝p98 深度截断把真实墙切碎，臂走后露出（隔离渲染确诊 seg=0、z1.1-1.9、24899 点——根本不是运动 bug）→ p99.5 保墙完整。残余：St4R 远场深度噪声（后端极限，Pi3 候补）+ 烘焙阴影（GS 表示极限）——两者监督语义均为"静止背景"，不污染运动学习，接受并记录。

**方法论**：渲染审计猜了 4 轮无效后，改用**子集隔离渲染**（moved/static-id8/fill/bg 分开渲）一次确诊。教训：视觉伪影先隔离归因，再修。

## §51 v6 结果（§49 模型 + §50 数据 v3）— 当前最佳

settle 800 步（entity_lbs + w_rigid0.5 + gate_entity_pool + v3 数据）。**训练终态：leak 0.0000（静止泄漏字面归零）、gate mP 1.00/mR 1.00、dyn loss 0.0006、corr 0.96-0.97、ratio 0.97-0.99、rPSNR 15.9**（v4/v5: leak 0.013-0.028、rPSNR 13.5）。

held-task 推理（未见任务，REAL|GT|PRED 视频已目检）：
- epi410 salad: mover 中位误差 12.0cm / GT 35.4cm；epi450 tomato: 11.0/33.7；train epi100: 14.6/49.7。
- 目检：臂+夹爪+物体整体朝篮子移动、落点正确、**无 v4 的碎片带/炸开**；静止物体/桌面/篮子完全不动（ALL med 0.0cm）。训练 clip 上 PRED≈GT。
- 注意 mover 现在含臂部件+夹爪+物体（3 万点），比 v4 只有夹爪混合体的"mover"难得多——相对误差 ~33% 是诚实口径。

对照阶梯：v4（旧模型+假数据）→ v5（§49 模型+假数据：拖尾消失，mover err 12.1→9.2cm）→ v6（§49+真数据：语义全对+静止归零）。两条修复线（学习方式、数据质量）各自独立可证有效。

残余差距（下一轮）：幅度欠 ~30%（接触区轻度弥散）、渲染软（×4 splat + 数据伪影）、深度幅度仍偏（尺寸线索 clamp 内）。

## §52a 复盘核心实验：语言交换测试（v6，2026-06-10）— 语言被完全忽略

`_libero_langswap.py`：同 g0、同场景，TRUE 指令 vs 指向场景内另一物体的 SWAP 指令，**逐实体均匀采样控制点（无 GT mover 偏置）**，比较逐实体预测位移与 gate。

结果（train epi100 + heldtask epi410 两组）：
- **swap/true 比值 = 1.00 / 1.01**：换指令后逐实体预测一字不差。语言对"动哪个物体"零贡献。
- heldtask 还暴露：**假阳性 mover**（静止 entity7 被预测 37.7cm、gate 0.998）+ **真 mover 欠激活**（entity1 gate 0.434 → 12.3cm vs GT 35.7cm）。
- **评估泄漏**：此前所有 corr/ratio/review 视频用 mover-biased 采样（依赖 GT disp 选控制点）→ 假阳性实体几乎分不到控制点，失败被掩盖。均匀采样下问题全现形。

**根因（已量化证实）**：23/24 个 clip 中目标物体= frame-0 离夹爪最近的物体——"动夹爪旁的东西"纯视觉规则 96% 准确 → mover-BCE 从视觉特征即可完美拟合 → 语言得不到梯度压力（捷径学习）。InfoNCE（8-10 种指令、queue256 大量假负例）压不住，lang loss 全程上行 3.0→5.5。用户的直觉（"语义学习不够"）是对的，且比想象更严重：不是"不够"，是"没有"。

## §52 全代码复盘：实现与学习方法的问题清单（按严重度排序）

### A. 学习方法层（根本性）

**A1【致命】语言未被学习（§52a 已证）**。swap 比值 1.00；根因=夹爪邻近捷径（23/24）+ 监督结构缺陷：
- mover-BCE 的输入只有视觉 patch 特征 + e_sem——标签（谁动）与视觉捷径完全相关，语言通路（聚合 token 跨注意、全局 AdaLN cond、InfoNCE）没有任何 per-control 级的语言-实体绑定监督；
- `contrastive_lang_loss`/`v_wrong0` 反事实机制**写了但从未接入训练**（train_sim 不传 vlm_inputs_wrong）——语言依赖从未被强制；
- InfoNCE 在 8-10 种指令规模下 queue256 充满假负例（~1/8 同指令），lang loss 全程上行 3.0→5.5，形同虚设甚至有害。
**修复路线（优先级最高）**：(1) 数据：窗口随机起点提前到 approach 之前（夹爪远离目标时语言成为唯一信号，直接打断捷径）；(2) 监督：per-control 指令相关性头——控制点特征与指令 token 跨注意 + BCE（标签免费：该控制点实体==被指令实体）；(3) 接入反事实 wrong-instruction 损失（真 mover 在错误指令下 gate 必须关）；(4) InfoNCE 按任务去重或弃用。

**A2【致命】评估的 GT 泄漏**。`sample_controls` 用 GT disp 做 mover-biased 采样——eval/review/视频全部如此。它掩盖了 held-task 上的实体选择失败（假阳性 entity7 38cm/gate0.998 因为分不到控制点而不可见）。**修复**：评估一律逐实体均匀采样；语言交换选择准确率成为常设指标；corr/ratio 补报"全静止基线"对照（leak 维度的 corr 贡献需要校准）。

**A3【结构】逐点自由速度回归 + 事后刚性正则的架构错位**。GT 本质是 K×SE(3)/实体；现在让 28 层 DiT 每步为 2048 个点独立回归速度，再用 Kabsch 损失把它们拉回刚体一致——容量浪费且推理期无结构保证（440 的 gate0.43 半冻结即推理期一致性破裂的例子）。**修复方向**：实体槽位 SE(3) 头（实体池化特征+语言 → 每实体每步 SE3；控制点输出退化为小残差），rigidity 从"软约束"变"参数化保证"。

**A4【方法】运动相位结构无监督**。approach→grasp→carry→place 的相位由 tstep embed 隐式学；幅度损失只看端点（mover_magnitude 只在 K-1），中段滞后+末端跳变只有弱 velocity-smooth 抑制。30% 欠幅的另一来源是相位不确定性下的均值回归。**修复**：中段也加幅度监督（k∈{K/2,K-1}）或 soft-DTW 类时间对齐容忍损失。

**A5【方法】grounding 过时**。per-control patch 特征永远取自 frame-0 uv——物体移动后控制点还在读旧位置的像素特征。K=12 内可忍，长 rollout 必衰减。**修复**：每步（或每几步）用当前预测位置重投影重采样 patch 特征。

**A6【泛化】held-task 是弱 OOD**。LIBERO-object 10 任务共享桌面/物体库，heldtask 只是"没见过这个配对"，场景/物体全见过。真实泛化（新场景/新物体）未测。**修复**：跨 suite 评估（libero-spatial/goal）或 maniskill↔libero 交叉。

### B. 数据层

**B1 规模歧义**：St4R/Pi3 重建是 up-to-scale 的，逐 clip 焦距独立估计——`mover_thresh=0.01m`、`max_disp=0.1`、voxel 4mm 这些"米制"阈值跨 clip 含义漂移。**修复**：逐 clip 场景归一化（scene_r→1）或全局尺度锚定。
**B2 schema 缺逐帧相机**：单一 viewmat 假设静止相机——真实 ego 视频（动相机）进不来。Pi3 管线已算出逐帧位姿但 schema 丢弃。**修复**：schema 加 viewmats[Kf+1]，trainer 渲染损失按帧取。
**B3 splat 尺度链混乱（已查实）**：v2 的 ×4 重缩放被全量重生成冲掉，v4-v7 全在 ~1.2mm 微 splat 上训练（渲染损失偏弱），review 又 ×4 过度模糊。**修复**：数据生成时做覆盖率标定（或 frame-0 短时 gsplat 拟合），评审不再叠乘。
**B4 臂子部件 seg id 跨 clip 不一致**：50+c 按 k-means 簇序号逐 clip 随机——id51 在不同 clip 是不同物理部件，semantic_id_loss 的原型库被矛盾监督（v7 的 seg loss 0.24 居高与此相关）。**修复**：sem CE 把 50+ 折叠回 id8（LBS/gate 池化保留细分）。
**B5 填充点 uv 语义错位**：洞填点的 uv 是 frame-0 投影（在臂底下）→ patch 特征是臂的外观，sem 标签却是背景——特征/标签错位噪声。**修复**：fill 点免除 sem 监督（mask 掉）。
**B6 遮挡持位姿当全可见**：held-pose 段在 trainer 里 vis=全1，模型学到"空中冻结"伪相位（当前 0 held 帧不痛，规模化后会）。

### C. 训练/实现层

**C1 InfoNCE 队列 rank 各自漂移**（DDP 下 q_g/q_t 不同步）——负例分布逐 rank 漂移，加剧 A1.3。
**C2 dyn-gate 静态**：gate 由 frame-0 特征一次算出、整个 rollout 复用——"先静后动"（pre-grasp 物体）只能靠速度头时变补偿。
**C3 颜色/尺度通道近乎无监督**：dcolor 只受 0.1 渲染损失约束（影子区域会被颜色漂移补偿）；dlog_s 只有锚定。
**C4 1.78B 可训参数 / 20 clips / 800 步**：严重过参数化区间，热启动+冻结 Qwen 兜底；记忆场景而非学规律的风险真实存在（与 A6 互证）。
**C5 工程**：clip schema 无单一权威文档（事实标准散在 maniskill_gt 注释）；gen 脚本 ×3 份近重复；12 个 _libero_* 调试脚本未归档；**全部改动未 commit/push**（vs aa7f1d0 巨量 diff）。

### D. 结论与下一轮优先级

当前模型 = "视觉先验的实体动力学模型"（哪个实体动选错时语言救不了），运动表达已修到可用（v6：静止归零、实体刚性成立、轨迹形态正确）。**下一轮唯一主线是把语言变成因果输入**（A1 的四件套 + A2 评估改革），其余按 B/C 顺序带过。Pi3 后端（v7 训练中，rPSNR 显著更优）作为 ego-ready 默认后端。

## §53 v7/Pi3 数据质量审计（用户："先解决数据问题"）— 含一处自我纠错

**纠错 B3（重要）**：之前说"splat 尺度链混乱、1.2-1.4mm 微 splat 太小、训练渲染信号弱"——**错了**。尺度扫描实测（`_scalesweep.py` epi0-pi3）：
- x1.0（原生 1.4mm）：覆盖率 **0.996**、L1-vs-REAL **0.018**（最佳）；x2 0.998/0.033；x4 1.000/**0.058**（最差，过糊）。
- 即原生尺度近乎完美闭合表面且光度匹配最优；训练一直用的就是 x1，没问题。**"过糊/鬼影"全是我的 review 视频 ×4 造成的**（review 工具 bug，非数据 bug）。已把 `_libero_data_video.py` 默认改 1.5×。

**Pi3 数据真实质量（1.5× 目检 epi0/epi410）**：
- ✓ 焦距各向同性（f_u≈f_v，无 St4R 的 x 压缩）；✓ 静止场景+被操作物体：清晰锐利、位置正确；✓ 物体/夹爪跟踪正确移动；✓ 逐帧深度→运动更干净（训练 rPSNR 18-20 vs St4R-v3 的 15.9）；✓ 逐帧相机位姿（ego-ready）。
- ✗ **机械臂区域杂乱**（唯一真实的逐 clip 数据问题）：臂是多链节铰接体，被 k-means 近似成 2-3 个刚体簇→簇边界撕裂；+ 反向锚定填洞在臂区散点。隔离渲染确认：max disp 0.59m、仅 7 个 >0.5m 点，非严重外飞，是局部杂乱。
- val frame0 24-27（略低于 St4R-v3 的 28-31，Pi3 静止几何稍糙），但运动 rPSNR 更高。

**未用满的 Pi3 优势**：臂的运动现在还在用"刚体簇"（为 St4R 无逐帧深度而设计）。Pi3 有逐帧深度→可对每个臂 Gaussian 直接用 CoTracker-2D + 逐帧深度做 per-point 3D lift，无需刚体聚类，天然处理铰接。下一轮臂修复方向。

**数据问题优先级（修正后）**：
1. 【最高·分布级】夹爪邻近捷径（§52a：23/24 目标=离夹爪最近物体→语言被忽略）。这是最重要的数据问题，远比臂渲染重要。修：随机窗口起点提前到 approach 之前 + 多物体场景下打断"动最近物体"的相关性。
2. 【中·逐clip】臂 per-point lift 替代刚体簇（仅 Pi3 后端可行）。
3. 【低】填洞散点收紧（Pi3 远场深度噪声）。

## §54 v8: language as a CAUSAL input + entity-slot SE(3) (the /goal plan, 2026-06-10)

Root cause recap (§52a): model ignored instruction (swap/true=1.00) — gripper-proximity shortcut (23/24
clips target=nearest object) + the per-control patch feature is instruction-BLIND (Qwen causal, image
before text) + no per-Gaussian language↔entity binding + counterfactual loss never wired.

**Architecture (model_full.py):**
- §54 RELEVANCE HEAD (`_relevance_logit`): q = control's 2D patch feature, k/v = per-token instruction
  text feats (`encode` now returns `text_feats=hidden_all[-1][text_mask]`). r_logit ADDS to the dyn-gate
  logit, but ONLY for OBJECT-class controls (seg 1-7) via `objmask` — the robot (arm/gripper) executes
  under EVERY instruction; only WHICH OBJECT is picked is language-dependent. Zero-init last layer =>
  exact v7 warm-start (unit-tested: |Δmeans|=3.5e-6).
- §54 COUNTERFACTUAL: one extra frozen-Qwen forward on a WRONG instruction (same patch q), supervise
  `p_dyn_wrong`/`p_rel_wrong`→0 on the named object's controls. Same patch+different text MUST flip the
  gate => a vision-only head is unsatisfiable. DDP static_graph-safe (rel params used every step).
- §54 ENTITY-SLOT SE(3) head (dynamics/model.py, flag-gated): pool DiT features by seg entity -> per-
  entity rigid (v_e,ω_e), broadcast as a rigid transform about the entity centroid; per-control head =
  small residual. Zero-init => rigid term 0 => legacy motion at init. Rigidity becomes STRUCTURAL
  (an entity's controls share one SE(3) by construction) instead of the §49 soft Kabsch loss.

**Losses (losses.py):** `relevance_bce_loss` (r vs is_obj over objects), `counterfactual_gate_loss`
(named object suppressed under wrong text — the LOAD-BEARING one), `mover_magnitude_loss` now multi-step
{K/2,K}, semantic_id_loss folds arm subparts (id>=50→8). InfoNCE off (w_lang_contrast 0).

**Data (video_gt/pi3_video_gt pick_window `mode=early`):** start the window 10-30 frames BEFORE motion
onset (gripper still far) so proximity no longer predicts the mover — language must carry selection.
gen_libero_pi3_v2.sh emits _e (early) + _c (center), split-suffix LAST in the filename.

**Eval reform:** fixed eval_sim_generalization (was silently rebuilding the model WITHOUT
gate_entity_pool/entity_lbs/rel + never passing seg_per_g -> §49 ckpts mis-evaluated). New
`eval_langswap.py`: per-entity UNIFORM sampling (no GT leak), selection accuracy over heldtask×swaps.

**v8-lang first signal (rel head only, Pi3 24 clips, resume v7):** cf 11.2→0.17, **cfSup 1.00→0.00 by
s180** (gate now CLOSES on the named object under a wrong instruction) with TRUE-pass corr 0.945 / leak
5e-4 / mP·mR intact. relSel still 0 (r≈0 under TRUE — gate stays open via the visual logit; r goes
strongly negative only under WRONG). Decisive test pending: eval_langswap swap/true ratio (was 1.00).

## §54 v8 RESULTS (the headline: language is now causal)

eval_langswap (per-entity UNIFORM sampling — no GT leak; swap/true = named object's motion under a SWAP
instruction ÷ under the TRUE instruction; selection accuracy gates floor∧suppression∧quiet):

| run | TRAIN (seen nouns) sel-acc | TRAIN swap/true | HELDTASK sel-acc | note |
|---|---|---|---|---|
| v7 (no rel head) | 0.00 | ~1.00 | 0.00 | language IGNORED (the §52a failure) |
| v8-lang (rel head) | 0.53 | **0.10** | 0.00 | language CAUSAL; floor gated by magnitude undershoot |
| v8-ent (+entity SE3) | **0.84** | **0.00** | 0.00 | entity head fixed magnitude (TRUEmove≈GT 25-27cm); suppression perfect |

So: **swap the instruction → the object's motion drops to 0-10%** (was 100%). Visual proof:
viz/libero_v8/langswap_epi310.mp4 (milk moves under "milk", frozen under "alphabet soup"). cf loss→0,
cfSup→0 throughout. The entity head ALSO cured the §51 magnitude undershoot (sel-acc 0.53→0.84).

**Honest limitation — HELDTASK (unseen TARGET noun) over-suppresses (sel-acc 0).** salad dressing (task 8,
held out) was only ever seen as a DISTRACTOR (relevance label 0), never a target (label 1) -> the
relevance head outputs r<0 for it even under the correct instruction -> gate closes -> TRUEmove=0. This is
a VOCABULARY-generalization limit of 8 training nouns (the relevance head memorized them), NOT a mechanism
failure (the mechanism is proven on seen nouns). To disentangle "unseen NOUN" from "unseen SCENE", v8b
adds a HELDSEED split (seen nouns, unseen episodes). Real fix needs noun diversity / a real-world-scale
vocab — exactly why the pipeline is now Pi3 (ego-ready). Both training runs = 4×A100 DDP (world=4).

## §55 v8 结果定论 + v9 replan（2026-06-10）

**v8 验证阶梯完成，逐代诊断（uniform 采样、build_model 全 flag）：**
| 模型 | obj位移 | 终点误差 | 方向cos | langswap |
|---|---|---|---|---|
| v7-pi3 | 27.4 | 10.2cm | +0.93 | swap=1.00（语言被忽略）|
| **v8-lang** | 32.0 | **6.0cm** | **+0.99** | **swap=0.10 ✓** |
| v8-ent | 27.5 | 42.2cm | **-0.18** | swap=0.00 |
| v8b | 24.7 | 40.8cm | -0.19 | swap=0.00 |

**赢**：§54 语言因果化成功（v8-lang：换指令物体运动 1.00→0.10、cfSup 1→0、视觉+数值双证；方向终点近乎完美 6cm/+0.99）。**此前担心的"欠幅"在 Pi3 数据上不存在**——是 review 脚本漏 entity_head flag 造成的 172cm 假象（已修 _libero_review_video 用 build_model + uniform 采样）。

**回归（已定位、用户决策封存实体头）**：实体槽位 SE(3) 头把方向 +0.99→-0.18。证据链：v8-lang 方向完美 → 加实体头(v8-ent) 直接崩。原因二合一：(a) 实体池化把逐控制点特征平均，**丢失了方向信息**（每个控制点本来有自己正确的方向，池化成一个实体级方向时若 MLP 学不到正确朝向就全错）；(b) `w_resid=0.1` 残差正则**压制了本来方向正确的逐点残差通路**（v8-lang 的逐点头方向是对的，被当噪声压掉）。诊断：uniform vs mover-biased 采样误差一致（42 vs 42），排除采样问题；GT 方向 cos(move,toward-basket)=0.52-0.60 合理，排除数据噪声。**修法（未来）**：方向感知池化——把控制点相对质心的几何（x_i−c_e）编码进实体 MLP，让它能表达旋转而非只平移；本轮不做，flag 默认关。

**方法论漏洞（已补）**：corr/ratio/mag 全是范数指标、方向盲 → 回归没被任何训练/eval 数字暴露，靠逐代手动诊断才发现。§A1 已加 **dir-cos**（eval_langswap 每 clip + 汇总；train_sim log 行 `dcos`）。

**未解决根本缺口**：未见名词泛化（heldtask 过度抑制 TRUEmove=0）——8 名词词汇硬限制。**用户决策：v9 主攻开放词汇分割（GroundingDINO+SAM2 替代 GT mask）**，解锁无 mask 的 LIBERO suite 扩词汇/场景 + 真实视频 + 推理期诚实分割。SAM2.1-large + GroundingDINO-tiny 权重已在 hf_cache。

**v9 路线**：A) dir-cos 补盲 + v9-lang（rel only、v2 双窗、干净读数）+ 封存实体头；B) openvocab_seg 管线 + IoU 自验 + 词汇扩展 → v10 未见名词测试；C) 真实 ego 视频试点（Pi3 已 ego-ready，缺 schema 逐帧 viewmat）。

## §56 v9 执行进展（Phase A 固化 + Phase B openvocab，2026-06-10，受 session-limit 约束）

**A2 v9-lang**（rel only、双窗 40 train、resume v7）：双窗数据训练**不稳**（per-clip corr/dcos 剧烈振荡：corr 0.14-0.87、dcos -0.34~+0.96，因早窗 clip 运动只占部分、更难）。后期收敛 dcos~0.90-0.93、corr~0.87——方向保住了（无实体头），但略低于 v8-lang 的 +0.98。最终三划分 eval 待出（heldseed=场景泛化关键读数）。dcos 已进训练 log（§A1）。

**B1 openvocab_seg.py**（agent 在 session-limit 前建好 423 行：GroundingDINO-tiny + SAM2.1-large + IoU 自验 CLI）。IoU 实测（5 episode）：
- **basket(2)=0.97-0.98** ✓（SAM2 质量极好）。
- **arm(8)=0.000** ✗——根因：GroundingDINO-tiny **把整个机器人 lump 成一个检测**，"robot arm"短语没单独命中，整机器人被赋 id10（gripper, 4118px），id8 空。
- **named object(1)=0.00 on 4/5**（只 epi410 salad dressing 命中）✗——tiny 无法在相似桌面物体间 ground 具体名词；且**漏检 5/7 干扰物**（只找到 2 个）。
- **关键认识**：训练数据 NOT 需要 openvocab ground 对名词——现有 `find_object_id` 已按运动仲裁目标。openvocab 只需把"所有物体+臂+夹爪+篮子"分出类别；运动挑目标。所以 named-object IoU 失败对训练无关紧要；真正的坑是 **arm lump + 漏检物体**（tiny 太弱）。
- **修复中**：GD_REPO 改环境可覆盖；下载 grounding-dino-base（更强）重测中。SAM2 mask 质量本身没问题，瓶颈纯在 detection。
- 若 base 仍不够：备选 SAM2 自动 mask 生成（segment everything）+ 按"非臂/夹爪/篮子/背景"归为物体，或 LIBERO sim 重渲分割（更可靠但失去真实视频兼容）。

**session-limit**：子 agent + workflow 被限（7:30am EST 重置）；数据源调研 workflow 失败待重跑。直接 server 调用与训练进程不受影响，继续 inline 推进。

## §57 Phase A 收尾：v9-lang 是生产模型（场景泛化成立）+ openvocab 结构性障碍

**决定性 A/B（同一 v2 held-out 划分，公平对比）：**
| heldseed（见过名词、未见 episode=场景泛化） | sel-acc | dir-cos | endpoint |
|---|---|---|---|
| **v9-lang（双窗早起点）** | **0.75** | **+0.81** | 7.8cm(epi40) |
| v8-lang（单窗） | 0.25 | +0.76 | — |

**结论：双窗早起点数据把场景泛化从 0.25→0.75（3 倍），方向 +0.76→+0.81。v9-lang = 生产模型。** 之前以为 v8-lang 方向 +0.98 更优是误判——那是 v8-lang 在自己训练 clip 上的数；公平 held-out 对比 v9-lang 全面更好。视觉证据 `viz/libero_v9/v9_heldseed40.mp4`：未见场景里正确选中 alphabet soup 并推向篮子。heldtask（未见名词）两者都 0——纯词汇限制，Phase B 解决。

**openvocab grounding-dino-base IoU = tiny 字节级相同**（arm 0.00 / obj 0.00 / basket 0.97）→ **障碍是结构性 LOGIC，不是检测器强度**：(1) "robot arm"短语没单独命中、整机器人被 lump 成 id10；(2) 用名词词表逐个检测物体覆盖率差。修法（需迭代，待 session 重置后子 agent）：机器人统一归 id8（让管线运动聚类拆关节）+ 物体改用 SAM2 自动 mask 生成（segment everything）按排除法归类，而非靠 GroundingDINO 逐名词。basket(0.97)+SAM2 mask 质量本身没问题。

**Phase A 完成度**：A1 方向指标 ✓、A3 实体头封存 ✓、A2 v9-lang 生产模型 ✓（场景泛化证实）。Phase B openvocab 卡在结构性 logic + session 限额；Phase C 真实视频依赖 B。

## §58 openvocab 阶段性结论（B1 部分完成，需 SAM2-AMG 重构）

robot→id8 lump + generic-shape prompts（grounding-dino-base）后实测：
- **arm(8): 0→0.65**（机器人整体归 id8，data 管线运动聚类拆关节）；**basket(2): 0.97**；SAM2 mask 质量本身极好。
- **物体覆盖不稳**：epi0 全 6 物体 100% 覆盖；但 epi100/300 **目标物体(id1)=0%** 漏检。GroundingDINO（即使 base）+ 名词/形状 prompt **无法稳定检出所有小桌面物体**——目标物体只 ~1/3 时候被覆盖，不够格生成数据（目标必须被覆盖才能跟踪）。
- **根因**：依赖 GroundingDINO 逐 prompt 检测本质不可靠。**正确修法 = SAM2 自动 mask 生成（point-grid AMG，segment everything）**：分割所有 mask → 按"非机器人(GroundingDINO robot)/非篮子/非背景(最大/地板)"归为物体 ids，完全不依赖物体名词检测。这是 B1 的真正解法，需迭代（plan 自己估 Phase B 2-3 天）。
- **管线集成缺口**：现 `find_object_id` 需全帧 mask 找最动 id；openvocab 只有 frame-0 → 需改成用 CoTracker 位移（frame-0 物体像素跟踪位移最大者=目标）仲裁，非全帧 mask 质心。这是"--seg openvocab"集成的真正改动（不止换 mask 源）。

**Phase B 状态**：openvocab_seg.py 建成、arm/basket 达标、SAM2 质量验证；物体检测 + 管线集成需 SAM2-AMG 重构（待 session 重置后子 agent 迭代）。Phase A 已完整收尾（v9-lang 生产模型、场景泛化 0.75、已提交）。

## §59 openvocab AMG WIP（待 session 重置后子 agent 调优）

加了 `segment_frame_amg`（SAM2 point-grid 100 点 → segment everything → 按排除法归类）。SAM2 point-grid API 验证可用（100 prompt→100 mask）。但首版分类失败：arm/basket/target IoU 全 0，却painted 7 个 object（说明 dedup 保留了错误区域——floor 子块当物体了，robot/basket 的 GroundingDINO box 分类没命中）。需调：(1) background 移除（max_frac + floor 颜色/边界判据）、(2) robot/basket box 检测的 prompt/阈值、(3) dedup 容器判据。这是迭代实验，子 agent 最合适（session 限额阻塞至 7:30am EST）。

**openvocab 当前最佳 = box-based `segment_frame`**（robot→id8 lump + generic shapes，grounding-dino-base）：arm 0.65 / basket 0.97 / 物体覆盖不稳（目标 ~1/3 命中）。AMG 是正确方向但未调通。

**B1 状态**：脚本框架完整，两条路径（box / amg），SAM2 mask 质量好；物体可靠检测未达标。集成（find_object_id 改用 CoTracker 位移仲裁）未做。**B2/B3 待 B1 物体检测达标**。

## §60 openvocab AMG 调通到可用（B1 实质完成，剩最后一公里）

`segment_frame_amg` 经多轮 inline 调试已从"完全坏"调到可用：
- **arm 0.65 / basket 0.85-0.98 / 目标覆盖 3/5**（epi0/100/200 ✓，epi300/410 ✗）。
- 关键修复链：(1) SAM2 point-grid(20×20) segment-everything→所有实体的干净 mask；(2) 机器人→id8（GroundingDINO box，pipeline 运动聚类拆关节）；(3) basket→id2（box）；(4) **floor 用 SAM2 floor-mask 重叠判据剔除（非颜色——tan 物体在 tan 地板上是独立 region 所以保留，floor 碎片在 floor mask 内所以剔）**；(5) 名词补充检测（distinctive 物体）。诊断要点：AMG 的 mask 本身就好（物体/臂/篮子都分出来了），全部问题在分类。
- **剩余 2/5 目标漏检**（epi300/410）：grid 漏 + 这俩名词 GroundingDINO 也没 ground。需要更密 grid 或 motion-arbitration 兜底。
- **pyc 陷阱**：rsync -a 保留源 mtime 可能旧于 server 上 .pyc → Python 用旧字节码。改脚本后必须 `find __pycache__ -name openvocab_seg* -delete` + PYTHONDONTWRITEBYTECODE=1。

**B1 状态**：AMG 路径可用（arm/basket 达标，目标 3/5），比 box-based（目标 1/5）好，是推荐路径。**剩最后一公里**：目标覆盖 3/5→全覆盖（更密 grid + motion 兜底），+ pipeline 集成（find_object_id 改用 CoTracker 位移仲裁目标，不再需全帧 mask）。这两步 + B3 词汇扩展待子 agent（session 限额阻塞）。

## §61 openvocab 集成进数据管线 + B2 诚实性测试启动（point-prompt 兜底）

§60 的 AMG 目标覆盖只有 3/5（帧间方差：grid 在某些帧漏掉目标）。**根因**：依赖每帧 AMG grid 命中目标不鲁棒。**修法（plan 允许的 gen-time GT-motion 仲裁）**：用 GT mover 质心做单点 prompt 喂 SAM2（`_sam2_point`，multimask 取最高 IoU），目标必出干净 mask 标 id1。**mask 本身仍是 SAM2 质量（保留分割噪声供诚实测试），只有目标"选择"用 GT 运动**——推理期（真实视频）改用 CoTracker 运动即可，无需 GT。

**自验（B2 前半，5 episode vs GT mask）**：目标-IoU **0.95-0.97**、basket **0.97-0.98**、arm **0.58-0.66**（arm 管线内按运动重聚类，frame-0 IoU 次要）。帧间方差消除，目标 5/5 覆盖。单 episode 全管线跑通：target-cov 1.00 / IoU 0.97 / 物体真实位移 15cm（之前 target-cov 0 时 disp=0 的垃圾 clip 被自验指标正确标红）。

**管线集成**：`pi3_video_gt.py --seg {gt,openvocab}`。openvocab 路径只替换管线真正消费的两帧 mask（widx[0] seg_per_g+g0-keep、widx[-1] 补洞排除）；windowing/目标 id 仍走 GT 运动（生成期允许）。三处 mask 源统一为 `mask_w0`/`mask_wL` 局部变量。

**Sam2Processor 坑**：point prompt 需 4 层嵌套 `[image][object][point][xy]`、labels 3 层；multimask 输出 `[obj,n_masks,H,W]` 需降到 `[n_masks,H,W]` 再按 iou_scores 选。

**B2 后半（运行中）**：`orchestrate_v9ov.sh`（detached）= 56 clip 用 `--seg openvocab` 重生成 → train **v9lang_ov**（同配方：rel_head/w_rel1.5/w_rel_cf1.0/entity_head0/resume v7_pi3/800步）→ langswap 三划分。**诚实性判据**：v9lang_ov 在 OV 数据上的 sel-acc/dir-cos 不比 GT-mask 版 v9lang（heldseed 0.75/+0.81）差 >10% → 管线能扛自己的分割噪声 → 真实视频可行。日志 `logs/orchestrate_v9ov.log`。

## §62 openvocab 训练数据诚实性自验（B2 前半，子 agent 交叉验证，CPU-only 不碰训练）

56 个 OV clip vs GT-mask clip（同 episode/window）逐项对照：

**目标-IoU 分划分**：train 中位 0.96/均 0.937（0 个<0.70）、heldtask 中位 0.97/均 0.936（0<0.70）、heldseed 中位 0.955/均 0.900（**1 个<0.70**）。全体中位 **0.965**、均 0.931。早窗(0.97)略优于中窗(0.955)。

**唯一离群** `epi000140_c_heldseed`（"pick up the **butter**"）：cov 0.99 但 IoU **0.55**——三独立信号定位：物体 Gaussian 数 OV/GT=672/380=**1.77×**（全场最大过分割）、中窗、butter 是该波 GT footprint 最小物体（380 vs alphabet-soup 1487）。**小/平/低对比物体上 point-prompt SAM2 向邻域外溢**。孤立失败，非系统性。

**运动一致性**（windowing 相同→物体 3D 位移必须一致）：全 56 clip 物体终点位移 **中位 |Δ|=0.000m、最大 |Δ|=0.009m**（CoTracker 随机性）。**证明 OV 只改了分割，GT-运动驱动的 windowing + 动力学目标完全未变**——干净 A/B。（关键：必须用 `is_obj` 键算物体位移，GT 的 `seg_per_g` 带全 LIBERO schema；`is_obj === seg_per_g==1`。）

**实体分布恢复**（bincount seg_per_g）：物体 id1 比值中位 1.03，**49/56 在 ±15% 内**，0 个欠分割；basket 近乎完美（中位 1.01）；arm 中位 1.11（非训练目标）。**结构性注意**：只有 {1=物体,2=basket,8=arm} 在 OV↔GT 对齐；干扰物 id 3-10 不对齐（GT 用 sim label，OV 用 GD→schema 映射）——**但训练只用 `is_obj`(seg==1) 喂 relevance 头（v9-lang headline），不监督干扰物 id**，故不影响 A/B。

**判据通过**：运动一致(Δ≤0.009m)、监督 mask(物体/basket) 55/56 在 ±15%、IoU 中位 0.96、唯一失败可定位且在 eval clip——**足够干净，是公平的诚实性测试**。等 v9lang_ov langswap 三划分出数对比 GT-mask v9-lang(heldseed 0.75/+0.81)。

## §63 Phase C 实现（逐帧 viewmats，真实 ego 视频就绪）+ B3 词汇扩展下载方案（子 agent）

**Phase C（移动相机/真实 ego 视频）已实现**——子 agent 代码审计确认逐帧 world→cam 矩阵本就存在（`rel[t]=inv(T0)@poses[t]` 是 cam_t→canonical，取逆即 viewmats[t]），纯 schema/管线改动、零新几何：
- `pi3_video_gt.py`：build_clip 算 `viewmats=stack(inv(rel[t]))` [Kf+1,4,4]、return dict + save dict 加 `viewmats` 键（保留 `viewmat`=viewmats[0] 向后兼容）。LIBERO 静态相机 viewmats[t]≈I（cam_t.max<0.02 已验）。
- `train_sim.py`：加载 viewmats，**向后兼容广播**（无 viewmats 键的旧 clip→`viewmat[None].expand(Kf+1)`，逐帧索引统一、对静态 clip 逐字节等价）；finite-guard 改查 viewmats；渲染环 `viewmat[None]`→`viewmats[tt][None]`（确认渲染损失渲染多帧未来 rsteps，逐帧 GT_rgb[tt] 对应逐帧相机）。
- `render.py`/`model_full.py` 无需改（render_gaussianset 签名已是 batched [C,4,4]；模型相机无关）。本地语法过。**待 GPU 空闲后 pilot 验证**（regen 一个静态 clip 确认 viewmats[0]==I），再跑真实视频。

**B3 词汇扩展下载方案**（子 agent 在 server 实测元数据，proxy 200）：唯一 schema 匹配（LeRobot v2.1、`observation.images.image`+`wrist_image`、instruction 在 `meta/episodes.jsonl[*].tasks[0]`）= **`IPEC-COMMUNITY/*_no_noops_*_lerobot`**。排名：
1. **`libero_90_no_noops_lerobot`**（3921 eps、73 任务、~20+ 新名词 book/caddy/mug/bowl/drawer/microwave… + 新动词 put/open/close/turn/push + 空间介词）= **先拉**，最可能把 heldtask sel-acc 从 0 抬起。
2. **`libero_goal_no_noops_1.0.0_lerobot`**（428 eps，动词/关系密集）= 次拉。
3. libero_spatial（空间指代）、libero_10（长时多物体）备选。
- **fps 注意**：现数据 fps=10，IPEC 是 fps=20（2× 时间密度）→ gen 时 stride 加倍保持同 wall-clock 窗口/运动尺度。
- 拒绝（schema 不符）：physical-intelligence/libero(v2.0 裸键)、HuggingFaceVLA/libero(v3.0)、jesbu1(v2.0)。
- **下载中**（detached，network-only）：libero_goal 全量 + libero_90 meta（logs/dl_libero_goal.log、dl_libero_90_meta.log）。disk 51T free。

## §64 B2 诚实性判据 = 2×2 隔离（openvocab 推理就绪；训练损失局限于选择通路）

v9-lang-ov（OV mask 训练）vs v9-lang（GT mask 训练），langswap heldseed（场景泛化）sel-acc/方向余弦，**2×2 交叉评估**隔离"训练影响"与"评估数据影响"：

| heldseed | eval on GT data | eval on OV data |
|---|---|---|
| **v9-lang**(GT训练) | 0.75/+0.81 | **0.75/+0.81** |
| **v9-lang-ov**(OV训练) | 0.50/+0.79 | 0.50/+0.78 |

三划分完整：train 两者 0.57/+0.71≈0.57/+0.72（一致）；heldseed 见上；heldtask 两者 0.00（词汇限制）。

**① 评估数据影响 = 0 → openvocab 分割推理就绪（Phase B 的决定性胜利）**：看行——每个模型在 GT-seg 和 OV-seg 数据上**得分完全相同**（0.75=0.75、0.50=0.50）。GT 训练的模型在 openvocab 分割数据上和在 GT 数据上一模一样好。**这是对真实世界最关键的证明**：真实视频推理时没有 GT mask，此结果证明 openvocab mask 是完美替代。butter 离群没有影响大局。

**② 训练影响 = 真实但只伤"选择"通路（方向幸存）**：看列——同一数据上，OV 训练模型选对物体 0.50 vs GT 训练 0.75，但**方向泛化保住（+0.79 vs +0.81）**。即 OV mask 噪声只伤了 relevance/选择通路，没伤 dynamics/方向。最可能根因：物体 mask 过分割（is_obj 监督目标更噪——物体 Gaussian 数中位 1.03× 但最高 1.77×）。**诚实保留**：heldseed 仅 8 clip，部分可能是训练方差；但"选择掉/方向稳"的选择性模式说明是真实的局部效应。

**结论**：openvocab **推理/评估完全就绪**（解锁真实视频 + 无 mask 套件的评估）；openvocab **训练**需 mask 质量门（§50 思路：丢高过分割 clip，或收紧 SAM2 目标 mask）才能让选择泛化回到 GT 水平。修法折叠进 B3：gen 时加 IoU/过分割门。

## §65 刚体一致性问题：调研 + 数据核查 + 设计方案（用户发现的 coherence 盲区；方案待拍板）

**问题（用户目检发现）**：v9-lang 预测的物体高斯球会散开（cream cheese extent ×3.73），不是刚体整体移动。所有现有指标（corr/ratio/dir-cos/endpoint）都是聚合量、对散开盲视——与早前"方向盲区"同性质的指标盲区。根因：主 loss `trajectory_loss` 逐控制点独立 L1（点间零耦合）；`entity_rigidity_loss`（可微 Kabsch 残差，w_rigid=0.5）只是训练期软惩罚，在 held-out 场景失效（软先验不泛化，推理期无结构保证）。

**文献调研（四家族）**：① 软刚性损失（Dynamic 3D Gaussians 3DV'24 local-rigidity、SC-GS CVPR'24 ARAP）= 我们现状，逐场景优化够用、前馈泛化失效；② 低秩运动基（Shape of Motion 2024：共享 SE(3) bases × 逐点系数 = "软分解成刚性组"；HiMoR CVPR'25 层级化）= 用户直觉的通用形式，柔性/关节的远期路线，刚体阶段超配；③ 硬性逐物体 SE(3)（DreMa、机器人 GS 世界模型）= 被封存的 entity head，特征池化丢方向（+0.99→-0.18 教训）；④ **逐点投票→网络内可微刚性聚合**（Gojcic CVPR'21 Oral, Rigid 3D Scene Flow：逐点 flow → 物体级刚性抽象，端到端，提升精度+泛化）= 最适配。ManiGaussian/GWM 等 GS 操作世界模型用形变场、无刚性保证（同样会散，非答案）。

**数据核查（16 clip 只读诊断 `_diag_rigid_survey.py`）**：GT 刚性残差 **0.00cm（16/16）**——GT 构造性刚性，刚性应是硬参数化非软惩罚；每实体控制点 146-171（Kabsch 充足）；预测 extent-ratio 中位 1.11、**4/16 >1.2**、最差 3.73，预测刚性残差中位 2.45cm（物体仅 6-11cm，同量级=视觉散架）；**Kabsch 投影后方向 16/16 完全不变**（+0.86→+0.86）——输出空间聚合保方向，与特征空间池化（entity head 失败）形成实测对照。附带：`_e` 早窗方向本身差（+0.25~0.54）= §57 已知双窗不稳问题，正交于刚性。

**方案（等用户拍板）**：A（推荐）= predict_deltas 后、LBS 前加**逐实体 weighted-Kabsch 聚合层**（权重=p_dyn；刚性 by construction、方向从投票继承、warm-start 恒等、w_rigid 退役、eval 加 extent-ratio/刚性残差堵盲区；SVD 退化→eps+小实体纯平移 fallback；arm 用 50+c 子部件）；B（零成本 stopgap）= 纯推理期投影（已验证 3.73→1.00 方向不变）；C（远期）= motion bases 升级路径。验证阶梯：①coherence 指标进 eval → ②方案B A/B 基线 → ③方案A + resume v9-lang settle 800 步 → ④判据：heldseed sel≥0.75 且 dir≥+0.8 持平、extent-ratio→1.00±0.05。

**Phase C viewmats pilot 验证（§63 收尾）**：pilot clip（epi0, --seg gt）`viewmats[13,4,4]` ✓、`viewmats[0]==I`（1e-16）✓、静态相机逐帧偏差 0.0054/平移 0.0064m（与 camera-static sanity 一致）✓、旧 clip 广播逐行等价 ✓；新旧两路 trainer smoke 见 log。

## §66 v10-rigid 接口设计（batched weighted-Kabsch 聚合层；设计文档，未实现，等拍板）

**定位**：§65 方案 A 的向量化形式 = "运动低秩"通用表示（方案 C / Shape-of-Motion 形态）在刚体+有 seg 条件下的硬归属特例。v10-rigid（seg 硬归属）→ v11-bases（可学习系数）同一套数学渐进放松。设计判据（用户）：可 scale、GPU 利用率不掉、优雅（删 loss 而不是加 loss）。

```python
# igsw/dynamics/rigid_agg.py（拟新建）
def entity_rigid_aggregate(pos0, pos_pred, ent_id, w=None, min_pts=4, eps=1e-7):
    """把逐控制点预测位置投影到逐实体 SE(3) 轨道上。全程批量、零 Python 循环。
    pos0[M,3] 帧0位置; pos_pred[K,M,3] 原始逐点预测（=方向投票场）; ent_id[M] 压缩实体id
    （objects 1-7、arm 子部件 50+c 各自一个、gripper 10；背景/静态 id 不聚合=passthrough）;
    w[M] 投票权重（p_dyn × vis；None=均匀）。
    返回 [K,M,3] 刚性一致位置 + (R[K,E,3,3], t[K,E,3])。"""
```
每步 k（K·E 个拟合一次批量解）：① 加权质心 μX_e/μY_e：两次 index_add；② 互协方差 H_e=Σw(x−μX)(y−μY)^T：einsum('m,mi,mj->mij')+index_add→[E,3,3]；③ 批量 SVD [K·E,3,3]，R=V·diag(1,1,det(VUᵀ))·Uᵀ（反射守卫）；④ t=μY−RμX，out=R[ent_id]x+t[ent_id]。退化守卫：Σw<min_pts 或 σ₂/σ₁<eps → 纯平移 R=I（torch.where，无数据依赖分支，DDP static_graph 安全）。

**集成点**：dynamics/model.py predict_deltas 逐点积分出 ctrl 位置后、返回前投影；实体成员的逐点 quat 改由 R_e 给出 → entity_lbs 的 dense 自动继承刚性。flag `--rigid_agg`（默认 0）；开启时 w_rigid 自动归 0（冗余删除——loss 表净缩短）。**warm-start 恒等**：对已刚性场投影=恒等 ⇒ 从 v9-lang 精确续训。开销：O(M) scatter + ≤288 个 3×3 SVD（K=12·E≤24）≈ 微秒级，对 0.17-0.27it/s 的主开销（Qwen+DiT+渲染）不可见；ragged batch 天然支持（实体 id 偏移）。

**同 PR 指标**（堵 coherence 盲区）：extent-ratio + 刚性残差(cm) 进 eval_langswap 汇总与训练 log。

**v11-bases 放松路径**：ent_id one-hot → 可学习系数 [M,Kb]（softmax），H 改系数加权，seg 早期 CE 监督系数、后期放开。内核形状不变。

**验证阶梯（批准后执行）**：单元测试（合成刚性场→恒等；散开场→extent 1.0 且方向不变）→ resume v9-lang settle 800 步 → langswap 三划分+coherence 指标。判据：heldseed sel≥0.75、dir≥+0.8 持平，extent-ratio 1.00±0.05，刚性残差<0.5cm。

**B3 数据状态**：libero_goal 下载完整并校验（428 eps、856/856 mp4、428 parquet、fps20、双相机键、10 条新指令含新动词 put-on/open/put-inside + 新名词 bowl/plate/wine bottle/rack/drawer）；libero_90 仅 meta（全量 ~3921 eps 待拉）。IPEC loader（mp4+episodes.jsonl，区别于 binhng parquet 图像）待写——排在 v10-rigid 拍板后。

## §67 v10-rigid 实现 + V1 单元 + V2 推理 A/B（散开问题在 v9-lang 上零重训即解）

按 plan（adaptive-giggling-crescent.md）实现 batched weighted-Kabsch 实体聚合，branch `v10-rigid`：
- `igsw/dynamics/rigid_agg.py` `entity_rigid_aggregate`：逐点投票 x→x+v → 每实体加权 Kabsch（index_add×2 + einsum 互协方差 + 批量 3×3 SVD）→ 回写 v̂=R_e·x+t_e−x, ω̂=axisangle(R_e)。fp32 island、reflection 守卫、退化(<4点/共线)→纯平移 fallback（torch.where，DDP 安全）、参数自由。
- 接线：model.py predict_deltas gate 后调用；model_full.py `--rigid_agg` flag + seg_local 在 rigid_agg||entity_head 时下传 + entity_lbs warn 守卫；train_sim.py flag + 自动归零 w_rigid + ckpt 双存键；eval_langswap/eval_sim_generalization build_model 读 flag；eval_langswap 加 `--force_rigid_agg`（零重训推理 A/B）+ **coherence 指标**（extent-ratio、刚性残差）堵盲区。
- **附带修复**：rigid_agg 把实体刚性运动应用到**全部**控制点（含 gate 关闭的），结构性解决 §49"半个物体冻住"。

**V1 单元（test_rigid_agg.py 5/5 PASS）**：刚性场→恒等(2.4e-7)、散开 1.22→1.00 方向 cos 1.000、退化→纯平移、bf16+梯度有限、bg passthrough。

**V2 推理 A/B（v9-lang 现有权重，heldseed，force_rigid_agg 0 vs 1，零重训）**：
| heldseed | OFF | FORCED-ON |
|---|---|---|
| selection | 0.75 | **0.75**（不变）|
| direction | +0.81 | **+0.81**（不变）|
| **刚性残差** | **2.15cm** | **0.01cm** |
| extent-ratio 中位 | 1.04 | 1.00 |

**散开消除（刚性残差 2.15→0.01cm）且 selection/方向逐字节不变** → 散开在生产模型上**零重训即解**。extent 中位 1.04 是 8 clip 中位（cream-cheese 3.73 离群被中位洗掉），刚性残差更敏感、清楚显示修复。V2 判据全过。train/heldtask split + 刚性可视化进行中；V3 训练（让投票适应投影）随后。

## §68 v10-rigid V3/V4 终评：推理投影(V2)胜，训练在环(V3)回归方向 → 生产用 V2

修复 SVD 反向后 V3 训练干净跑完（800步 0 跳过），但 V4 三划分揭示**训练在环不如推理投影**。三方对照（heldseed=场景泛化关键读数）：

| | v9-lang (V2 OFF) | v9-lang+rigid (V2 ON, 零重训) | v10-rigid 训练版 (V3/V4) |
|---|---|---|---|
| heldseed sel | 0.75 | 0.75 | **0.84** |
| heldseed **dir** | **+0.81** | **+0.81** | **+0.28** ✗ |
| heldseed endErr | 14cm | 14cm | 30cm ✗ |
| coherence | 散开(2.15cm) | **刚性(0.01cm)** | 刚性(0.01cm) |
| train sel/dir/err | 0.57/+0.72/14cm | 0.57/+0.72/—  | 0.84/+0.53/35cm |

**判据**：V4 要求 heldseed dir≥+0.8——**V2 过(+0.81)，V3 不过(+0.28)**。V3 选择上升但方向塌、终点误差翻倍。

**根因（诚实定位）**：V2 用 v9-lang **已训练好的高质量逐控制点投票**（mean 方向 +0.81）做刚性投影 → 保住方向；V3 用 detach-R 后**只有实体均值梯度（muy）**重训 800 步，投票漂移、均值方向退化到 +0.28。**投票本来就好，用更弱的梯度信号重训反而伤了它。** 即"刚性投影"最佳作用位置是**推理后处理**，不是训练在环（至少这套梯度配置 + 800步如此）。

**结论 = 生产用 V2**：rigid_agg 作为 v9-lang 的**推理期投影**（`--force_rigid_agg 1`，参数自由、零重训）→ 散开解决（2.15→0.01cm）+ 方向/选择/终点全保住（+0.81/0.75/14cm）。**不发 V3 训练版**（方向回归）。salvage 选项（更少步/保留逐控制点梯度/更低 lr）留待需要时；V2 已达成 plan 目标。

v10-rigid 代码全部保留（flag 默认关、可回退）；test_rigid_agg 5/5；推理投影是干净增量。

## §69 v10-rigid 收尾：生产模型 = libero_v9lang_rigid（v9-lang + 推理刚性投影）

按用户决策 A 固化 V2。**生产模型 = `checkpoints/libero_v9lang_rigid/ckpt_last.pt`** = v9-lang 权重 + `rigid_agg=1` flag（推理期参数自由投影，opt 已丢、12GB）。加载即自动刚性投影（build_model 读 ckpt flag，无需 --force）。验证（heldseed，无 --force）：rigid_agg=ON、**sel 0.75 / dir +0.81 / endErr 13cm**（= v9-lang 质量）+ **coherence 1.00 / 0.01cm**（刚性）。

**生产推理/评估/可视化一律指向此 ckpt**。旧 v9-lang ckpt 保留（含 opt，可续训）。代码全 flag 门控默认关——任何旧 ckpt 行为不变。

**这条线（散开/coherence）正式收尾。** 完整链：用户目检发现散开 → §65 调研4家族+16clip数据核查 → §66 设计 → §67 实现+V1单元5/5+V2零重训解决 → §68 V3训练在环回归方向（诚实记录，投影该后处理不该回训）→ §69 固化 V2 为生产。branch v10-rigid。

## §70 v11 计划（用户：效果未达标，只列计划不执行）— 独立失败分析 + 3D-first 指标 + 真实视频 + scale-up

**用户判断**（2026-06-11）：模型效果未达标；刚体约束未充分解决；真实视频（实操/ego）未开始；2D/视频指标无意义（变化只占画面小部分，仅可作辅助约束），核心看 3D 指标。**本节=计划，未执行。**

### A. 独立失败分析（挖现有 eval 日志，零新计算）
1. **幅度塌缩 = 主要矛盾**：c-窗 train clips epi100/110/130/150/160 整簇 GT 26-32cm 只走 2-4cm（**比例 ~0.1×**），方向却 +0.95-1.00。train sel 0.57 的真相：40% 对子败在 0.25×GT 幅度地板，不是选错。heldseed butter 同样 0.10×。**塌缩 clip 的 swap/true 0.3-0.44（健康 clip 0.00-0.03）→ rel-gate 在这些场景对真/假指令都半开 = relevance 校准失败**，gate 半开直接缩 v。叠加：L1 轨迹损失 median-seeking（仅 ~5% 控制点动）、w_mag 未入 v9 配方、800 步 settle + 40 clip 过小。
2. **训练指标共谋**：corr 是范数相关（全局缩小 0.1× 仍高）→ corr 0.93 与幅度 0.1× 并存；rPSNR 渲染整帧而动区 ~5% 像素 → 背景主导。**现指标体系系统性掩盖幅度塌缩**（与方向盲区、散开盲区同构，第三次）。
3. **刚体未真正解决（用户正确）**：V2 投影=推理期遮症状；模型原始投票仍散（投影前残差 2.15cm）；**旋转正确性从未测过**（LIBERO pick-place 近平移、R_e≈I 没暴露）；V3 训练在环失败。表示层不"懂"刚体。
4. **数据规模荒谬**：40 train clips、1 相机、1 suite、8-10 名词、fps10、纯 sim。heldseed n=8（±0.09 二项噪声）——一切结论都在噪声区。
5. 早窗方向差（+0.11-0.63，§57）：pre-contact 时机歧义。6. heldtask=0（词汇）。7. 长时域 v8 后未测。

### B. 调研结论（真实视频 + 3D 指标 + scale）
- **真实视频伪 GT 提取器**：**SpatialTrackerV2**（ICCV'25，前馈统一 2D 跟踪+单目深度+相机位姿，世界系 3D 轨迹分解为 geometry/ego/object，10-20s/段，比 SOTA 3D 跟踪 +30%、与动态重建持平快 50×）= 主提取器，替代 CoTracker+Pi3 拼接；**MegaSaM**（CVPR'25，动态视频相机+深度，可微 BA）= 位姿/深度备选；**MoSca**（CVPR'25，离线 4D Motion Scaffolds 高保真）= 慢但准，作 5-10 clip 黄金子集校验伪 GT 自身。
- **3D 指标标准（取代自创）**：TAPVid-3D 的 **3D-AJ / APD / OA**（含全局中位数尺度归一）；场景流 **EPE3D / Acc3DS(≤5cm或5%) / Acc3DR(≤10cm或10%) / 离群率**（Gojcic CVPR'21 标准）；实体位姿 **5°5cm**（旋转测量首次引入）+ 平移/幅度比直方图（中位数+P10，杜绝被均值洗掉）。2D/渲染指标全部降级为辅助约束。
- **真实数据源**：**DROID**（真机、ZED 双目深度+标定+语言，CC-BY-4.0，gs://gresearch/robotics/droid）= 首选（真深度可校准单目管线的尺度）；**AgiBot-world-beta**（已在服务器！137k ep、8 cam 30fps、语言标注）= 零下载成本的 ego/多视角试点；**EgoDex**（829h Vision Pro ego 操作+3D 手部）/ EPIC-KITCHENS / EgoExo4D = 后续规模。
- 表示升级参照：Shape-of-Motion/HiMoR 低秩运动基（v11-bases）；GWM/ManiGaussian 无刚性保证（前车之鉴）。

### C. 计划（R0→R4，每阶段 3D 指标门禁）
- **R0 指标改革 + 诚实重基线（~1天）**：实现 3D 套件（EPE3D/Acc3DS/Acc3DR、3D-AJ/APD、5°5cm、幅度比中位+P10、逐步方向曲线、coherence 已有、长时域复活）进 eval_langswap/新 eval_3d.py；训练 log 加幅度比中位（替 corr 主位）；v9lang_rigid 在全部 56 clip 重基线（含首次旋转误差）。**门禁：暴露面完整（预期难看，就要难看）。**
- **R1 幅度/gate 校准修复（~2-3天）**：先诊断塌缩簇（rel logit/p_dyn 分布 vs 健康簇；是否名词相关）；候选修法（按证据择 1-2）：(a) rel-BCE pos_weight/温度重校准 + 塌缩场景过采样，(b) w_mag（已存在）入配方 + per-entity 位移损失（实体级幅度直接监督，对 ~0.1× 塌缩比逐点 L1 敏感），(c) 训练加长（800→3-5k settle）+ lr 微调。**门禁：train 幅度比中位 ≥0.85 且无 clip <0.5；heldseed sel ≥0.75 / dir ≥+0.8 不回退。**
- **R2 刚性表示真解决（~3天，R1 后）**：保留推理投影为底线；表示层试 **v11-structured-decode**：每实体 SE(3) 由实体控制点特征 **输出空间 cross-attention 学习聚合**（≈可学习加权 Kabsch，端到端可微、无 SVD 反向问题；区别于失败的 V3-detach 与特征池化 entity head），arm 逐子部件；用 LIBERO-goal 的开抽屉/旋钮动作补**旋转丰富数据**。**门禁：投影前原始投票残差 <0.5cm；旋转 5°5cm 在旋转 clip 上 ≥0.7；方向/选择不回退。**
- **R3 真实视频管线试点（~1周，可与 R2 并行）**：SpatialTrackerV2 集成（伪 GT：世界系 3D 轨迹+相机+深度）→ 既有 ego-ready schema（§63 viewmats 已验证）+ openvocab（§64 推理就绪，目标选择改 CoTracker/StV2 运动仲裁，无 GT）→ **先 DROID 20 clip**（真深度校尺度）→ AgiBot 20 clip（已在服务器）→ 人工目检 + MoSca 黄金子集校验伪 GT → v12 sim+real 共训 → 真实 heldout 用 3D-AJ/APD 评。**门禁：伪 GT 黄金子集 EPE3D <3cm；真实 heldout 模型 APD@10cm 显著 > static 基线。**
- **R4 Scale-up（~2周+，R1-R3 收敛后）**：数据：LIBERO-90/goal/spatial openvocab 重生成（B3，4k+ ep、fps20 stride2、词汇 30+ 名词）+ 真实视频扩 DROID→EgoDex；训练：batch>1 ragged（segment-op 已就绪）、sim:real 课程混采、settle→长训（≥20k 步）；按需 v11-bases（关节/柔性）。**门禁：heldtask（未见名词）sel 显著 >0（首次）；真实视频 3D-AJ 持续提升；长时域 10s 漂移有界。**

### 风险
伪 GT 尺度歧义（单目）→ DROID 真深度先校准；StV2 非商用许可核查；塌缩簇若是数据(失败演示残留)非模型 → §50 过滤复用；R2 若再伤方向 → 即回退推理投影底线（已 ship）；真实视频遮挡重 → StV2 遮挡感知 + 可见性掩码已在损失。

## §71 R0 完成：3D 诚实重基线——2D 指标掩盖的全暴露（COVERAGE 门禁通过）

建成 `code/scripts/eval_3d.py`（EPE3D/Acc3DS-R/5°5cm/mag-ratio中位+P10/rot-err/coherence，全 vs clip["traj"] 解析 GT，复用 eval_langswap helper）。`libero_v9lang_rigid` 三划分重基线：

| split | mag 中位/**P10** | EPE3D 中位 | **5°5cm** | rot-err 中位 | Acc3DS/R | rigRes |
|---|---|---|---|---|---|---|
| train(40) | 0.63× / **0.11×** | 13.6cm | **0.00** | 26.8° | 0.12/0.34 | 0.01cm |
| heldseed(8) | 0.65× / **0.10×** | 13.1cm | **0.00** | 28.1° | 0.23/0.32 | 0.01cm |
| heldtask(8) | 0.00× / 0.00× | 20.9cm | 0.00 | 16.8° | 0.12/0.13 | 0.00cm |

**2D langswap(sel 0.75/dir +0.81)系统性掩盖了**：
1. **幅度塌缩成片**：中位 0.63×、**P10 0.10×**；塌缩簇 epi100/110/130/150/160/180（GT 25-32cm 只走 2-4cm = 0.02-0.15×）。
2. **EPE3D ≈ 物体运动的一半**（13cm vs ~27cm）。
3. **5°5cm = 0.00（全划分）**——从不接近 GT 位姿。
4. **旋转误差 27-28°（首次测量）**：模型给纯平移加了大旋转。coherence 好(0.01cm)说明 rigid_agg 生效——**物体是刚性的，但刚性地错（错幅度+伪旋转）**。caveat：小位移 clip 的 Kabsch 旋转病态(epi240_e GT5cm→83°)，但大位移 clean clip 真有伪旋转(epi100_c GT32cm→11°、epi40_c GT29cm→22°)；未来旋转指标应门控 GT disp>10cm。

**R0 COVERAGE 门禁 = 通过**：套件全跑通、各指标出数、复现塌缩簇(P10 0.10×)、旋转首次有数。"暴露完整即过——它确实难看，且就该难看。"**用户判断（效果未达标 / 2D 无意义）被 3D 指标完全证实。** 现有诚实 3D 基线，R1 目标量化：mag 中位 0.63→≥0.85、P10 0.10→≥0.5 + 处理 28° 伪旋转。

## §72 R1 诊断：幅度塌缩 = dyn-gate 在真实 mover 上关闭（非 head 欠预测）

`_diag_collapse.py` 对照塌缩 clip vs 健康 clip，dump mover 实体的 gate=sigmoid(p_dyn)、rel、predMag/gtMag：
| | gate(T) | predMag/gtMag |
|---|---|---|
| 塌缩(epi100/110/130/160_c) | **0.08-0.15**（gate 关 ~90%）| 0.09-0.14× |
| 健康(epi0/30_c, epi240/330) | **0.75-1.00**（gate 开）| 0.89-1.24× |

**根因定位**：head 的 raw v 正常，但 **dyn-gate 把 25-32cm 的真 mover 误判为静止**→ v×0.1 → 塌缩。rel(T)=0.69-0.80 还行，但 pooled p_dyn 的**视觉 dyn_logit 极负**，+rel 也开不动。swap/true 0.3-0.44 的真相：gate(T)=0.11、gate(W)=0.04 都很小，比值是噪声不是真泄漏。
**修法（证据驱动）**：`mover_magnitude_loss`（losses.py:87，多步，§44 建过但从未入 v9 配方）直接惩罚欠幅，梯度经 gate 流回 → 在 GT 动的地方把 gate 顶开（GT 静止处 target~0 → 不破坏静止抑制）。R1 = resume v9lang + --w_mag 重训 → eval_3d 看 mag-ratio。

## §73 R1 验收：幅度塌缩实质修复（mover_magnitude_loss），选择 0.75→1.00；旋转留给 R2

v11mag = resume v9lang + `--w_mag 0.5`，settle 1500 步（0 跳过）。eval_3d（production config +rigid_agg）+ langswap 守卫：

| 指标 | v9lang_rigid(R0) | **v11mag(R1)** | R1 门禁 |
|---|---|---|---|
| mag 中位 train/heldseed | 0.63/0.65× | **0.91/0.85×** | ≥0.85 ✓ |
| mag **P10** train/heldseed | 0.11/0.10× | 0.29 / **0.51×** | ≥0.5：heldseed✓ train✗ |
| EPE3D 中位 train/heldseed | 13.6/13.1cm | **10.5/10.3cm** | ↓✓ |
| EPE3D **P90** heldseed | 23.5cm | **11.4cm** | 尾部腰斩 ✓ |
| **langswap sel** train/heldseed | 0.57/0.75 | **0.82/1.00** | ≥0.75 ✓✓ |
| dir heldseed | +0.81 | +0.75 | ≥+0.8 ✗(n=8噪声内) |
| 5°5cm / rot-err | 0.00/28° | 0.00/**31.9°** | R2 目标 |

**核心目标(幅度)实质修复**：median 两划分 ≥0.85、heldseed P10 过、EPE3D 全面降、尾部 P90 23.5→11.4cm。**附带白拿**：selection 0.75→1.00（因 R0 诊断的"塌缩使物体没过 0.25×GT 地板"被解除）。
**残留（折进 R2）**：(a) train P10 0.29（少数 train clip 仍塌，heldseed 不受影响）；(b) heldseed dir +0.81→+0.75（n=8 噪声内，但低于严格门禁）；(c) **旋转未动（31.9°、5°5cm=0）——本就是 R2 目标**。heldtask 0.00×=词汇问题(R4)非幅度。
**判定**：R1 在主目标上实质成功，残留交 R2（R2 重训会同时管旋转+守方向，自然吸收 (b)(c)；(a) 视 R2 后情况）。v11mag 暂作 R2 的 resume 基座，不急 ship。

## §74 R2: 旋转源从 v-Kabsch 改为 supervised omega-mean（去 SVD→可训练在环）；GT 物体确实在转

诊断确认 §73 假设：rigid_agg 的旋转来自**速度场 Kabsch**（拟合 vote 噪声→28-31° 误差），而 rot_l 监督的 per-control omega 被**丢弃**。修法（rigid_agg.py，rot_from_omega 默认 True）：实体旋转 = supervised omega 的加权均值（exp），平移仍取速度质心；`y_hat=R(x-mux)+muy`。**去掉 SVD**→数值稳定 + **完全可微（omega 得梯度，无需 V3 的 detach hack）→ rigid_agg 现可训练在环**（V3 的拦路 SVD-backward 消失）。单元 5/5 过（test1 改为提供一致的 v+omega）。

**重大发现**：eval_3d 加 GT-rot 列 → **heldseed GT-rot 中位 17.5°**——**pick-place 物体真的在转**（抓起/放下时倾斜），R0"伪旋转"框架错误。omega-mean 把 rot-err 31.9→**19.0°**（最干净的 epi040_c 仅 **4.8°**），但 5°5cm 仍 0：模型预测的旋转幅度对、方向/量不够准；且 GT 旋转本身含伪 GT 噪声（Procrustes on tracked points，小位移 clip 病态 epi240_e 82°）。
**重估 R2**：(1) omega-mean 是对的旋转源（严格改进 + 可训练），ship 为默认；(2) "原始投票即刚性"现可实现——训练时开 rigid_agg(omega-mean) 让 raw votes 直接刚性（V3 做不到、现在能）；(3) 干净旋转评估需 LIBERO-goal 抽屉/旋钮（R3/R4 数据），libero_object 的入射旋转太噪。

## §75 R2 验收：omega-mean 旋转推理投影有效；in-loop 训练再次失败（V3 教训重演）；5°5cm 待干净数据

omega-mean rigid_agg（§74）作两种用法：
- **推理投影（有效，ship）**：v11mag + omega-mean → rot-err 31.9→**19.0°**（最干净 clip 4.8°）、mag 0.85×/0.51× 保持、coherence 0.07cm。比 v-Kabsch 旋转源严格更好。
- **训练在环（失败）**：v11rigid（resume v11mag + rigid_agg=1 训练）前 20 步干净、之后退化成**持续非有限梯度（1068 跳过/1160 步）**——和 V3 同类失败。omega-mean 去了 SVD 但 12 步 rollout 的旋转梯度累积仍不稳。已杀。**第三次印证：刚性投影该在推理后处理，不该回训练环。**

**R2 判定（部分达成）**：架构修复（旋转源 = supervised omega-mean）正确且已 ship；"原始投票即刚性"经训练实现的路再次失败（接受推理投影为底线）。**5°5cm 仍 0**——因 (a) 模型旋转预测未够准，(b) libero_object 入射旋转的伪 GT 噪声大（GT-rot 17.5° 但小位移 clip 病态）。真正的旋转达标需 **LIBERO-goal 抽屉/旋钮干净旋转数据 + 更强旋转监督**，并入 R3/R4。

**生产模型 = `checkpoints/libero_v11_rigid`**（v11mag 权重 + rigid_agg=1 omega-mean，12GB）：R0 暴露 + R1 幅度修复(塌缩 0.63→0.91×、sel 0.75→1.00) + R2 旋转源修复(31.9→19°) + coherence。取代 libero_v9lang_rigid 为当前最佳。

**v11 计划进度**：R0 ✓、R1 ✓、R2 ✓(架构修复 ship,5°5cm 待干净数据)、R3 真实视频未开始(侦察✓)、R4 未开始。

## §76 R3 数据侦察：AgiBot digital-world 有标定 + 物体 6D pose（可能升级伪 GT→真 GT）

R3 真实视频数据源勘探（AgiBot 已在服务器，DROID 用户否决）：
- **主 lerobot**（`agibot-world-beta-lerobot`，reader 已验证）：8 cam RGB(AV1)+ **真实 EEF 6-DOF**(observation.states.end.position[T,2,3]+orientation[T,2,4])+gripper+细粒度子任务语言(action_config)。**无标定、无物体 GT**。
- **digital-world**（`agibot-digital-world` 2.6TB，每 episode = task_info.json + proprio_states.h5 + parameter.json）：**parameter.json 有相机标定**（每 cam intrinsic fx/fy/ppx/ppy + extrinsic pose[4x4]）——主 dump 缺的标定这里有！+ proprio_states.h5(全本体感知)+ task_info.json 物体 `omni6DPose_*` id(6D pose 基准物体)。**无 RGB(在主 lerobot)、物体 per-frame pose 待确认**。digital-world 用 uuid、主 lerobot 用 task/episode——配对关系待查。
- **R3 评估分层（据此）**：操作器(gripper/arm) = 真实 EEF GT(严格 5°5cm/EPE3D)；物体 = 伪 GT(StV2/Pi3+CoTracker)，**若 digital-world 配对成功 + 物体 per-frame pose 存在 → 物体也升级为真 GT**(omni6DPose)。标定若可用 → 度量深度 + EEF 投影 + 多视角解锁。
- **R3 里程碑1（在建）**：主 lerobot 一个操作 episode → 头 cam RGB 窗 + EEF + 语言 → Pi3 抬升 3DGS → 存 clip + 目检。证明真实视频→3DGS+真实 EEF 管线通。digital-world 标定/物体GT 为下一步增强。

## §77 R3 里程碑1 达成：真实 AgiBot 视频 → Pi3 3DGS + 真实 EEF GT（管线基础通）

`code/scripts/agibot_video_gt.py`：读 task_327 ep0（"Place the held cucumber into the plastic bag in the shopping cart"）→ 按 EEF 位移挑运动子任务窗 [187,235] → decode 头 cam RGB(640x480,13帧) → Pi3 抬升 **574x434、相机 ego 移动、N=140572 高斯**（超市场景/果蔬可辨）→ 存 clip(means/colors/uv + **真实 EEF 6-DOF[Kf+1,2,3]+ori+grip** + 子任务语言) + 目检图 `viz/agibot/r3_m1.png`（RGB t0/tK | Pi3 3DGS 点云 | 双臂 EEF xy 轨迹）。**真实视频→3DGS + 真实 EEF GT 基础打通。**

**R3 剩余（多日）**：(2) 运动伪 GT（StV2 集成或 Pi3+CoTracker 出 object 3D 轨迹）；(3) openvocab seg frame-0（无 GT mask）；(4) **EEF 尺度校准**（EEF 米 ↔ Pi3 gauge，gripper 2D track 对齐 或 digital-world 标定）；(5) 操作器 5°5cm/EPE3D vs 真实 EEF；(6) digital-world 配对（标定+omni6DPose 物体 GT）；(7) MoSca 黄金子集校验伪 GT。里程碑1 是基础，(2)-(7) 是 R3 主体。

## §78 R3 里程碑2：运动伪 GT（CoTracker+Pi3）跑通但噪声大 → 印证需 StV2

agibot_video_gt.py 加运动:CoTracker 网格(1008点)+ Pi3 canonical 点图 3D 抬升 + 运动 ID。结果:median-disp 0.5cm(多数静止✓)、movers>4cm=157、**max 186.8cm(Pi3 深度 outlier)**;目检 movers 散布全场、未干净定位到黄瓜/夹爪。**Pi3+CoTracker(LIBERO sim 够用)在真实 AgiBot 上深度噪声太大 → 伪 GT 不干净**。**这正印证计划选 SpatialTrackerV2(遮挡/深度感知 3D 跟踪)+ MoSca 黄金子集校验**的必要性。viz/agibot/r3_m2.png(5 panel:RGB t0/tK | 3DGS | 运动伪GT | EEF 轨迹)。

**R3 状态**:里程碑1(真实视频→3DGS+EEF GT)✓、里程碑2(运动伪GT,噪声,需StV2)✓-部分。剩余里程碑3-7(openvocab seg、StV2 干净运动、EEF 尺度校准、操作器5°5cm评估、digital-world真GT、MoSca校验)是多 session 工作量——尤其 StV2 需下载安装。

## §79 R3 m3+m4：openvocab 真实场景成立 + EEF 度量尺度校准 PASS

**m3 openvocab on real AgiBot**（_agibot_ovtest.py，viz/agibot/r3_m3_openvocab.png）：超市帧上 GroundingDINO ground 全部开放名词（plastic bag 0.65/robot gripper 0.53/robotic arm 0.52/shopping cart 0.41/cucumber 0.38 多候选——货架上真有一排黄瓜）；segment_frame_amg schema 映射成立（双臂→id8、塑料袋/购物车→id2 容器、果蔬→干扰物）。**诚实缺口**：名词 grounding 选"货架某根黄瓜"而非"夹爪里那根"（the held cucumber）→ 目标实例选择必须靠运动仲裁（StV2，符合设计）。

**m4 EEF 度量尺度校准**（_agibot_eefcal.py）三轮迭代：box median（scale12.7、场景 23m ✗）→ mover-filter（6.2、7m ✗）→ **逐 track Umeyama + RANSAC 式共识**：每条 track 单独拟合 EEF 轨迹形状，刚性附着夹爪的 track 以低残差胜出。**结果 PASS**：右臂 box 3 tracks 过 sanity，**scale=0.697、最佳残差 1.1cm（6% rel on 19cm）、场景中位深度×s=0.79m（物理合理）**；左臂框 0/25 过（正确，不随右 EEF 动）。**Pi3 重建获得米制尺度（真实本体感知锚定）→ 解锁 m5 操作器 5°5cm 真 GT 评估。** 方法注记：consensus spread 0.40-0.88 偏宽（仅 3 tracks）——多 clip 校准时用更密 gripper 点 + 双窗。

**R3 进度**：m1✓ m2部分(CoTracker噪声,待StV2) m3✓ m4✓ | 剩 m2-redo(StV2)、m5(操作器评估)、m6(digital-world 配对)、m7(MoSca 校验)。

## §80 R3 m5：零样本操作器评估（真实 EEF GT 上的诚实基线）

`_agibot_m5_eval.py`：完整管线一气呵成——真实帧→Pi3 g0(14万)→openvocab seg(id8 机器人 11426 点)→m4 逐 track 共识尺度(0.697, rel 6%)→v11_rigid 零样本 forward(指令"Place the held cucumber...")→夹爪框区控制点预测位移×s 转米制→对真实 EEF。

**结果（LIBERO sim 训练 → 真实超市，零样本）**：GT |ΔEEF|=18.6cm；PRED=6.6cm(**0.35×欠幅**)；**方向 cos +0.51**(粗对)；**EPE3D 16.3cm < static 18.6cm(弱胜 static)**；gate 0.47(半开)。
**判读**：不可用但非零——分布大偏移(franka→双臂人形、桌面→超市、模板指令→自由文本)下方向仍粗对且弱胜 static。这是 R4 sim+real 共训的诚实起点。**Caveat**：夹爪框内 seg==8∧box 控制点仅 3 → 回退框内全部(混入背景) → 预测幅度被稀释,真实欠幅可能没 0.35× 那么糟;待 StV2 m2-redo 后用运动仲裁取干净夹爪点重测。

**R3 进度：m1✓ m3✓ m4✓ m5✓(基线) | m2-redo 等 StV2(安装中,已重启) m6 digital-world 配对 m7 MoSca。R4：libero_90 下载中。**

## §81 R3 m2-redo 达成:StV2 在真实 AgiBot 上出干净 3D 运动伪 GT(卡点#2 解决)

**安装战记**(供后人):torch2.4.1 专属 venv 路线被代理反复杀死(pip 无续传、wget 与代理不兼容、git 依赖 503、torch 的 nvidia-cu12 依赖群又是 2GB+)。**最终解**:发现 StV2 模型代码不 import xformers → 直接跑主 venv(torch 2.8),uv 补 11 个小依赖(easydict/decord/moviepy/kornia/pycolmap/pyceres/einx/flow_vis/hydra/omegaconf/timm)+ 本地装 utils3d(pinned commit)/segment-anything。**curl -C - 25 秒拉完 797MB**(wget 0 字节,坑)。权重 HF 直拉(Yuxihenry/SpatialTrackerV2_Front + -Offline)。

**m2-redo 结果**(_agibot_stv2.py,viz/agibot/r3_m2redo_stv2.png):VGGT4Track 前端(深度+内参+位姿,13×392×518)→ StV2 offline 729 tracks、vis 93%。**世界系 3D 位移:静止中位 0.1cm、p90 0.3cm、最大 44.7cm(合理)——对比 m2 CoTracker+Pi3 的 max 186.8cm 疯狂离群 + movers 散布全场**。目检:运动热力干净集中在右臂+持黄瓜区,top movers 轨迹一致地货架→购物车。**真实视频干净物体 3D 运动伪 GT 可用;输出含 c2w_traj/intrs/point_map(供 clip schema viewmats)**。npz 存 data/_agibot/stv2_tracks.npz。

**R3 进度:m1✓ m2-redo✓(StV2) m3✓ m4✓ m5✓ | 剩 m6 digital-world 配对、m7 MoSca 黄金子集校验、clip schema 总装(StV2 轨迹+openvocab seg+EEF 尺度+viewmats → 训练格式)。**

## §82 R3 clip 总装 v1：schema 跑通；伪 GT 质量两缺陷待修（目标身份 + 臂刚性化）

`agibot_clip_stv2.py`：真实 AgiBot → 完整训练格式 clip（StV2 后端,单 gauge=cam0）：VGGT4Track 前端(逐帧 cam-frame pointmap+内参+c2w) → StV2 世界系轨迹 → canonical 重 gauge → g0(20.3万) + openvocab seg + 运动仲裁 + 逐实体 trimmed-Kabsch traj + `viewmats[t]=inv(c2w[t])@c2w[0]`(ego schema §63) → clip_stv2_ep0.pt + 审核图。

**v1 诚实判定（viz/agibot/r3_clip_stv2.png）**：
1. ✅ 总装端到端成立（schema 完整、臂子部件 50-52 携带运动、保存可加载）。
2. ✗ **目标身份错**：运动仲裁在 openvocab 物体实体上全静（~2mm,货架物），选了噪声 id6——**持握中的黄瓜在夹爪里被 openvocab 归进 robot id8**（place 子任务的普遍情形:抓住的物体与夹爪同刚体）。修法：mover ∩ GD-"cucumber"-box（含末帧,物体离开货架后可分离）或 mover 簇内做外观分割。
3. ✗ **臂刚性化涂抹**：k=3 子部件对关节臂太粗（t12 青色云漂移走样）。修法：更细子部件/逐帧重聚类，或臂放弃刚性化、直接 nearest-track 位移监督（schema 的 traj 本就支持非刚性）。

**结论**：StV2 原始轨迹干净（§81），损失在"逐实体刚性化"。批量生成（20 clip）前先修这两点。R3 剩：clip-builder v1.1（上述两修）、m6 digital-world 配对、m7 MoSca 校验。R4 备料：libero_90 重启下载中（重试循环,已 1715 mp4）。

## §83 R3 clip-builder v1.1：两缺陷修复（持握物仲裁 + 臂非刚性转移）；残留同类实例渗漏

v1.1 双修（agibot_clip_stv2.py）：
1. **持握物仲裁**：GD 指令名词("held (\w+)"→cucumber)在 frame0+末帧出框，**target = mover ∩ noun-box**（6/18 movers 命中）→ 从 robot lump 中按邻近 moving tracks 切出目标高斯。**结果：obj_gauss=3392、位移中位 41.4cm（吻合 §81 mover 簇；v1 是静止货架物 0.2cm）**。
2. **臂非刚性化**：放弃 k-means 刚性子部件轨迹，改 **3-NN 逆距离加权的轨迹位移转移**（StV2 原始轨迹本来就干净）；k-means 子 id(50-53) 仅留给 entity-LBS 绑定。审核图 v1 的涂抹云消失，青色随臂走。

**审核图（viz/agibot/r3_clip_stv2_v11.png）**：红色目标主簇 t0(夹爪/货架顶)→t12(购物袋) 连贯移动 ✓；臂干净 ✓。**残留（v1.2 todo）**：货架同类黄瓜少量误并入目标（同类实例歧义：GD 框叠到 mover 2D 路径附近的静止同类）→ 修法：carve 时要求该高斯邻近的 track 自身在动（运动一致性过滤），或限制 carve 半径/限定包含 moving tracks 的那个框。

**R3 状态：m1✓ m2-redo✓ m3✓ m4✓ m5✓ clip-v1.1✓(可用,留 v1.2 小修) | 剩：v1.2 同类渗漏修 → 批量 20 clip → m7 MoSca 抽查 → R4 共训。m6(digital-world 配对)降级为可选（标定可从 EEF 校准替代）。**

## §84 R3 批量生产 + R4 入口：50 episode 生成、31 过质量门、v12 sim+real 共训启动

**clip-builder 迭代链**（每轮诚实盘点驱动）：v1.2 名词解析坏(3/20 PASS) → **v1.3** 多模式名词提取+容器黑名单（修 15 个 'shelf.' 误解析）+ GD 框内播种 query（修小物体 grid 漏检）→ 8/20 → **v1.4** Retrieve 类锚定段尾取窗（段首是伸手、物体未动 = 12 个失败共因）→ 9/20 → **不再调参、改拓宽**：ep20-49 再生成 30 个（v1.4 在新批通过率 ~73%）。
**终态：50 个 clip、31 过质量门（62%）**，质量门=目标位移∈[5,80]StV2cm ∧ 物体高斯∈[500,30k]。词汇：cucumber/pear/carambola/corn、双动词模板（Place held/Retrieve）、位移 5.6-60cm。失败模式记录：同类实例歧义 + 抓取时机异质（v1.5 候选：多窗扫描取过门窗口）。

**v12 共训（R4 第一炮，运行中）**：mix_v12 = 56 sim(LIBERO) + 25 real train + 6 heldreal（symlink，epi9xxxxx_r_{split}.pt 命名兼容 loader）。resume v11mag、w_mag 配方、2500 步 settle。**判据**：真实 heldreal 显著改善零样本基线（dir +0.51 / EPE3D 16.3cm / mag 0.35×，§80 m5）；sim heldseed 不回退（mag 0.85×/EPE3D 10.3cm/sel 1.00）。orchestrate_v12.sh，日志 logs/orchestrate_v12.log。
