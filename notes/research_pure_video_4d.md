# Pure-Video → (frame-0 3DGS + per-Gaussian 3D motion): feed-forward data-gen pipeline

> Research note for Instruct-GS-World. Goal: replace the clean-SIM GT data-gen (`maniskill_gt.py`, §38-44) and
> the LIBERO-GT data-gen (§45) with a pipeline that works from **ONLY an RGB manipulation video** — no depth
> sensor, no intrinsics/extrinsics, no poses — by ESTIMATING camera + geometry + 3D motion feed-forward, and
> emitting the **same clip schema** the dynamics model already trains on. The model (gate + sem→gate + magnitude
> loss, §44k) is unchanged; only the data producer changes.
>
> Date: 2026-06-09. Cites arXiv + our clones (file:line). Decision-grade; we build the top pick next.

---

## 0. TL;DR — the recommendation

The §38 failure ("Pi3 depth + CoTracker 2D→3D lift, too occlusion-noisy") happened because we **chained two
brittle single-purpose models** (monocular depth that is per-frame-inconsistent, + a 2D tracker that has no 3D
or occlusion notion) and lifted 2D→3D ourselves. The 2024-2026 SOTA solved exactly this by making **one
feed-forward network jointly estimate depth + camera + per-pixel 3D motion + visibility**, trained end-to-end so
the depth is temporally consistent and the tracks carry occlusion scores. That is the whole fix.

**Top recommendation (build this):** **SpatialTrackerV2** (ICCV'25, arXiv 2507.12462) as the single
RGB-video→{per-frame depth, camera pose+intrinsics, per-pixel 3D tracks in world space, **visibility**,
**dynamic-vs-static** probability} engine. It is exactly the §39 pick ("the §39 pick" — agent.md:525) that we
named but never actually wired up because we pivoted to sim. One model covers research questions **#1 (cam +
geometry)** and **#2 (occlusion-robust 3D motion)** at once. CC-BY-SA-4.0, code + HF weights, 10-20 s/clip.

- Build frame-0 3DGS by **unprojecting STv2's frame-0 depth** with STv2's frame-0 camera (one Gaussian/pixel),
  exactly as our existing `to_gaussians.py` does — but now with **consistent, occlusion-aware** geometry.
- Get the per-Gaussian trajectory by **querying STv2 at the frame-0 pixel of every Gaussian** (`uv` is already
  stored per Gaussian in our schema) → STv2 returns that pixel's full 3D world track + visibility → that IS the
  per-Gaussian motion. No separate lift, no 2D→3D guesswork.
- `seg_per_g` (mover/static + object id) comes from STv2's per-track **dynamic probability** for the gate, plus an
  open-vocab 2D segmenter (SAM2 / Grounded-SAM2 / the VLM referring mask) projected to Gaussians for the
  instruction's target object.

**Fallback / upgrade A (newer, denser):** **Track4World** (TencentARC, arXiv 2603.02573, Mar 2026) — VGGT-style
ViT → **dense 3D tracking of EVERY pixel** in world coords + geometry, one feed-forward pass, handles
occlusion/deformation, **and ships a `track4world_pi3.pth` variant** (reuses the Pi3 weights we already have, 3.6G
`checkpoints/Pi3/model.safetensors`). Dense-by-design removes the "query each Gaussian" step. License =
Tencent + CC-BY-4.0; weights on HF `TencentARC/Track4World`. **Strictly better fit if it installs cleanly** — it
is purpose-built for "dense first-frame→all-frames 3D tracking", which is precisely our per-Gaussian-motion need.

**Fallback B (already on disk, zero download):** **St4RTrack** (ICCV'25, arXiv 2504.13152) — weights ALREADY on
the server (`third_party/St4RTrack/checkpoints/{model.safetensors 2.2G, St4RTrack_wo_reweight/St4RTrack_release_version.pth 4.2G, MASt3R_base.pth 2.6G}`). Pure RGB, dual-branch (reconstruction + tracking) →
per-pixel 3D world trajectories + camera, 30 fps. Non-commercial research license. Use as the immediate
de-risk while STv2/Track4World are being installed.

**Camera+geometry-only fallback (if a tracker's geometry is weak):** **VGGT-Ω / VGGT-Omega** (arXiv 2605.15195,
FB) for dynamic-capable pose+depth, or **MonST3R** (repo on server, needs weights) — but prefer the unified
trackers so depth and motion come from the *same* consistent model.

**Do NOT** rebuild the §38 chain (Pi3-depth + CoTracker-2D + manual lift). Do NOT use DGS-LRM as the entry point
— it needs **posed** video (Plücker rays from known camera, arXiv 2506.09997 §inputs), so it is a *downstream*
quality booster, not a from-RGB solution.

---

## 1. The problem, precisely (why §38 failed and what "clean enough" means)

§38 (agent.md:519-521) diagnosed our AgiBot run as **noisy GT**, not a bad architecture:
- **mover-visibility only 0.40-0.66** — the hand grasps the object, the object goes behind things, and our
  2D→3D lift had no occlusion model, so occluded points got garbage 3D positions ("jumped/lost tracks (noise)").
- **GT max displacement up to 2.1× scene-radius** — tracks teleported.
- Two compounding defects: (1) Pi3 monocular depth = per-frame approximate (not temporally consistent →
  ghosting); (2) CoTracker(2D) + Pi3-lift = occlusion-fragile.

The clean-SIM pivot (§39-44) proved the **model** is correct given clean GT (dyngate7: held-out corr 0.85,
top-mover ratio 0.81, static-leakage 0.01 — §44k). So the only open problem is: **can a pure-video pipeline
produce GT clean enough that the model still learns?** The metrics that define "clean enough" are exactly the
ones `eval_sim_generalization.py` already computes (corr, top-mover ratio, static-leakage, Δ_null/Δ_wrong) — see
§6 for how we measure pipeline error against LIBERO GT *before* committing to a big run.

The single most important property a candidate must have for manipulation: **per-track visibility / occlusion
scoring**, so we can (a) drop or down-weight occluded segments of a Gaussian's trajectory instead of feeding
teleport noise, and (b) keep moving an occluded object by trusting the *visible* part of its rigid body.

---

## 2. Research question #1 — feed-forward camera + geometry from RGB video (dynamic)

| Model | arXiv | Dynamic? | Outputs | Pure RGB in? | Speed | License | On server? |
|---|---|---|---|---|---|---|---|
| **VGGT** | 2503.11651 | weak (static-trained; ghosts on motion) | pose enc (t,quat,fov), depth, world pointmaps, conf, 3D tracks | yes | <1 s, OOM ~300 fr | code commercial-OK; `VGGT-1B` weights **CC-BY-NC**, `VGGT-1B-Commercial` OK | repo only, **no weights** |
| **VGGT-Ω (Omega)** | 2605.15195 | **yes** (static+dynamic, +77% cam on Sintel) | same as VGGT, single dense head, register-based | yes | faster, 30% mem of VGGT | VGGT license (NC weights) | no |
| **π³ / Pi3** | 2507.13347 | weak (static) | local+global pointmaps, conf, cam-to-world poses | yes | fast | code BSD-3; weights CC-BY-NC | **weights yes** (3.6G) |
| **MonST3R** | 2410.03825 | **yes** | per-frame world pointmaps + poses + K, dynamic mask | yes (+global align ~1 min/60 fr) | ~1 min/clip | **CC-BY-NC-SA** | repo only, **no weights** |
| **CUT3R** | 2501.12387 | **yes**, online recurrent | world pointmap + pose, no align | yes | 16 fps online | **CC-BY-NC-SA** | repo only, **no weights** |
| **St4RTrack** | 2504.13152 | **yes** | per-pixel 3D world tracks + per-frame geometry + cam (K,R,T) | **yes, no pose needed** | 30 fps (4090) | non-commercial research | **weights yes** (2.2+4.2+2.6 G) |
| **SpatialTrackerV2** | 2507.12462 | **yes** | depth + cam pose+intrinsics + per-pixel 3D tracks + **visibility** + **dynamic prob** | **yes** (also RGBD mode) | 10-20 s/clip | **CC-BY-SA-4.0** | no (clone it) |
| **Track4World** | 2603.02573 | **yes** | **dense** per-pixel 3D world tracks + geometry + cam, one pass | **yes** | feed-forward | Tencent + CC-BY-4.0 | no (clone it) |

**Key facts that decide it:**
- Plain **VGGT/Pi3/DUSt3R/MASt3R are static-scene models → they ghost on manipulation motion even in 2-3-frame
  windows** (agent.md:144, "DUSt3R/MASt3R/VGGT/Pi3 assume static → break on manipulation motion (ghosting)").
  Confirmed by the literature: feed-forward static models "still struggle to capture dynamic motion"
  (VGGT-Ω / DynamicVGGT motivation). So a static geometry model alone is the wrong substrate for the *moving*
  part of the scene — though it is fine for the *static background* and for frame-0.
- The dynamic-capable geometry models (**VGGT-Ω, MonST3R, CUT3R**) give clean per-frame pointmaps + poses + K,
  but they do **not** give per-pixel *correspondence over time* (you'd still have to track). So for OUR task
  (we need per-Gaussian *motion*), the **unified trackers (STv2, St4RTrack, Track4World) dominate**: they output
  camera + geometry **and** the 3D trajectory in one consistent model, so the geometry and the motion live in the
  same gauge — exactly what we need to drive Gaussians.

**Verdict #1:** use a **unified RGB→{camera, geometry, 3D-motion} tracker** as the spine (STv2 top, Track4World
upgrade, St4RTrack on-disk fallback). Keep a dynamic geometry model (VGGT-Ω) only as a *second opinion* on the
static-scene depth/pose if the tracker's depth is too soft for a clean frame-0 3DGS.

---

## 3. Research question #2 — monocular dynamic 3D motion / tracking through occlusion

This is the half that killed §38. Ranked for the manipulation failure mode (small objects + heavy hand-object
occlusion):

**(1) SpatialTrackerV2 — TOP.** (arXiv 2507.12462; agent.md:525 "the §39 pick".)
- **Input:** RGB video + N query points `Q_i=(x_i,y_i)`; you choose the query points, so you can query a **dense
  grid OR exactly our per-Gaussian frame-0 `uv`** (it tracks "arbitrary user-specified query points", paper §3).
  Also has an RGBD mode (`--data_type RGBD`) if we ever have depth.
- **Output:** 2D tracks `T^2d∈R^{T×N×2}`, **3D tracks `T^3d∈R^{T×N×3}` in camera coords**, **visibility
  `p_vis` per track**, **dynamic probability `p_dyn` per track**, camera poses `P` (bundle-adjusted), video depth
  `D_norm`, depth scale/shift `(a,b)`. Front-end = VGGT-style temporal alternating-attention depth+pose
  initializer with `Ptok/Stok` tokens; back-end = Joint Motion Optimization (SyncFormer) iteratively refining
  2D+3D tracks + visibility + dynamics; dynamic points filtered from BA via `p_dyn`.
- **Occlusion:** **TAPVid-3D Occlusion-Accuracy = 90.6** (vs v1's 83.0); AJ 21.2 vs v1 10.0 (+112%); APD3D 31.0
  vs v1 16.8 (+84.5%); beats DELTA by 61.8%/50.5% on AJ/APD3D. "outperforms existing 3D tracking methods by 30%,
  matches dynamic-recon accuracy at 50× the speed." This `p_vis` is the lever §38 lacked.
- **License** CC-BY-SA-4.0; **code** `github.com/henry123-boy/SpaTrackerV2`; **weights** HF `Yuxihenry`
  (Google-Drive mirror `drive.google.com/.../1GYeC639gA23N_OiytGHXTUCSYrbM0pOo`); **10-20 s/sequence**; Python
  3.11 / torch 2.4.1 — our venv is torch 2.8.0+cu126 (agent.md:112), so build it in an isolated env or pin a
  compatible torch (see §5 risk).

**(2) Track4World — TOP UPGRADE (newest, dense-native).** (arXiv 2603.02573, Mar 2026.)
- **Input:** pure monocular RGB only (no pose/depth/intrinsics). **Output:** **dense per-pixel 3D world-centric
  trajectories of EVERY pixel** + reconstructed geometry + camera, in ONE feed-forward pass; explicitly targets
  "rapid motion, occlusions, and deformations." Built on a **VGGT-style ViT** + a new **3D correlation** scheme
  for 2D+3D dense flow between arbitrary frame pairs.
- It directly fixes the stated gap: "monocular 3D tracking works are limited to either tracking **sparse points on
  the first frame** or a **slow optimization-based** dense framework" — Track4World does dense, feed-forward,
  first-frame→all-frames. That is precisely the per-Gaussian-motion shape we want (frame-0 Gaussians → their 3D
  paths).
- **Backbone variants on HF `TencentARC/Track4World`:** `track4world_da3.pth` (DepthAnythingV3, supports
  `--metric_scale` → **metric meters**), **`track4world_pi3.pth` (Pi3 backbone — reuses our on-disk Pi3)**,
  `track4world_moge.pth` (MoGe). License = Tencent license + CC-BY-4.0. Output appears PLY + (per the demo)
  world-coordinate tracks; coordinate frame metric (DA3) or relative (Pi3/MoGe). CUDA 12.1 / Py 3.11 / torch
  2.5.1.
- **Why it may beat STv2 for us:** dense-by-construction (no "query each Gaussian" loop), newest accuracy, and a
  Pi3 variant we can warm-start from. **Risk:** brand-new (Mar 2026) → install maturity unknown; validate on
  LIBERO before trusting.

**(3) St4RTrack — on-disk fallback (zero download).** (arXiv 2504.13152.)
- Pure RGB, internally estimates K/R/T (square pixels, centered PP, static focal). Dual-branch: reconstruction
  branch = geometry of frame j in frame-i world; tracking branch = where frame-i points move at j → **per-pixel
  3D world trajectory `X^i_j`**. Pure feed-forward (optional 5-min TTA on 4×A100). 30 fps on 4090.
- 3D-tracking APD3D (Point Odyssey, world frame): **St4RTrack 67.95 vs SpatialTracker-v1 38.54 vs MonST3R 33.47**;
  dynamic points 68.72 vs 51.20. Recon EPE 0.2406 vs MonST3R 0.3044.
- **Caveat:** occlusion handling is *implicit* (the alignment loss only scores *visible* correspondences;
  TTA uses CoTracker3 pseudo-labels) — it does **not** emit an explicit per-track visibility score like STv2's
  `p_vis`. So it is a weaker occlusion story than STv2/Track4World, but it is already on the box with weights and
  is a fast first integration. Non-commercial research license.

**(4) Others (context, not chosen):** **DELTA** (dense efficient 2D→3D tracking — STv2 beats it),
**TAPIP3D / DOT / SpatialTracker-v1** (2D-centric or older), **Shape-of-Motion** (arXiv 2407.13764 —
*optimization* per scene: fuses mono-depth + 2D tracks into a global 4DGS with motion bases; high quality but
**minutes-to-hours per clip**, not feed-forward → only as an offline GT-quality oracle), **DynOMo / POGS /
GraspSplats / DynaMem** (online or manipulation-specific GS tracking, optimization-based). **Stereo4D** needs
stereo. **Use the feed-forward unified trackers; keep Shape-of-Motion as an optional slow "gold GT" cross-check.**

---

## 4. Research question #3 — unified "video → 4D Gaussians" in one model

| Model | arXiv | What it is | Posed input? | Use for us |
|---|---|---|---|---|
| **DGS-LRM** | 2506.09997 | feed-forward **posed** monocular video → per-pixel deformable 3DGS + 3D scene flow, 0.495 s/24-fr A100 | **YES (Plücker rays, known cam)** | **downstream booster** after we have cam from a tracker; NOT a from-RGB entry; weights not released |
| **L4GM** | 2406.10324 | video → per-frame 3DGS, temporal attn (NVIDIA) | object-centric, needs clean fg | not for cluttered manipulation |
| **4DGT** | 2506.xxxxx | 4D Gaussian transformer from real monocular video | semi-posed | research-grade; watch |
| **Stereo4D** | 2412.xxxxx | learns 3D motion from internet **stereo** | stereo | N/A (mono only) |
| **Track4World** | 2603.02573 | dense 3D tracks + geometry + cam (see §3) | **no** | **see §3 — this is the closest single-model RGB→(geometry+motion)** |

**Verdict #3:** there is **no single open model that goes raw-RGB → a *trained* dynamic-3DGS asset** that fits
manipulation clutter today. **DGS-LRM** is the closest in spirit (per-pixel deformable Gaussians + scene flow,
real-time) but **requires camera poses as input** and has no released weights — so it is a *second-stage* quality
upgrade, fed by camera poses from STv2/Track4World, **not** the data-gen entry point. The pragmatic "video→4D
Gaussians" for us = **tracker (STv2/Track4World) gives geometry+camera+motion → we assemble the 3DGS + per-Gaussian
trajectory ourselves** (we already own all that assembly code: `to_gaussians.py`, `maniskill_gt.py` schema). This
is more robust than betting on an immature end-to-end 4DGS model and keeps our schema/trainer untouched.

---

## 5. Research question #4 — the CONCRETE pipeline for our case (build this)

**Inputs we have per LIBERO/real clip:** an RGB video (agentview), the language instruction, K+1 sampled frames.
**Outputs we must emit:** the exact `maniskill_gt.py` clip dict (file:line `code/scripts/maniskill_gt.py:700-711`):
`means, quats, scales, opacities, colors` (the frame-0 GaussianSet g0), `uv` (per-Gaussian frame-0 pixel),
`seg_per_g` (per-Gaussian entity/mover id), `traj [Kf+1, N, 3]` (per-Gaussian world position each frame),
`K_intr, viewmat, H, W, Kf, instruction, gt_rgb [Kf+1,H,W,3]`, plus `split`. The trainer
(`train_sim.py`) + `SimClipDataset` consume exactly these (agent.md §39-40).

### Pipeline P1 (TOP — SpatialTrackerV2 spine)

**Step A — run the tracker once on the RGB clip.**
`SpatialTrackerV2.inference(frames, query_points=Q)` where `Q` = **the frame-0 pixel grid at the resolution we
will turn into Gaussians** (e.g. every pixel of the 256² LIBERO agentview, or a stride-1/2 grid). STv2 returns,
per query point n and frame t: `T^3d[t,n]` (3D in camera coords), `T^2d[t,n]`, `p_vis[t,n]`, `p_dyn[n]`, plus
per-frame camera `P_t` (R,T), intrinsics (focal), video depth `D_t`, depth scale/shift `(a,b)`.
- Put everything in the **frame-0 camera world** (gauge = first camera), matching our convention
  (`viewmat`/`K_intr` are the frame-0 camera; agent.md:120 "World = first camera").

**Step B — frame-0 3DGS (`g0`, `uv`, `K_intr`, `viewmat`).**
Unproject STv2's **frame-0 depth `D_0`** with STv2's **frame-0 intrinsics** → one Gaussian per (kept) pixel:
`means = K^{-1}·[u,v,1]·D_0`, `colors = RGB_0` (→ SH-DC), `opacities = inv_sigmoid(0.1)`, `scales` from kNN
spacing, `quats = identity`. This is **byte-for-byte what `igsw/lifting/to_gaussians.py` already does** — we just
swap Pi3's depth for STv2's (consistent, occlusion-aware) depth. `uv` = the source pixel of each Gaussian
(already produced). `K_intr`/`viewmat` = STv2's frame-0 camera. **Optionally fuse multi-frame depth into a more
complete g0** exactly as our validated `maniskill_gt._fuse_canonical_gaussians` (§42, +2.6 dB dynamic-region) —
register each frame's depth back to frame-0 via the per-Gaussian track (we now have the track, so the static
"camera static" assumption isn't needed; use the 3D track itself).

**Step C — per-Gaussian trajectory `traj` (the §38-killer step, now clean).**
For each Gaussian g with frame-0 pixel `uv_g`, its motion **IS** the STv2 3D track of query point `uv_g`:
`traj[t, g] = world(T^3d[t, n(g)])`. Because we queried STv2 at exactly our Gaussian pixels, there is **no
separate 2D→3D lift** — the lift, the depth, and the temporal correspondence all come from the one consistent
model. **Occlusion handling (the fix):** use `p_vis[t,g]`:
- where `p_vis` is high → trust the 3D track;
- where `p_vis` is low (occluded) → **do not feed the raw (noisy) 3D point**. Instead, since manipulation movers
  are near-rigid, **fit a per-entity rigid transform `T_{e,t}` from the *visible* points of that entity** (Kabsch/
  Umeyama on the visible-Gaussian subset) and propagate it to the occluded Gaussians of the same entity:
  `traj[t, g_occluded] = T_{e,t} · means_g`. This is the SIM trick (`X_t = T_{e,t} T_{e,0}^{-1} X_0`,
  agent.md:526/558) recovered from real video — exactly the "handle occluded movers" requirement. Static
  Gaussians (p_dyn low) get `traj[t,g] = means_g` (no motion).

**Step D — `seg_per_g` (mover/static + object id), for the gate + sem→gate + magnitude loss (§44k).**
1. **Mover/static:** threshold STv2 `p_dyn` (and/or 3D-displacement of the track) → per-Gaussian mover label.
   This is the GT for `mover_bce_loss` (the gate) — replaces the sim's exact mover label. (Robustify by
   clustering moving Gaussians in 3D so a coherent object, not speckle, is labelled mover — mirrors §38's
   "GT top-movers ARE a coherent object" check.)
2. **Object identity / instruction target:** run **Grounded-SAM2** (or SAM2 + the instruction noun, or the
   VLM/Qwen3-VL referring-mask — agent.md:575, "Qwen referring points/boxes = cleanest no-GT which-pixels=object")
   on frame-0 → a 2D mask per object; **project masks to Gaussians via `uv`** → `seg_per_g` entity ids and an
   `object_of_interest` flag for the instruction's target. (LIBERO even ships `object_of_interest_mask` GT — use
   it directly there to *validate* this 2D→Gaussian projection, §45/§6.)
   - This gives the **occlusion-robust 3D identity** the user's §44e hypothesis needs (sem→gate): the object id
     is assigned in 3D at frame-0 and travels with the Gaussian, so even when the arm occludes it in 2D later,
     the Gaussian still "knows" it is the cube/butter. Distill it into `sem_proto` exactly as dyngate5+.

**Step E — `gt_rgb`, instruction, split, validate, save.** Stack the K+1 RGB frames; copy the instruction;
assign train/held split. **GT-correctness safeguard (mandatory, copy from sim):** render the analytic-moved g0
(`apply_traj_to_gaussians`, `maniskill_gt.py:520`) against the REAL future RGB frames; require **frame-0 PSNR ≥
18** and **movefrac ≥ 0.02** (the sim gate, agent.md:536/§45 acceptance) — drop clips that fail. Save the dict
with `torch.save` in the maniskill format → `SimClipDataset` reads it unchanged.

**Net:** the ONLY new code is a `video_gt.py` that wraps STv2 and re-uses `to_gaussians.py` +
`_fuse_canonical_gaussians` + `apply_traj_to_gaussians` + the validate() from `maniskill_gt.py`. Trainer, model,
losses, eval = untouched.

### Pipeline P2 (UPGRADE — Track4World spine)
Same A-E, but Step C is free: Track4World already returns **dense per-pixel 3D world tracks**, so `traj` = the
dense track sampled at `uv` (no query loop), and Step B's geometry comes from the same pass. Use
`track4world_pi3.pth` to reuse on-disk Pi3, or `track4world_da3.pth --metric_scale` for **metric** coordinates
(removes the scale-ambiguity risk, §7). Prefer P2 if it installs and passes the LIBERO validation (§6) at least
as well as P1.

### Pipeline P3 (de-risk NOW — St4RTrack, on disk)
Same A-E, tracker = St4RTrack (weights already at `third_party/St4RTrack/checkpoints/`). St4RTrack gives per-pixel
3D world tracks + geometry + camera with **no download**, so we can stand up `video_gt.py` and the LIBERO
validation *today*, then swap in STv2/Track4World once installed. Weaker occlusion story (no explicit `p_vis`) →
rely more on the per-entity rigid-fit in Step C; treat P3 as the scaffold, P1/P2 as the production tracker.

### Optional Stage-2 quality booster (later): DGS-LRM
Once a tracker gives us posed video (camera per frame), feed (RGB + camera as Plücker rays) to **DGS-LRM**
(2506.09997) for per-pixel deformable Gaussians + dense scene flow at 0.5 s/clip → an even cleaner `traj` and a
photoreal g0. Gated on DGS-LRM weights being released. Not needed for v1.

---

## 6. Research question #6 — LIBERO-GT validation (quantify "is pure-video clean enough to train on")

We have **LIBERO `binhng/libero_object_lerobot_mask_depth`** WITH GT depth (8-bit normalized agentview),
GT seg `image_mask`, **GT `object_of_interest_mask`**, and (via robosuite) GT camera intrinsics(fovy)+extrinsic +
depth near/far (§45). This lets us measure the pure-video pipeline's error **against GT before any big run**.
Run STv2/Track4World/St4RTrack on the **RGB-only** LIBERO video, then compare to the GT it never saw:

**(a) Estimated-camera error.** Sim(3)-Umeyama-align the estimated camera trajectory to the GT extrinsics, then
report **ATE** (RMSE of aligned positions) and **RPE_T / RPE_R** (relative translation/rotation per pair). Report
estimated focal vs GT fovy-derived focal (%, after the same scale). [Standard protocol: Sim(3) Umeyama align →
ATE/RPE.] **Pass bar:** ATE small relative to scene radius; RPE_R within a few degrees.

**(b) Estimated-depth error (scale-aligned).** Affine-invariant: fit one global (scale, shift) by least-squares
between estimated depth and GT depth **shared across all frames of a clip**, then report **AbsRel**
= mean(|d̂−d|/d) and **δ<1.25 / 1.25² / 1.25³**. (LIBERO depth is 8-bit normalized → first map it to metric via
the robosuite near/far before comparing; §45.) **Pass bar:** AbsRel small enough that the unprojected g0 renders
≥18 PSNR against frame-0 RGB (our existing safeguard).

**(c) Motion / track error vs GT-mask object motion (THE decisive metric).** The GT motion of the instruction's
object = rigid transform of the `object_of_interest_mask` depth point cloud frame-0→t (centroid translation +
Procrustes/PCA rotation — the §45 "per-object rigid registration" we already planned to compute GT). Compare the
pipeline's per-Gaussian `traj` (restricted to that object's Gaussians) to this GT:
- **3D end-point error (EPE)** and **APD3D** (% within thresholds) of the object's points, frame-by-frame.
- **Occluded-segment EPE** specifically on frames where the GT mask says the object is ≥X% occluded by the arm —
  this directly measures the §38 failure mode. STv2's `p_vis` should make this finite (vs §38's teleports).
- **Crucially, re-use our own model metrics** on the pure-video clip as the GT: run
  `eval_sim_generalization.py`-style **corr(GT_disp, PRED_traj_disp)**, **top-mover ratio**, **static-leakage**
  (predicted motion on GT-static Gaussians), and **mover precision/recall** of the `p_dyn`-derived label vs the
  GT mask. These are the *same numbers* dyngate7 hit on sim (corr 0.85 / ratio 0.81 / leak 0.01, §44k) → a direct
  apples-to-apples "is the pure-video GT as learnable as the sim GT?" read.

**Decision rule (go/no-go for a big pure-video run):** if, on held-out LIBERO clips, the pure-video pipeline's GT
gives (i) g0 render ≥18 PSNR, (ii) object-motion APD3D high + occluded-segment EPE bounded (no teleports), and
(iii) the model trained on it reaches corr/ratio/leak within ~20% of the sim-GT numbers — **it is clean enough →
scale**. Else, iterate the occlusion-handling (per-entity rigid-fit, `p_vis` threshold) or fall back to
re-rendering LIBERO in-sim for exact depth+pose (§45 fallback) for the geometry while keeping the *video* tracker
only for the motion.

---

## 7. Known risks + mitigations

1. **Occlusion (the §38 killer).** *Mitigation:* pick a tracker with explicit per-track **visibility** (STv2
   `p_vis`, OA=90.6; Track4World occlusion-targeted) + the **per-entity rigid-fit-from-visible-points** in Step C
   so occluded Gaussians ride their object's rigid transform instead of carrying raw track noise. Validate with
   the occluded-segment EPE in §6(c). This is the explicit fix for "mover-visibility 0.40-0.66, jumped tracks."
2. **Scale ambiguity / non-metric gauge.** STv2/St4RTrack/Pi3 are up-to-scale; VGGT/Pi3 "NOT metric"
   (agent.md:120-121). *Mitigation:* the model trains on **relative** displacements and a per-clip gauge anyway
   (sim was already scale-normalized per clip), so consistency within a clip is what matters — and the unified
   tracker guarantees geometry+motion share one scale. For absolute metric, use **Track4World `da3` +
   `--metric_scale`** (metric meters) or LIBERO's known near/far. Always normalize per-clip by scene radius (as
   sim does) before the loss.
3. **Dynamic-vs-static camera.** LIBERO agentview is *static*; AgiBot/real may move. *Mitigation:* the trackers
   estimate ego-motion regardless; with a static camera, `p_dyn`/scene-flow cleanly separates movers from the
   static table (no ego-motion to confound). With a moving camera, STv2/Track4World are trained for it; just keep
   the world gauge = frame-0 camera. Note DGS-LRM *requires* a non-stationary camera (can't do teleporting/static
   discrete poses) — another reason it's a Stage-2 booster, not the entry.
4. **Tracker geometry too soft for a crisp frame-0 3DGS.** Mono depth → soft Gaussians (our §39 finding: naive
   lift ~10-17 PSNR). *Mitigation:* (a) multi-frame **fusion** of the tracker depth (§42, +2.6 dB) into g0; (b)
   optional brief per-clip gsplat optimization of g0 for render quality (we already do this, §7/§8 of agent.md);
   (c) cross-check depth with VGGT-Ω if needed.
5. **Install / env friction.** STv2 pins torch 2.4.1, Track4World torch 2.5.1, our venv is torch 2.8.0+cu126.
   *Mitigation:* build each tracker in its **own uv venv** and run it as an **offline data-gen step** (it never
   shares a process with the trainer — exactly like `cache_clips.py` caches Pi3+Qwen outputs offline, agent.md
   §10). The trainer only ever reads the saved clip dicts. So torch-version conflicts are a non-issue.
6. **License.** STv2 CC-BY-SA-4.0 and Track4World CC-BY-4.0 are **commercial-OK with attribution**
   (ShareAlike for STv2); St4RTrack/MonST3R/CUT3R/Pi3-weights/VGGT-1B/co-tracker are **non-commercial** — fine
   for research now (agent.md:123 "for research all fine; productization later"); for any product, prefer
   STv2/Track4World (or VGGT-1B-Commercial) and avoid the NC weights.
7. **Track4World maturity.** Brand-new (Mar 2026). *Mitigation:* gate adoption on the §6 LIBERO validation;
   keep STv2 as the proven default and St4RTrack (on disk) as the instant fallback.

---

## 8. Concrete build order (what we do next)

1. **Clone + env (offline data-gen tools, isolated venvs).**
   - `third_party/SpaTrackerV2` (henry123-boy) + weights (HF `Yuxihenry` / Google-Drive). Its own uv venv (torch
     2.4.1). [P1]
   - `third_party/Track4World` (TencentARC) + HF `TencentARC/Track4World` weights (`track4world_pi3.pth` reuses
     our Pi3). Own venv (torch 2.5.1). [P2]
   - St4RTrack is already cloned **with weights** → use immediately for the scaffold. [P3]
2. **Write `code/scripts/video_gt.py`** = `maniskill_gt.py`'s twin, but: tracker(RGB)→{depth,cam,3D-tracks,
   p_vis,p_dyn}; reuse `to_gaussians.py` (g0), `_fuse_canonical_gaussians` (complete g0), the Step-C
   visible-rigid-fit for `traj`, Grounded-SAM2/`object_of_interest` for `seg_per_g`, and `validate()` (≥18 PSNR /
   movefrac ≥0.02). Emit the exact clip dict (`maniskill_gt.py:700-711`).
3. **LIBERO validation harness** (§6): run the tracker RGB-only on LIBERO, compute (a) ATE/RPE vs GT extrinsics,
   (b) scale-aligned AbsRel/δ vs GT depth, (c) object-motion APD3D + occluded-segment EPE vs the
   `object_of_interest_mask` rigid GT, and the corr/ratio/leak/mover-P-R model metrics. Produce the one-line
   go/no-go vs the sim-GT numbers (dyngate7: corr 0.85 / ratio 0.81 / leak 0.01).
4. **If go:** scale `video_gt.py` over LIBERO (sharded across the 4 GPUs, like `gen_sim_dataset.py`) →
   `data/libero_video/`; warm-start the dyngate7 recipe (fusion + gate + sem→gate + magnitude loss); eval
   held-out with the same metrics + export pred-vs-GT 3DGS .ply for the user's 3D review.
5. **Later boosters:** DGS-LRM (needs released weights) on the now-posed video for a cleaner `traj`/g0; Shape-of-
   Motion as a slow gold-GT cross-check on a few clips.

---

## 9. Source map (arXiv + our clones)

**arXiv (web, 2024-2026 SOTA):**
- VGGT 2503.11651 · VGGT-Ω 2605.15195 · π³/Pi3 2507.13347 · MonST3R 2410.03825 · CUT3R 2501.12387
- St4RTrack 2504.13152 · **SpatialTrackerV2 2507.12462** · **Track4World 2603.02573** · DELTA, TAPIP3D (refs)
- DGS-LRM 2506.09997 · L4GM 2406.10324 · 4DGT · Stereo4D · Shape-of-Motion 2407.13764 · DynOMo 2409.02104
- Robo3R 2602.10101 (multi-view RGB, manipulation feed-forward recon — NOT monocular, context only)
- Eval protocols: scale-shift LS depth align + AbsRel/δ; Sim(3)-Umeyama align + ATE/RPE (depth-eval &
  pose-eval literature).
- Code/weights: STv2 `github.com/henry123-boy/SpaTrackerV2` (CC-BY-SA-4.0, HF `Yuxihenry`); Track4World
  `github.com/TencentARC/Track4World` (Tencent+CC-BY-4.0, HF `TencentARC/Track4World`: da3/pi3/moge `.pth`).

**Our server clones (`/mnt/pfs/public/xuhaoming/instruct_gs_world/`, via `ssh -p 8600 root@106.13.104.32`):**
- `third_party/St4RTrack/checkpoints/model.safetensors` (2.2G), `St4RTrack_wo_reweight/St4RTrack_release_version.pth`
  (4.2G), `MASt3R_base.pth` (2.6G) — **weights present, runnable now**; license: README:228 non-commercial.
- `checkpoints/Pi3/model.safetensors` (3.6G) — Pi3 weights present (reusable by `track4world_pi3.pth`).
- `third_party/{vggt,monst3r,CUT3R,co-tracker,Pi3}` — repos present; **VGGT/MonST3R/CUT3R have NO weights**
  (would download; all NC). Licenses: monst3r/CUT3R/co-tracker = CC-BY-NC(-SA); VGGT = VGGT license; Pi3 = BSD-3.
- `third_party/SpaTrackerV2`, `third_party/Track4World` — **NOT yet cloned** (step 1).
- Our schema/assembly to reuse: `code/scripts/maniskill_gt.py:700-711` (clip dict), `:520` (`apply_traj_to_gaussians`),
  `_fuse_canonical_gaussians` (§42), `code/igsw/lifting/to_gaussians.py` (depth→Gaussians),
  `code/igsw/data/sim_clips.py` (consumes the dict), `code/scripts/eval_sim_generalization.py:106-124`
  (corr/ratio/leak/mover-P-R metrics), `code/scripts/inspect_libero.py` (LIBERO fields incl.
  `object_of_interest_mask`).
</content>
</invoke>
