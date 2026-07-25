# Stage-B Fixed-Identity DynamicGS Pipeline Report

## Current contract observed

- `maniskill_gt.py` builds a single-camera fixed-ID clip: `means/quats/scales/opacities/colors`,
  `uv`, `seg_per_g`, `traj[K+1,N,3]`, one `K_intr/viewmat/H/W`, `gt_rgb`, and `instruction`.
- `gen_multiview_dynamic_gs_probe.py` already fuses four canonical RGB-D/seg scan cameras, but the
  compatibility path projects every Gaussian into `scan_0` and can optionally filter to
  `policy_visible_only`.
- `model_full._control_visual()` samples one Qwen image grid from one `vlm_inputs` image using
  `control_uv[M,2]` and one `control_uv_hw`.
- `train_sim.py` and `eval_sim_generalization.py` assume each selected control has a valid uv in
  the same single `gt_rgb[0]` image passed to Qwen.

Therefore complete multiview canonical Gaussians are not trainer-compatible yet: hidden or
out-of-frame Gaussians have no valid single policy-camera Qwen feature.

## Proposed Stage-B data contract

Keep the legacy fields for rendering/eval, but treat the following as the authoritative Stage-B
grounding metadata:

- `contract_version = "dynamic_gs_stage_b_v1"`.
- `means/quats/scales/opacities/colors`: one deduped canonical multiview Gaussian scene.
- `seg_per_g[N]`: simulator entity id per Gaussian.
- `traj[K+1,N,3]`: fixed-identity analytic trajectory from simulator entity poses.
- `source_view[N]`, `source_uv[N,2]`: original scan camera and pixel used as the representative
  observation after voxel/entity dedupe.
- `uv_by_view[N,V,2]`: projection of every Gaussian into every canonical scan image.
- `uv_valid_by_view[N,V]`: in-frame and depth-consistent visibility mask.
- `best_view[N]`, `best_uv[N,2]`: first valid grounding view; fallback is source view if all masks
  are invalid.
- `scan_rgb[V,H,W,3]`, `scan_K_intr[V,3,3]`, `scan_viewmat[V,4,4]`, `scan_cam_ids[V]`: the
  canonical images and calibration Qwen/multiview rendering need.
- Backward-compatible fields `uv`, `uv_valid`, `K_intr`, `viewmat`, `gt_rgb` remain tied to
  `policy_view=0` only; they must not be interpreted as complete grounding for Stage-B.

## Required model changes

Minimal patch to `model_full.py`:

1. Add `_control_visual_multiview(vlm_inputs_by_view, control_uv_by_view, control_uv_valid_by_view,
   uv_hw, d_device, d_dtype)`.
2. For each view, call the existing `encoder.image_grid_features()` and sample features with the
   existing grid-sample logic.
3. Fuse per-control features with the visibility mask:
   - first implementation: masked mean over valid views;
   - optional stronger implementation: learned view attention from sampled feature plus camera ray.
4. If a control has no valid view, use a learned `null_view_feature` and expose `no_view_mask` for
   logging. Do not silently clamp invalid uv to image borders.
5. Keep output shape identical to `_control_visual`: `vis_tok[M,d]`, `vis_film[M,2d]`,
   `vis_vhead[M,3]`, so `GaussianDynamics` and SC-GS rollout remain unchanged.

This is preferable to filtering because the dynamics still predicts motion for the complete
canonical scene, including back-side and temporarily occluded object Gaussians.

## Required trainer/eval changes

Minimal patch to `train_sim.py` and `eval_sim_generalization.py`:

1. Detect `contract_version == "dynamic_gs_stage_b_v1"`.
2. Build one Qwen input per scan image:
   `vlm_inputs_by_view = [encoder.build_inputs(instruction, scan_rgb[v])]`.
3. After `ctrl_idx` sampling, pass:
   - `control_uv_by_view = clip["uv_by_view"][ctrl_idx]`;
   - `control_uv_valid_by_view = clip["uv_valid_by_view"][ctrl_idx]`;
   - `control_uv_hw = (H, W)`.
4. Keep render supervision on `policy_view` initially so current video/eval outputs remain comparable.
5. Log `valid_views_per_control.mean()`, `zero_valid_control_frac`, and `policy_visible_frac`.
6. Keep `sample_controls()` mover-biased over the full `N`; do not restrict sampling to policy-visible
   points.

## New code added

`code/scripts/gen_multiview_dynamic_gs_dataset.py` generates a Stage-B PickCube clip with complete
multiview canonical Gaussians, voxel/entity dedupe, per-view uv/visibility masks, and the legacy
fields needed for current render validation.

Example:

```bash
python code/scripts/gen_multiview_dynamic_gs_dataset.py \
  --seed 1000 --K 16 --cam 256 --window_steps 80 \
  --out data/multiview_stage_b/pickcube_s1000_train.pt
```

Remote equivalent:

```bash
cd /root/xuhaoming/public/instruct_gs_world
python code/scripts/gen_multiview_dynamic_gs_dataset.py \
  --seed 1000 --K 16 --cam 256 --window_steps 80 \
  --out data/multiview_stage_b/pickcube_s1000_train.pt
```

For CPU-only metadata checks, add `--device cpu --validate_render 0`; this intentionally skips the
gsplat PSNR validation because the renderer path is CUDA-only.

## Blockers before training Stage-B

- Current `model_full._control_visual()` accepts only one image grid and one uv tensor.
- Current trainer/eval pass one `gt_rgb[0]` image to Qwen and ignore `uv_valid`.
- Local tree is missing remote `code/igsw/data/sim_clips.py`; run or patch Stage-B trainer work on
  the remote authoritative tree unless local is synced first.
- The generator currently covers PickCube scan geometry only. Extending to Push/Pull/Stack needs
  scan-env subclasses for those ManiSkill tasks, but the saved contract should stay identical.
