# Fixed-Identity DynamicGS Data Plan

## Why change the data contract

The current real-video route estimates a 3D point/Gaussian set per window and then
links motion with CoTracker plus lifted point maps. That creates a noisy target:
occlusion can break tracks, monocular/depth lifting can jump, and the model is asked
to learn from correspondence errors rather than from scene dynamics.

The target data contract should match Dynamic 3DGS style reconstruction:

1. Build one canonical Gaussian scene for the starting state.
2. Keep the Gaussian identities fixed over the whole clip.
3. Store a time function for each Gaussian: position, rotation, and optionally scale/opacity.
4. Keep appearance mostly canonical for rigid manipulation data unless the object truly changes.
5. Train the world model to predict this motion function, not to rediscover frame-to-frame
   correspondences.

This matches the project goal better than per-frame reconstruction: the learner sees a
stable object identity and learns how instruction-conditioned dynamics moves that identity.

## Evidence from the literature

- Dynamic 3D Gaussians models a dynamic world as persistent 3D Gaussians that move and
  rotate over time, enabling dense 6-DoF tracking and novel-view synthesis.
- Deformable 3D Gaussians learns Gaussians in canonical space plus a deformation field for
  monocular dynamic scenes.
- 4D Gaussian Splatting and Spacetime Gaussian Feature Splatting both treat time as part of
  the Gaussian representation rather than rebuilding unrelated 3DGS frames.
- 4D LangSplat shows that dynamic semantic fields should attach language/semantic state to
  the 4D scene, not only to a static frame.

Primary source pointers:

- Dynamic 3D Gaussians: https://github.com/JonathonLuiten/Dynamic3DGaussians
- Deformable 3D Gaussians: https://arxiv.org/abs/2309.13101
- 4D Gaussian Splatting: https://github.com/hustvl/4DGaussians
- Spacetime Gaussian Feature Splatting: https://arxiv.org/abs/2312.16812
- 4D LangSplat: https://4d-langsplat.github.io/

## Concrete data stages

### Stage A: clean sim fixed-ID data

This is already the right first gate. For each ManiSkill episode:

- Build `g0` once from the start frame RGB-D.
- Save `seg_per_g` so each Gaussian belongs to a simulator entity.
- Save per-frame entity poses.
- Generate `traj[t, i] = T_entity(t) @ inv(T_entity(0)) @ g0.means[i]`.
- Validate by rendering the analytically moved Gaussians against the simulator RGB.

This produces learnable motion targets without tracking or per-frame matching.

### Stage B: multi-view canonical scan

Replace the single frame-0 RGB-D build with a short pre-action scan:

- Record multiple calibrated RGB-D/seg views of the scene before manipulation.
- Fuse all views into one canonical Gaussian set in world coordinates.
- Deduplicate/fuse overlapping Gaussians by voxel/entity key.
- Keep `uv` for the policy camera so Qwen spatial grounding can still sample visual features.
- Keep `seg_per_g` and entity poses as in Stage A.

Only `g0` construction changes. The trainer should continue consuming the same fixed-ID
`g0 + traj` contract.

### Stage C: real data version

For real robot data, the analogous route is:

- Before manipulation, scan the scene with calibrated RGB-D or multi-view RGB plus a
  dynamic/static reconstruction method.
- Optimize one canonical 3DGS for the scene/object state.
- Use object-level 6-DoF pose tracks or an occlusion-aware 3D tracker to produce a motion
  function for canonical Gaussians.
- Avoid generating separate per-frame Gaussian sets unless they are only used as render
  supervision, never as the identity source.

## Trainer implications

- Continue direct trajectory loss on fixed Gaussian identities.
- Add stronger rigid-appearance anchors for sim and rigid-object real data:
  color, opacity, and scale should remain close to `g0` unless explicitly supervised otherwise.
- Evaluate with both motion metrics and clean renders:
  correlation/top-mover ratio for motion, frozen-appearance render for motion-only inspection,
  and full render to catch appearance drift.

## Immediate repo tasks

1. Keep the 4s random-start sim training running as the current fixed-ID baseline.
2. Save entity pose metadata in generated sim clips so render/eval can use rigid rotation,
   not only translated means.
3. Add an appearance-anchor training option before the next long sim run.
4. Prototype multi-view canonical scan in sim as a drop-in replacement for single-view `g0`.
5. Compare single-view vs multi-view canonical data with the same trainer/eval contract.
