# v52 Learning-Objective-First Object State Target

## What changed

1. Point tracks no longer define hard pseudo object IDs. They provide only surface-point correspondence, visibility, motion, and confidence-weighted `must-link` / `cannot-link` evidence.
2. The learned target is a persistent, independently changing, compositional visual entity represented by identity, dynamic state, relative geometry, lifecycle, and visual support.
3. The objective must reject known shortcut decompositions before any 22,500-step training can start. Object State must pass held evaluation before Dynamics or latent effect can be added.

## Why

A model cannot learn the intended object state when its loss only rewards agreement with its own assignments or with an unverified trajectory-clustering heuristic.

## Target contract

For object root $k$ at time $t$:

$$
O_t^k=(u_t^k,d_t^k,g_t^k,\ell_t^k,m_t^k),
$$

where $u_t^k$ is persistent identity, $d_t^k$ is changeable dynamic state, $g_t^k$ is relative geometry, $\ell_t^k$ is lifecycle, and $m_t^k$ is visual support.

The training objective combines:

$$
L_{\text{state}}=
L_{\text{track-cycle}}
+L_{\text{relation}}
+L_{\text{owner-evidence}}
+L_{\text{identity}}
+L_{\text{motion}}
+L_{\text{lifecycle}}
+L_{\text{geometry}}
+L_{\text{compositional}}
+L_{\text{minimality}}.
$$

Only identity is temporally invariant. Dynamic state, geometry, visibility, and support are expected to change.

The dynamic target at time $t$ uses only observed trailing motion $t-h\rightarrow t$ for several history horizons $h$. It does not ask the Object State encoder to predict $t\rightarrow t+h$. Future transition prediction remains outside v52.

## Supervision boundary

- Deployment Student input: observed RGB history only, encoded by frozen DINO and the causal Student.
- Training-only evidence: frozen point tracks and frozen DINO features.
- Allowed teacher claims: same surface point, observed visibility, measured relative motion, high-confidence pair relations.
- Forbidden teacher claims: fixed object index, semantic instance truth, absence inferred from missing visibility alone.
- Unknown evidence contributes no target gradient.
- Persistent cycle, pair-relation, and identity-negative losses are gated by object evidence. Static scene tracks cannot be pulled into object roots by a relation loss that conflicts with their scene target.
- Scene tracks use the scene-owner target and low-weight compositional reconstruction. Transient tracks receive an owner target but no identity-permanence target.
- DINO appearance helps form confidence-weighted pair evidence and the auxiliary compositional reconstruction. A root identity is not regressed to a surface-point DINO vector, because different parts of one object need not share the same patch appearance.
- Geometry and trailing motion targets are relation-weighted object aggregates, not individual point coordinates or point flows. An isolated track without reliable object-relation support contributes no object geometry or dynamic target.
- Motion alone does not create an object target. A moving track needs coherent relation support from another track; unsupported or short-lived motion is routed to transient rather than consuming a persistent environment-object root.

## Objective falsification gate

The same objective used for training must assign a lower score to a reasonable decomposition than to every corruption below:

- merge all objects into one root;
- split one object across time;
- swap identity after occlusion;
- assign all foreground to scene;
- lock background into a moving object root;
- allocate one root per track;
- corrupt dynamic motion;
- corrupt lifecycle.

Each corruption must increase its intended loss term, not merely the total by accident.

## Promotion boundary

- v52 trains from scratch for 22,500 optimizer steps.
- v43-v51 model and optimizer checkpoints are diagnostic baselines only.
- DINO and point tracker remain frozen.
- No action-free future prediction, Dynamics, latent effect, language, RGB reconstruction target, or instance segmentation is included.
- Held point-track evaluation is reported as teacher agreement, not independent object truth.
- Dynamics and object-bound latent effect remain prohibited until independent Object State evaluation is available and passes.

## Independent promotion evaluation

The independent promotion path never imports or runs the training point tracker. It reads a small evaluation-only `independent_object_truth_v1` sidecar with stable object IDs, per-frame instance masks, and explicit `unknown/absent/present` lifecycle labels. These labels are not used by training.

The JSON manifest contains `contract`, `complete`, `provenance`, and `items`. `provenance.kind` must be `simulator_ground_truth` or `human_annotation`, and `provenance.uses_training_tracker` must be `false`. Each item declares `split`, `episode_filename`, and a relative `annotation` path. Its PyTorch annotation contains:

- `contract: independent_object_truth_v1`;
- `episode_filename`;
- `frame_indices: [T]` increasing `int64` indices;
- `instance_masks: [T,H,W]`, where `-1` is ignored, `0` is scene/background, and positive values are stable object IDs;
- `object_ids: [O]` listing the evaluated environmental objects;
- `object_presence: [T,O]`, using `-1` for unknown, `0` for absent, and `1` for present.

Ambiguous manipulator pixels, shadows, reflections, and uncertain boundaries should be `-1`, not forced into scene or an environmental object. The independent gate measures foreground routing, background routing, object separation, temporal assignment, identity reappearance, relative center, lifecycle, and deletion locality on these external masks. Passing the track-teacher evaluator alone cannot set `deployment_promotion_ready=true`.
