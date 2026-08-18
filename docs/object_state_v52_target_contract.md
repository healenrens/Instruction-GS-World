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

## Supervision boundary

- Deployment Student input: observed RGB history only, encoded by frozen DINO and the causal Student.
- Training-only evidence: frozen point tracks and frozen DINO features.
- Allowed teacher claims: same surface point, observed visibility, measured relative motion, high-confidence pair relations.
- Forbidden teacher claims: fixed object index, semantic instance truth, absence inferred from missing visibility alone.
- Unknown evidence contributes no target gradient.

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
