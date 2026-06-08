"""SC-GS rollout: control-set dynamics + dense LBS deformation.

The dynamics runs on M sparse CONTROL gaussians (a subset of the dense set, so
control motion is LBS-consistent). Each step a `delta_fn(state, step)` predicts
control deltas, which (a) advance the control state on-manifold and (b) deform the
full dense set via LBS. The delta_fn is a closure supplied by the caller (it holds
the per-layer Qwen3-VL context), keeping this module conditioning-agnostic.
"""

from __future__ import annotations

import torch

from ..gaussians.types import GaussianSet
from ..gaussians.deform import build_lbs_binding, lbs_step
from .model import GaussianState
from .manifold import apply_deltas_tensors


class SCGSRollout:
    def __init__(
        self,
        dense_g0: GaussianSet,
        n_control: int = 2048,
        k: int = 4,
        sigma_scale: float = 2.0,
        generator: torch.Generator | None = None,
        ctrl_idx: torch.Tensor | None = None,
    ):
        self.dense0 = dense_g0
        n = len(dense_g0)
        if ctrl_idx is not None:
            ctrl_idx = ctrl_idx.to(dense_g0.device)
        elif n_control >= n:
            ctrl_idx = torch.arange(n, device=dense_g0.device)
        else:
            ctrl_idx = torch.randperm(n, device=dense_g0.device, generator=generator)[:n_control]
        self.ctrl_idx = ctrl_idx
        ctrl_feats = dense_g0.features[ctrl_idx] if dense_g0.features is not None else None
        self.control0 = GaussianSet(
            dense_g0.means[ctrl_idx], dense_g0.quats[ctrl_idx], dense_g0.scales[ctrl_idx],
            dense_g0.opacities[ctrl_idx], dense_g0.colors[ctrl_idx], ctrl_feats,
        )
        self.knn_idx, self.knn_w = build_lbs_binding(
            dense_g0.means, self.control0.means, k=k, sigma_scale=sigma_scale
        )

    @property
    def n_control(self) -> int:
        return len(self.control0)

    def rollout(self, delta_fn, K: int, start_step: int = 0, collect: bool = True):
        """delta_fn(control_state: GaussianState, step_idx: LongTensor[B]) ->
        (v, om, dls, dlo, dc, dfeat) each [B,M,*]. Returns
        (dense_states[list GaussianSet], deltas[list (v,om,dls)], ctrl_traj[list mean])."""
        cs = GaussianState.from_gaussianset(self.control0)   # [1,M,*]
        dense = self.dense0
        dense_states, deltas, ctrl_traj = [], [], []
        for step in range(K):
            step_idx = torch.full((1,), start_step + step, dtype=torch.long, device=dense.means.device)
            logit_o = torch.logit(cs.opacities.clamp(1e-6, 1 - 1e-6))
            v, om, dls, dlo, dc, _ = delta_fn(cs, step_idx)
            dense = lbs_step(
                dense, cs.means[0], v[0], om[0], dls[0], dlo[0], dc[0], self.knn_idx, self.knn_w
            )
            nm, nq, ns, no, nc, _ = apply_deltas_tensors(
                cs.means, cs.quats, cs.scales, logit_o, cs.colors, None, v, om, dls, dlo, dc
            )
            cs = GaussianState(nm, nq, ns, no, nc, cs.features)   # carry static relevance feature
            dense_states.append(dense)
            if collect:
                deltas.append((v, om, dls))
                ctrl_traj.append(cs.means[0])
        return dense_states, deltas, ctrl_traj
