"""SC-GS-style Linear-Blend-Skinning deformation (Huang et al., CVPR 2024).

The dynamics model predicts per-step deltas on M sparse CONTROL gaussians; those
deltas are propagated to the N dense gaussians via LBS over each dense point's k
nearest control points. This is applied INCREMENTALLY each rollout step (relative
transforms about the control points' current positions), which composes correctly
for autoregressive rollout.

Per dense gaussian i with control neighbours {j} (canonical weights w_ij):
    x_i' = Σ_j w_ij [ R_j (x_i - p_j) + p_j + v_j ]
    q_i' = normalize( blend_j(w_ij, R_j) ⊗ q_i )
    log s_i' = log s_i + Σ_j w_ij δs_j
    logit σ_i' = logit σ_i + Σ_j w_ij δσ_j
    c_i' = clamp( c_i + Σ_j w_ij δc_j , 0, 1 )
where R_j = Exp(ω_j), v_j the control translation. Control points then move
p_j ← p_j + v_j (handled by the caller's manifold update on the control state).
"""

from __future__ import annotations

import torch

from .types import GaussianSet
from ..dynamics.manifold import axis_angle_to_quat, quat_to_rotmat, quat_mul


@torch.no_grad()
def build_lbs_binding(
    dense_means: torch.Tensor,     # [N,3]
    control_means: torch.Tensor,   # [M,3]
    k: int = 4,
    sigma_scale: float = 1.0,
    chunk: int = 20000,
    eps: float = 1e-8,
):
    """Return (knn_idx [N,k] long, knn_w [N,k]) — canonical LBS weights (sum to 1)."""
    n = dense_means.shape[0]
    idx_out = torch.empty(n, k, dtype=torch.long, device=dense_means.device)
    w_out = torch.empty(n, k, dtype=dense_means.dtype, device=dense_means.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(dense_means[s:e], control_means)         # [c,M]
        knn_d, knn_i = torch.topk(d, k, dim=1, largest=False)     # [c,k]
        sigma = knn_d.mean(dim=1, keepdim=True).clamp_min(eps) * sigma_scale
        w = torch.exp(-(knn_d ** 2) / (2 * sigma ** 2))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        idx_out[s:e] = knn_i
        w_out[s:e] = w
    return idx_out, w_out


def _blend_quat(qn: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Weighted quaternion blend with sign alignment. qn [N,k,4], w [N,k] -> [N,4]."""
    ref = qn[:, :1, :]                                            # [N,1,4]
    sign = torch.sign((qn * ref).sum(-1, keepdim=True))          # align hemisphere
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    qa = qn * sign
    q = (w[..., None] * qa).sum(dim=1)                           # [N,4]
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)


def lbs_step(
    dense: GaussianSet,
    control_means: torch.Tensor,   # [M,3] current control positions (BEFORE this step's move)
    v: torch.Tensor,               # [M,3] control translation
    omega: torch.Tensor,           # [M,3] control rotation (so(3))
    dlog_s: torch.Tensor,          # [M,3]
    dlogit_o: torch.Tensor,        # [M,1] or [M]
    dcolor: torch.Tensor,          # [M,3]
    knn_idx: torch.Tensor,         # [N,k]
    knn_w: torch.Tensor,           # [N,k]
    dlog_scale_clamp: float = 3.0,
) -> GaussianSet:
    """Propagate one step of control deltas to the dense set via LBS."""
    R = quat_to_rotmat(axis_angle_to_quat(omega))                # [M,3,3]
    q_ctrl = axis_angle_to_quat(omega)                           # [M,4]
    dlogit_o = dlogit_o.reshape(-1)                              # [M]

    p_n = control_means[knn_idx]                                 # [N,k,3]
    R_n = R[knn_idx]                                             # [N,k,3,3]
    v_n = v[knn_idx]                                             # [N,k,3]
    q_n = q_ctrl[knn_idx]                                        # [N,k,4]
    dls_n = dlog_s[knn_idx]                                      # [N,k,3]
    dlo_n = dlogit_o[knn_idx]                                    # [N,k]
    dc_n = dcolor[knn_idx]                                       # [N,k,3]
    w = knn_w                                                    # [N,k]

    rel = dense.means[:, None, :] - p_n                          # [N,k,3]
    rot_rel = torch.einsum("nkij,nkj->nki", R_n, rel)           # [N,k,3]
    contrib = rot_rel + p_n + v_n                                # [N,k,3]
    new_means = (w[..., None] * contrib).sum(dim=1)             # [N,3]

    q_blend = _blend_quat(q_n, w)                                # [N,4]
    new_quats = quat_mul(q_blend, dense.quats)
    new_quats = new_quats / torch.linalg.norm(new_quats, dim=-1, keepdim=True).clamp_min(1e-8)

    add_logs = (w[..., None] * dls_n).sum(dim=1).clamp(-dlog_scale_clamp, dlog_scale_clamp)
    new_scales = (dense.log_scales + add_logs).exp()
    new_logit_o = dense.opacity_logits + (w * dlo_n).sum(dim=1)
    new_opac = torch.sigmoid(new_logit_o)
    new_colors = (dense.colors + (w[..., None] * dc_n).sum(dim=1)).clamp(0.0, 1.0)

    return GaussianSet(new_means, new_quats, new_scales, new_opac, new_colors, dense.features)
