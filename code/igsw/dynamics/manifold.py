"""On-manifold Gaussian delta application (wxyz quaternion convention).

The dynamics head predicts per-Gaussian deltas; we apply them on the correct
manifolds so long autoregressive rollouts do not drift (additive-quaternion /
additive-scale updates, used by ManiGaussian/4DGS, accumulate error):
    μ ← μ + v                         (world-frame translation)
    q ← normalize( Exp(ω) ⊗ q )       (SO(3) left-multiply, ω = so(3) tangent)
    s ← s · exp(δs)                   (positive scale stays positive)
    σ ← sigmoid( logit(σ) + δσ )      (opacity stays in (0,1))
    c ← clamp( c + δc , 0, 1 )
    f ← f + δf
"""

from __future__ import annotations

import torch

from ..gaussians.types import GaussianSet, inverse_sigmoid


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions. a,b: [...,4] -> [...,4]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def axis_angle_to_quat(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """so(3) tangent vector ω [...,3] -> unit wxyz quaternion (exp map).

    NaN-grad safety: angle = sqrt(Σω² + eps²) keeps the EPS INSIDE the sqrt so the
    gradient ∂angle/∂ω = ω/sqrt(Σω²+eps²) is finite even at ω=0 (plain ‖ω‖ gives 0/0
    = NaN there, and a torch.where(small, …, sin/‖ω‖) still back-props the unsafe branch
    as 0·NaN=NaN). With angle ≥ eps>0, sin(half)/angle needs no branch and is smooth
    everywhere; at ω→0 it →0.5 (identity quat), the correct limit.
    """
    angle = torch.sqrt((omega * omega).sum(dim=-1, keepdim=True) + eps * eps)   # [...,1], ≥ eps
    half = 0.5 * angle
    ratio = torch.sin(half) / angle                                  # safe (angle ≥ eps), grad finite at 0
    xyz = omega * ratio
    w = torch.cos(half)
    q = torch.cat([w, xyz], dim=-1)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)


def quat_to_rotmat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """wxyz unit quaternion [...,4] -> rotation matrix [...,3,3]."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def apply_deltas_tensors(
    means, quats, scales, opacity_logits, colors, features,
    v, omega, dlog_scale, dlogit_opacity, dcolor, dfeature=None,
    dlog_scale_clamp: float = 3.0,
):
    """Batched on-manifold update on raw tensors ([B,N,*] or [N,*]). Returns new tensors."""
    new_means = means + v
    nq = quat_mul(axis_angle_to_quat(omega), quats)
    new_quats = nq / torch.linalg.norm(nq, dim=-1, keepdim=True).clamp_min(1e-8)
    new_scales = scales * torch.exp(dlog_scale.clamp(-dlog_scale_clamp, dlog_scale_clamp))
    new_opacities = torch.sigmoid(opacity_logits + dlogit_opacity.squeeze(-1))
    new_colors = (colors + dcolor).clamp(0.0, 1.0)
    new_features = None
    if features is not None:
        new_features = features + dfeature if dfeature is not None else features
    return new_means, new_quats, new_scales, new_opacities, new_colors, new_features


def apply_deltas(
    gs: GaussianSet,
    v: torch.Tensor,          # [N,3] translation
    omega: torch.Tensor,      # [N,3] so(3) tangent
    dlog_scale: torch.Tensor, # [N,3]
    dlogit_opacity: torch.Tensor,  # [N] or [N,1]
    dcolor: torch.Tensor,     # [N,3]
    dfeature: torch.Tensor | None = None,  # [N,D]
    dlog_scale_clamp: float = 3.0,
) -> GaussianSet:
    """Return a NEW GaussianSet with deltas applied on-manifold (differentiable)."""
    new_means = gs.means + v
    new_quats = quat_mul(axis_angle_to_quat(omega), gs.quats)
    new_quats = new_quats / torch.linalg.norm(new_quats, dim=-1, keepdim=True).clamp_min(1e-8)
    new_scales = gs.scales * torch.exp(dlog_scale.clamp(-dlog_scale_clamp, dlog_scale_clamp))
    dlogit = dlogit_opacity.reshape(-1)
    new_opacities = torch.sigmoid(gs.opacity_logits + dlogit)
    new_colors = (gs.colors + dcolor).clamp(0.0, 1.0)
    new_features = None
    if gs.features is not None and dfeature is not None:
        new_features = gs.features + dfeature
    elif gs.features is not None:
        new_features = gs.features
    return GaussianSet(new_means, new_quats, new_scales, new_opacities, new_colors, new_features)
