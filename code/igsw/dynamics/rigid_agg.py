"""§66 v10-rigid: batched weighted-Kabsch ENTITY AGGREGATION.

The rigid special case of low-rank motion (one SE(3) per entity per step), fitted to the
per-control motion VOTES (x_i -> x_i + v_i). Direction lives in the votes (output space),
so unlike the shelved feature-pooled entity head, aggregation cannot erase it (§65 audit:
projection preserved dir-cos 16/16). Zero learnable params; on an already-rigid field the
projection is the identity -> exact warm-start from v9-lang.

Fully vectorized: two index_add (weighted centroids) + one einsum/index_add (cross-cov)
+ one batched 3x3 SVD over E entities. No Python loop over entities; DDP/static_graph
safe (no data-dependent branching — degenerate entities are masked via torch.where).
"""
from __future__ import annotations

import torch


def _rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """[E,3,3] -> [E,3] axis-angle. Stable for small angles (the per-step regime:
    |omega| <= max_rot=0.3 rad); not intended for theta ~ pi."""
    skew = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                        R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], dim=-1)          # 2 sin(theta) * axis
    s = 0.5 * skew.norm(dim=-1)                                    # sin(theta)
    c = (0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0)).clamp(-1.0, 1.0)
    theta = torch.atan2(s, c)                                      # [E]
    axis = skew / (2.0 * s[:, None]).clamp_min(1e-12)
    return axis * theta[:, None]


def entity_rigid_aggregate(x: torch.Tensor, v: torch.Tensor, omega: torch.Tensor,
                           seg: torch.Tensor, w: torch.Tensor | None = None,
                           min_pts: int = 4, sv_eps: float = 1e-6,
                           exclude_ids: tuple = (0,), rot_from_omega: bool = True):
    """Project the per-control motion field onto per-entity SE(3) orbits.

    x     [N,3]    current control positions (this step's state)
    v     [B,N,3]  predicted per-control displacement votes (B=1 in the rollout)
    omega [B,N,3]  predicted per-control axis-angle rotations
    seg   [N]      entity ids (objects 1-7, arm 8 / sub-parts 50+c, gripper 10;
                   ids in exclude_ids — background 0 — pass through unchanged)
    w     [N]      optional vote weights (the dyn-gate probability); detached inside —
                   the gate keeps its own BCE gradient path, the fit only CONSUMES it.

    Returns (v_hat, omega_hat), same shapes/dtypes. For each aggregated entity every
    member gets v from the SAME rigid transform (R_e, t_e) and omega = axis-angle(R_e),
    so the control-state update AND the dense LBS inherit rigidity downstream.
    """
    assert v.shape[0] == 1, "rollout is B=1"
    dev = x.device
    seg = seg.long()
    agg = torch.ones_like(seg, dtype=torch.bool)
    for i in exclude_ids:
        agg &= seg != i
    if not bool(agg.any()):
        return v, omega
    ids = torch.unique(seg[agg])
    E = int(ids.numel())
    lut = torch.full((int(seg.max().item()) + 1,), -1, dtype=torch.long, device=dev)
    lut[ids] = torch.arange(E, device=dev)
    e_all = lut[seg]                                               # [N], -1 = passthrough
    m = e_all >= 0
    eM = e_all[m]                                                  # [Nm]

    # fp32 island: bf16 autocast is poison for SVD; manifold math wants full precision.
    with torch.autocast(device_type=dev.type, enabled=False):
        x32 = x[m].float()
        v32 = v[0][m].float()
        y32 = x32 + v32
        om32_in = omega[0][m].float()                             # SUPERVISED per-control rotation (rot_l)                                            # the votes
        if w is None:
            w32 = torch.ones(x32.shape[0], device=dev)
        else:
            w32 = w.detach().float().reshape(-1)[m].clamp_min(1e-3)
        Wsum = torch.zeros(E, device=dev).index_add_(0, eM, w32).clamp_min(1e-6)
        mux = torch.zeros(E, 3, device=dev).index_add_(0, eM, w32[:, None] * x32) / Wsum[:, None]
        muy = torch.zeros(E, 3, device=dev).index_add_(0, eM, w32[:, None] * y32) / Wsum[:, None]
        if rot_from_omega:
            # §74 R2: entity rotation = weighted mean of the SUPERVISED per-control omega (the rot_l
            # target, ~0 on a translating object, real on a hinge). The legacy velocity-Kabsch below
            # instead fit the vote SPREAD -> 28-31° spurious rotation on pure translations (§73). No SVD
            # here -> stable + fully differentiable (omega keeps its gradient; no detach hack).
            from .manifold import axis_angle_to_quat, quat_to_rotmat
            om_e = torch.zeros(E, 3, device=dev).index_add_(0, eM, w32[:, None] * om32_in) / Wsum[:, None]
            R = quat_to_rotmat(axis_angle_to_quat(om_e))           # [E,3,3]
        else:
            # (legacy ablation) v-Kabsch rotation — fits the velocity spread; detached (unstable backward).
            xc = x32 - mux[eM]
            yc = y32 - muy[eM]
            H = torch.zeros(E, 9, device=dev).index_add_(
                0, eM, (w32[:, None, None] * xc[:, :, None] * yc[:, None, :]).reshape(-1, 9)).view(E, 3, 3)
            H = H + 1e-9 * torch.eye(3, device=dev)
            U, S, Vt = torch.linalg.svd(H)
            Vm = Vt.transpose(-1, -2)
            det = torch.det(Vm @ U.transpose(-1, -2))
            D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
            R = Vm @ D @ U.transpose(-1, -2)
            npts = torch.zeros(E, device=dev).index_add_(0, eM, torch.ones_like(w32))
            degen = (npts < float(min_pts)) | (S[:, 1] <= sv_eps * S[:, 0].clamp_min(1e-12))
            R = torch.where(degen[:, None, None], torch.eye(3, device=dev).expand(E, 3, 3), R).detach()
            om_e = _rotmat_to_axis_angle(R)
        # unified rigid apply: rotate about the entity centroid mux, translate centroid mux -> muy.
        # (y_hat = R(x - mux) + muy ; identical algebra for both rotation sources.)
        y_hat = torch.einsum("nij,nj->ni", R[eM], (x32 - mux[eM])) + muy[eM]
        v_hat32 = y_hat - x32
        om32 = om_e[eM]

    v_out = v.clone()
    om_out = omega.clone()
    v_out[0, m] = v_hat32.to(v.dtype)
    om_out[0, m] = om32.to(omega.dtype)
    return v_out, om_out
