"""§87 6D rotation representation (Zhou et al., CVPR'19 "On the Continuity of Rotation
Representations in Neural Networks") — the parametrization both new rotation heads share.

Why 6D: axis-angle/quaternion have discontinuities/double-cover; 6D + Gram-Schmidt is continuous
and, in the DELTA form used here (predict offsets around the identity frame), exactly identity at
zero output -> zero-init heads warm-start as a no-op. Math is done in an fp32 island (bf16-safe).
"""
from __future__ import annotations

import torch


def rot6d_delta_to_matrix(d6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[..., 6] zero-centred deltas -> [..., 3, 3] rotation. d6=0 => R=I exactly.

    Gram-Schmidt on (e1 + d6[:3], e2 + d6[3:]) per Zhou et al.; the identity offset keeps the
    input vectors away from the degenerate zero/parallel configuration at init.
    """
    with torch.autocast(device_type=d6.device.type, enabled=False):
        d = d6.float()
        a1 = d[..., 0:3] + torch.tensor([1.0, 0.0, 0.0], device=d.device)
        a2 = d[..., 3:6] + torch.tensor([0.0, 1.0, 0.0], device=d.device)
        b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
        a2p = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = a2p / a2p.norm(dim=-1, keepdim=True).clamp_min(eps)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)                  # columns -> [...,3,3]


def matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """[..., 3, 3] -> [..., 3] axis-angle. Stable for the small-angle per-step regime."""
    with torch.autocast(device_type=R.device.type, enabled=False):
        Rf = R.float()
        skew = torch.stack([Rf[..., 2, 1] - Rf[..., 1, 2],
                            Rf[..., 0, 2] - Rf[..., 2, 0],
                            Rf[..., 1, 0] - Rf[..., 0, 1]], dim=-1)
        s = 0.5 * skew.norm(dim=-1)                               # sin(theta)
        c = (0.5 * (Rf[..., 0, 0] + Rf[..., 1, 1] + Rf[..., 2, 2] - 1.0)).clamp(-1.0, 1.0)
        theta = torch.atan2(s, c)
        axis = skew / (2.0 * s[..., None]).clamp_min(eps)
        return axis * theta[..., None]
