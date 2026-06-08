"""Convert a feed-forward point map (Pi3/VGGT) into a GaussianSet.

Design choices (documented, not simplified):
- One Gaussian per surviving pixel (after conf + edge masking).
- **Scale init from local 3D pixel-neighbor spacing** (right/down neighbours in
  the per-view HxW grid). For image-lifted "surfels" this reflects the true local
  sampling density better than a global kNN, and is O(N). Robustly clamped to the
  [p_lo, p_hi] percentile band to reject residual depth-discontinuity outliers.
- Rotation = identity quaternion (isotropic init).
- Opacity = `opacity_init` (default 0.8 so an un-optimized cloud renders opaque).
- Colors = source pixel RGB in [0,1] (rendered as direct radiance / SH deg 0).
"""

from __future__ import annotations

import torch

from ..gaussians.types import GaussianSet


def _per_view_neighbor_scale(points: torch.Tensor) -> torch.Tensor:
    """points [N,H,W,3] -> per-pixel isotropic scale [N,H,W] from 3D grid spacing."""
    n, h, w, _ = points.shape
    # right neighbour distance
    d_right = torch.full((n, h, w), float("nan"), device=points.device)
    d_right[:, :, :-1] = torch.linalg.norm(points[:, :, 1:] - points[:, :, :-1], dim=-1)
    d_right[:, :, -1] = d_right[:, :, -2]
    # down neighbour distance
    d_down = torch.full((n, h, w), float("nan"), device=points.device)
    d_down[:, :-1, :] = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1)
    d_down[:, -1, :] = d_down[:, -2, :]
    return 0.5 * (d_right + d_down)


def points_to_gaussians(
    points: torch.Tensor,      # [N,H,W,3]
    images: torch.Tensor,      # [N,3,H,W] in [0,1]
    mask: torch.Tensor,        # [N,H,W] bool
    opacity_init: float = 0.8,
    scale_factor: float = 1.0,
    scale_pct: tuple[float, float] = (0.01, 0.99),
    min_scale: float = 1e-4,
    return_uv: bool = False,
) -> GaussianSet:
    device = points.device
    n, h, w, _ = points.shape
    colors_hwc = images.permute(0, 2, 3, 1)            # [N,H,W,3]
    raw_scale = _per_view_neighbor_scale(points)       # [N,H,W]
    # pixel coordinate (x=col, y=row) of each location, for correspondence/tracking
    vv, uu = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    uv_grid = torch.stack([uu, vv], dim=-1)[None].expand(n, h, w, 2)   # [N,H,W,2]

    m = mask & torch.isfinite(raw_scale) & torch.isfinite(points).all(dim=-1)
    means = points[m]                                  # [M,3]
    colors = colors_hwc[m].clamp(0, 1)                 # [M,3]
    scl = raw_scale[m]                                 # [M]
    uv_kept = uv_grid[m]                               # [M,2]

    # robust clamp of scale to reject depth-edge outliers (guard degenerate/empty clips)
    if scl.numel() >= 16:
        lo = torch.quantile(scl, scale_pct[0])
        hi = torch.quantile(scl, scale_pct[1])
        scl = scl.clamp(min=max(min_scale, float(lo)), max=float(hi)) * scale_factor
    else:
        scl = scl.clamp_min(min_scale) * scale_factor

    m_count = means.shape[0]
    scales = scl[:, None].repeat(1, 3)                 # isotropic [M,3]
    quats = torch.zeros((m_count, 4), device=device)
    quats[:, 0] = 1.0                                   # identity wxyz
    opacities = torch.full((m_count,), float(opacity_init), device=device)

    gs = GaussianSet(
        means=means.contiguous(),
        quats=quats,
        scales=scales.contiguous(),
        opacities=opacities,
        colors=colors.contiguous(),
    ).validate()
    if return_uv:
        return gs, uv_kept.contiguous()
    return gs


def lift_result_to_gaussians(res: dict, **kwargs) -> GaussianSet:
    """Convenience: take Pi3Lifter.lift() output dict -> GaussianSet."""
    return points_to_gaussians(res["points"], res["images"], res["mask"], **kwargs)
