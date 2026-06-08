"""Differentiable rendering of a GaussianSet via gsplat.

gsplat.rasterization expects:
    means [N,3], quats [N,4] (wxyz, will be normalized internally),
    scales [N,3] (>0), opacities [N] in (0,1), colors [N,3] (RGB, sh_degree=None),
    viewmats [C,4,4] (world->camera), Ks [C,3,3], image width/height.
Returns render_colors [C,H,W,3], render_alphas [C,H,W,1], meta.
"""

from __future__ import annotations

import torch

from .types import GaussianSet


def render_gaussianset(
    gs: GaussianSet,
    viewmats: torch.Tensor,   # [C,4,4] world->cam
    Ks: torch.Tensor,         # [C,3,3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    backgrounds: torch.Tensor | None = None,
):
    """Returns (colors [C,H,W,3], alphas [C,H,W,1], meta)."""
    from gsplat import rasterization

    if viewmats.ndim == 2:
        viewmats = viewmats[None]
    if Ks.ndim == 2:
        Ks = Ks[None]
    colors, alphas, meta = rasterization(
        means=gs.means,
        quats=gs.quats,
        scales=gs.scales,
        opacities=gs.opacities,
        colors=gs.colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="RGB",
        sh_degree=None,
        backgrounds=backgrounds,
    )
    return colors, alphas, meta


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))
