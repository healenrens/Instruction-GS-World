"""Canonical 3D Gaussian Splatting scene container.

All attributes are stored in ACTIVATED (physical) space so a GaussianSet can be
handed straight to a rasterizer:
    means      [N,3]   world-space centers
    quats      [N,4]   rotation, **wxyz**, unit-norm (gsplat convention)
    scales     [N,3]   per-axis std-dev, strictly positive
    opacities  [N]     in (0,1)
    colors     [N,3]   linear RGB in [0,1] (SH degree 0 / direct radiance)
    features   [N,D]   optional language-aligned semantic feature (may be None)

The dynamics model predicts deltas that are applied ON the manifold (see
igsw/dynamics) — Lie-algebra rotation update, log-scale update, logit-opacity
update — so this container also exposes the raw/unconstrained views needed there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


@dataclass
class GaussianSet:
    means: torch.Tensor          # [N,3]
    quats: torch.Tensor          # [N,4] wxyz unit
    scales: torch.Tensor         # [N,3] > 0
    opacities: torch.Tensor      # [N] in (0,1)
    colors: torch.Tensor         # [N,3] in [0,1]
    features: torch.Tensor | None = None  # [N,D] or None

    # ---- basics ----------------------------------------------------------
    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def device(self):
        return self.means.device

    def to(self, *args, **kwargs) -> "GaussianSet":
        f = self.features.to(*args, **kwargs) if self.features is not None else None
        return GaussianSet(
            self.means.to(*args, **kwargs),
            self.quats.to(*args, **kwargs),
            self.scales.to(*args, **kwargs),
            self.opacities.to(*args, **kwargs),
            self.colors.to(*args, **kwargs),
            f,
        )

    def detach(self) -> "GaussianSet":
        f = self.features.detach() if self.features is not None else None
        return GaussianSet(
            self.means.detach(), self.quats.detach(), self.scales.detach(),
            self.opacities.detach(), self.colors.detach(), f,
        )

    def clone(self) -> "GaussianSet":
        f = self.features.clone() if self.features is not None else None
        return GaussianSet(
            self.means.clone(), self.quats.clone(), self.scales.clone(),
            self.opacities.clone(), self.colors.clone(), f,
        )

    # ---- unconstrained (raw) views for optimization / delta application ---
    @property
    def log_scales(self) -> torch.Tensor:
        return torch.log(self.scales.clamp_min(1e-8))

    @property
    def opacity_logits(self) -> torch.Tensor:
        return inverse_sigmoid(self.opacities.clamp(1e-6, 1 - 1e-6))

    def validate(self):
        n = len(self)
        assert self.means.shape == (n, 3), self.means.shape
        assert self.quats.shape == (n, 4), self.quats.shape
        assert self.scales.shape == (n, 3), self.scales.shape
        assert self.opacities.shape == (n,), self.opacities.shape
        assert self.colors.shape == (n, 3), self.colors.shape
        assert torch.isfinite(self.means).all(), "non-finite means"
        return self

    # ---- io --------------------------------------------------------------
    def save_ply(self, path: str):
        """Write a minimal PLY (xyz + rgb) for quick visual inspection."""
        import numpy as np
        from plyfile import PlyData, PlyElement

        xyz = self.means.detach().cpu().numpy()
        rgb = (self.colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        verts = np.empty(
            xyz.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        PlyData([PlyElement.describe(verts, "vertex")]).write(path)
