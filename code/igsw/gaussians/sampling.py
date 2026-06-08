"""Downsample a GaussianSet to a fixed control-set size for the dynamics model.

v1 uses uniform random subsampling (fast, unbiased). A voxel-grid option keeps
spatial coverage. Farthest-point sampling is left for v2 (O(N·K) is too slow at
N~120k without a CUDA kernel; gsplat/torch-cluster FPS can be dropped in later).
"""

from __future__ import annotations

import torch

from .types import GaussianSet


def _gather(gs: GaussianSet, idx: torch.Tensor) -> GaussianSet:
    f = gs.features[idx] if gs.features is not None else None
    return GaussianSet(gs.means[idx], gs.quats[idx], gs.scales[idx],
                       gs.opacities[idx], gs.colors[idx], f)


def downsample_gaussians(
    gs: GaussianSet,
    n_target: int,
    method: str = "random",
    generator: torch.Generator | None = None,
    voxel: float | None = None,
) -> GaussianSet:
    n = len(gs)
    if n <= n_target:
        # pad by sampling with replacement to reach exactly n_target (keeps fixed N)
        if n == n_target:
            return gs
        extra = torch.randint(0, n, (n_target - n,), device=gs.device, generator=generator)
        idx = torch.cat([torch.arange(n, device=gs.device), extra])
        return _gather(gs, idx)

    if method == "voxel" and voxel is not None:
        q = torch.floor(gs.means / voxel).long()
        keys = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
        _, first = torch.unique(keys, return_inverse=False, return_counts=False), None
        # pick one per voxel (first occurrence)
        order = torch.argsort(keys)
        sk = keys[order]
        keep_mask = torch.ones_like(sk, dtype=torch.bool)
        keep_mask[1:] = sk[1:] != sk[:-1]
        idx = order[keep_mask]
        if idx.numel() > n_target:
            sel = torch.randperm(idx.numel(), device=gs.device, generator=generator)[:n_target]
            idx = idx[sel]
        elif idx.numel() < n_target:
            extra = torch.randint(0, len(gs), (n_target - idx.numel(),), device=gs.device, generator=generator)
            idx = torch.cat([idx, extra])
        return _gather(gs, idx)

    # random (default)
    idx = torch.randperm(n, device=gs.device, generator=generator)[:n_target]
    return _gather(gs, idx)
