"""S0 — sparse 2D-Gaussian token extraction for the GPSToken-JEPA world model (PLAN §3).

A token is born from the frame-0 image, NOT from the dense 26万 Gaussians:
  1. PLACE   : entropy + motion/relevance saliency adaptive partition (reuse igsw.gaussians.gpstoken,
               §93, cross-validated vs the GPSToken authors) -> M token centers (uv) + footprints (sigma).
  2. LIFT 3D : each token center -> nearest dense Gaussian in uv space -> its 3D `means` = the token's
               frame-0 3D position (the dense means ARE the depth-correct Pi3/sim points; PLAN "深度脚手架").
               The nearest-dense index `idx` also gives the per-token GT trajectory: traj[:, idx].
  3. FEATURE : bilinear-sample the FROZEN Qwen image-patch grid at uv -> per-token frozen feature
               (PLAN §3.2: base feature = frozen perception; JEPA shapes it dynamics-aware later).

Returns a plain dict of tensors (no nn.Module) so it is reusable by trainer + eval.
"""
from __future__ import annotations

import numpy as np
import torch

from ..gaussians.gpstoken import grad_mag, gpstoken_init, mover_saliency  # noqa: F401 (re-export)


def place_tokens(rgb_uint8: np.ndarray, uv: torch.Tensor, n_keep: int, L: int, dev,
                 sal=None, beta: float = 0.0):
    """entropy(+saliency) partition -> token centers/sigma (pixels) + nearest-dense index.
    Returns cen [M,2], sigma [M,2], idx [M] (M may be < L after dedup of nearest-dense collisions)."""
    import cv2
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    E = grad_mag(gray)
    _, gauss = gpstoken_init(E, min(L, n_keep), sal=sal, beta=beta)
    cen = torch.tensor([[g[0], g[1]] for g in gauss], device=dev, dtype=torch.float32)   # [L,2]
    sig = torch.tensor([[g[2], g[3]] for g in gauss], device=dev, dtype=torch.float32)   # [L,2]
    uvk = uv[:n_keep].to(dev).float()                                                     # [n_keep,2]
    nn_idx = torch.cdist(cen, uvk).argmin(dim=1)                                          # [L]
    # dedup collisions so each token is a distinct dense identity (keeps the per-token GT clean)
    uniq, first = torch.unique(nn_idx, return_inverse=False), None
    keep = torch.zeros(nn_idx.shape[0], dtype=torch.bool, device=dev)
    seen = set()
    order = []
    for i, v in enumerate(nn_idx.tolist()):
        if v not in seen:
            seen.add(v)
            order.append(i)
            keep[i] = True
    sel = torch.tensor(order, device=dev, dtype=torch.long)
    return cen[sel], sig[sel], nn_idx[sel]


def sample_grid_feat(grid: torch.Tensor, ghw, uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinear-sample a frozen patch-feature grid [gh,gw,Hf] at pixel coords uv [M,2] (x=col,y=row in
    HxW image space) -> per-token feature [M,Hf]. grid_sample handles the (different) grid resolution."""
    g = grid.permute(2, 0, 1)[None].float()                                  # [1,Hf,gh,gw]
    u = uv[:, 0] / max(W, 1.0) * 2.0 - 1.0
    v = uv[:, 1] / max(H, 1.0) * 2.0 - 1.0
    samp = torch.stack([u, v], dim=-1)[None, None]                           # [1,1,M,2] (x,y)
    out = torch.nn.functional.grid_sample(g, samp.to(g.dtype), align_corners=False, mode="bilinear")
    return out[0, :, 0, :].transpose(0, 1).contiguous()                      # [M,Hf]


def project_to_uv(xyz_world: torch.Tensor, K_intr: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """World 3D [M,3] -> pixel uv [M,2] via viewmat (world->cam) + pinhole K. Used to find each token's
    FUTURE uv (from the GT track) so the JEPA target feature is sampled where the token actually moved."""
    ones = torch.ones(xyz_world.shape[0], 1, device=xyz_world.device, dtype=xyz_world.dtype)
    cam = (torch.cat([xyz_world, ones], dim=-1) @ viewmat.T)[:, :3]          # [M,3]
    z = cam[:, 2:3].clamp_min(1e-4)
    uvh = cam[:, :3] / z                                                     # normalized
    uv = (uvh @ K_intr.T)[:, :2]                                             # [M,2]
    return uv


def to_cam(xyz_world: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """World 3D -> camera-frame 3D [M,3] (for the 2D-flow+depth geometry variant)."""
    ones = torch.ones(xyz_world.shape[0], 1, device=xyz_world.device, dtype=xyz_world.dtype)
    return (torch.cat([xyz_world, ones], dim=-1) @ viewmat.T)[:, :3]


def cam_to_world(cam: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """Camera-frame 3D -> world 3D (inverse of to_cam)."""
    inv = torch.linalg.inv(viewmat)
    ones = torch.ones(cam.shape[0], 1, device=cam.device, dtype=cam.dtype)
    return (torch.cat([cam, ones], dim=-1) @ inv.T)[:, :3]
