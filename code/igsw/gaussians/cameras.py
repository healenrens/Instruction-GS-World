"""Camera utilities for the lifted geometry.

Pi3 returns `camera_poses` (camera->world, OpenCV) and per-camera `local_points`
but NO explicit intrinsics. We recover a pinhole K from the local point map by
the exact pinhole relation (OpenCV: x right, y down, z forward):
    u = fx * (X/Z) + cx ,   v = fy * (Y/Z) + cy
solved by least squares over all valid pixels.
"""

from __future__ import annotations

import torch


def intrinsics_from_local_points(local_points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """local_points [H,W,3] (camera frame) -> K [3,3] (pinhole, least squares)."""
    h, w, _ = local_points.shape
    dev = local_points.device
    vv, uu = torch.meshgrid(
        torch.arange(h, device=dev, dtype=torch.float32),
        torch.arange(w, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    valid = torch.isfinite(z) & (z.abs() > eps) & torch.isfinite(x) & torch.isfinite(y)
    xz = (x[valid] / z[valid])
    yz = (y[valid] / z[valid])
    u = uu[valid]
    v = vv[valid]

    def _solve(a, b):  # b = p0*a + p1
        A = torch.stack([a, torch.ones_like(a)], dim=1)  # [M,2]
        sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
        return sol[0], sol[1]

    fx, cx = _solve(xz, u)
    fy, cy = _solve(yz, v)
    K = torch.eye(3, device=dev, dtype=torch.float32)
    K[0, 0], K[0, 2] = fx, cx
    K[1, 1], K[1, 2] = fy, cy
    return K


def viewmat_from_pose(camera_pose_c2w: torch.Tensor) -> torch.Tensor:
    """camera->world [4,4] -> world->camera viewmat [4,4] (what rasterizers want)."""
    return torch.linalg.inv(camera_pose_c2w)
