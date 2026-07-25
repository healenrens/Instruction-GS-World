"""Shared causal geometry for RoboTwin VLA training and deployment.

The input geometry is computed from one current RGB frame on a fixed grid. Full-video
SpaTracker output is accepted only by ``tracker_targets_on_grid`` to build supervision.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


CAUSAL_GEOMETRY_VERSION = "vggt_t1_grid48_v1"


def regular_grid_uv(height: int, width: int, grid_size: int = 48, *, device=None,
                    dtype=torch.float32) -> torch.Tensor:
    """Match SpaTracker's frame-0 query grid exactly, in (x, y) pixel order."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    margin = width // 64
    ys = torch.linspace(margin, height - margin, grid_size, device=device, dtype=dtype)
    xs = torch.linspace(margin, width - margin, grid_size, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx.flatten(), gy.flatten()), dim=-1)


def unproject_uv(uv: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject pixel coordinates with camera-frame depth."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (uv[..., 0] - cx) * depth / fx
    y = (uv[..., 1] - cy) * depth / fy
    return torch.stack((x, y, depth), dim=-1)


def project_xyz(xyz: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project camera-frame XYZ to pixels."""
    z = xyz[..., 2].clamp_min(1e-6)
    u = intrinsics[0, 0] * xyz[..., 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * xyz[..., 1] / z + intrinsics[1, 2]
    return torch.stack((u, v), dim=-1)


def _single_map(points_map: torch.Tensor) -> torch.Tensor:
    while points_map.ndim > 3:
        if points_map.shape[0] != 1:
            raise ValueError(f"expected singleton VGGT dimensions, got {tuple(points_map.shape)}")
        points_map = points_map[0]
    if points_map.ndim != 3 or points_map.shape[-1] != 3:
        raise ValueError(f"expected VGGT point map [H,W,3], got {tuple(points_map.shape)}")
    return points_map.float()


def _single_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    while intrinsics.ndim > 2:
        if intrinsics.shape[0] != 1:
            raise ValueError(f"expected singleton VGGT dimensions, got {tuple(intrinsics.shape)}")
        intrinsics = intrinsics[0]
    if intrinsics.shape != (3, 3):
        raise ValueError(f"expected intrinsics [3,3], got {tuple(intrinsics.shape)}")
    return intrinsics.float()


def causal_geometry_from_prediction(points_map: torch.Tensor, intrinsics: torch.Tensor,
                                    grid_size: int = 48) -> dict[str, torch.Tensor | int]:
    """Build fixed-grid current geometry from a single-frame VGGT prediction.

    Only the predicted depth channel is sampled. XYZ is re-unprojected from the exact
    fixed-grid UV coordinates so projection is deterministic and shared with deployment.
    """
    pmap = _single_map(points_map)
    K = _single_intrinsics(intrinsics).to(device=pmap.device)
    height, width = int(pmap.shape[0]), int(pmap.shape[1])
    uv = regular_grid_uv(height, width, grid_size, device=pmap.device, dtype=pmap.dtype)
    norm = torch.stack((2.0 * uv[:, 0] / (width - 1) - 1.0,
                        2.0 * uv[:, 1] / (height - 1) - 1.0), dim=-1)
    depth = F.grid_sample(pmap[..., 2][None, None], norm[None, None], mode="bilinear",
                          align_corners=True).reshape(-1)
    if not bool(torch.isfinite(depth).all()) or not bool((depth > 1e-3).all()):
        raise ValueError("single-frame VGGT produced invalid grid depth")
    means = unproject_uv(uv, depth, K)
    center = means.mean(dim=0, keepdim=True)
    radius = (means - center).norm(dim=-1).amax().clamp_min(1e-6)
    return {"means": means, "uv": uv, "K_intr": K, "center": center, "radius": radius,
            "H": height, "W": width}


def preprocessed_rgb(video_tensor: torch.Tensor) -> torch.Tensor:
    """Convert preprocess_image output [1,1,3,H,W] (0..255) to uint8 [H,W,3]."""
    x = video_tensor
    while x.ndim > 3:
        if x.shape[0] != 1:
            raise ValueError(f"expected one current frame, got {tuple(video_tensor.shape)}")
        x = x[0]
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"expected preprocessed RGB [3,H,W], got {tuple(x.shape)}")
    return x.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)


def tracker_targets_on_grid(current_xyz: torch.Tensor, grid_uv: torch.Tensor,
                            current_intrinsics: torch.Tensor, tracker_uv0: torch.Tensor,
                            tracker_traj: torch.Tensor, tracker_intrinsics: torch.Tensor,
                            tracker_vis: torch.Tensor | None, max_match_px: float = 2.0) -> dict:
    """Attach full-video tracker labels to the fixed causal input grid.

    Track visibility and matching affect only ``geom_valid`` and target tensors. They never
    remove or alter current input tokens. Duplicate tracker outputs map to one best label.
    """
    dev = current_xyz.device
    grid_uv = grid_uv.to(dev).float()
    old_uv0 = tracker_uv0.to(dev).float()
    old_traj = tracker_traj.to(dev).float()
    old_K = tracker_intrinsics.to(dev).float()
    vis = (torch.ones(old_traj.shape[:2], dtype=torch.bool, device=dev) if tracker_vis is None
           else tracker_vis.to(dev).bool())
    if old_traj.ndim != 3 or old_traj.shape[1] != old_uv0.shape[0]:
        raise ValueError("tracker UV/trajectory shape mismatch")
    if vis.shape != old_traj.shape[:2]:
        raise ValueError("tracker visibility shape mismatch")

    grid_size = int(round(grid_uv.shape[0] ** 0.5))
    if grid_size * grid_size != grid_uv.shape[0]:
        raise ValueError("target grid must be square")
    x_coords = grid_uv[:grid_size, 0]
    y_coords = grid_uv[::grid_size, 1]
    ix = (old_uv0[:, None, 0] - x_coords[None]).abs().argmin(dim=1)
    iy = (old_uv0[:, None, 1] - y_coords[None]).abs().argmin(dim=1)
    nearest = iy * grid_size + ix
    distance = (old_uv0 - grid_uv[nearest]).norm(dim=-1)
    finite = torch.isfinite(old_traj).all(dim=(0, 2)) & (old_traj[0, :, 2] > 1e-3)
    usable = finite & vis[0] & vis[-1] & (distance <= max_match_px)
    best: dict[int, tuple[tuple[float, float], int]] = {}
    for i in usable.nonzero(as_tuple=False).flatten().tolist():
        j = int(nearest[i])
        score = (-float(vis[:, i].float().mean()), float(distance[i]))
        if j not in best or score < best[j][0]:
            best[j] = (score, i)

    steps, n_grid = old_traj.shape[0], grid_uv.shape[0]
    traj = current_xyz[None].repeat(steps, 1, 1)
    mapped_vis = torch.zeros(steps, n_grid, dtype=torch.bool, device=dev)
    geom_valid = torch.zeros(n_grid, dtype=torch.bool, device=dev)
    match_distance = torch.full((n_grid,), float("inf"), device=dev)
    if best:
        grid_index = torch.tensor(list(best), dtype=torch.long, device=dev)
        track_index = torch.tensor([best[j][1] for j in best], dtype=torch.long, device=dev)
        selected = old_traj[:, track_index]
        target_uv = project_xyz(selected, old_K)
        depth_ratio = selected[..., 2] / selected[0, :, 2][None]
        target_depth = current_xyz[grid_index, 2][None] * depth_ratio
        target_xyz = unproject_uv(target_uv, target_depth, current_intrinsics)
        valid_target = torch.isfinite(target_xyz).all(dim=(0, 2)) & (target_depth > 1e-3).all(dim=0)
        grid_index = grid_index[valid_target]; track_index = track_index[valid_target]
        target_xyz = target_xyz[:, valid_target]
        traj[:, grid_index] = target_xyz
        traj[0, grid_index] = current_xyz[grid_index]
        mapped_vis[:, grid_index] = vis[:, track_index]
        geom_valid[grid_index] = mapped_vis[-1, grid_index]
        match_distance[grid_index] = distance[track_index]

    return {"traj": traj, "vis": mapped_vis, "geom_valid": geom_valid,
            "match_distance": match_distance, "matched_tracks": int(geom_valid.sum())}
