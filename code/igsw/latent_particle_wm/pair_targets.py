"""Attach full-video tracker targets to a causal grid anchored at any frame."""
from __future__ import annotations

import torch

from igsw.causal_geometry import project_xyz, unproject_uv

CAUSAL_PAIR_VERSION = "vggt_t1_grid48_arbitrary_pair_v1"


def _nearest_regular_grid(grid_uv: torch.Tensor, query_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grid_size = int(round(grid_uv.shape[0] ** 0.5))
    if grid_size * grid_size != grid_uv.shape[0]:
        raise ValueError("target grid must be square")
    x_coords = grid_uv[:grid_size, 0]
    y_coords = grid_uv[::grid_size, 1]
    ix = (query_uv[:, None, 0] - x_coords[None]).abs().argmin(dim=1)
    iy = (query_uv[:, None, 1] - y_coords[None]).abs().argmin(dim=1)
    nearest = iy * grid_size + ix
    return nearest, (query_uv - grid_uv[nearest]).norm(dim=-1)


def tracker_pair_targets_on_grid(
    current_xyz: torch.Tensor,
    grid_uv: torch.Tensor,
    current_intrinsics: torch.Tensor,
    tracker_traj: torch.Tensor,
    tracker_intrinsics: torch.Tensor,
    tracker_vis: torch.Tensor | None,
    start: int,
    end: int,
    max_match_px: float = 2.0,
) -> dict[str, torch.Tensor | int]:
    """Re-anchor tracker supervision at `start` without changing current inputs.

    Tracker image motion and relative depth change are transferred to the
    independently reconstructed current-frame geometry. The returned trajectory
    starts exactly at `current_xyz`.
    """
    if not (0 <= start < end < tracker_traj.shape[0]):
        raise ValueError(f"invalid pair ({start}, {end}) for {tracker_traj.shape[0]} frames")
    device = current_xyz.device
    grid_uv = grid_uv.to(device).float()
    tracker_traj = tracker_traj.to(device).float()
    tracker_intrinsics = tracker_intrinsics.to(device).float()
    visibility = (
        torch.ones(tracker_traj.shape[:2], dtype=torch.bool, device=device)
        if tracker_vis is None
        else tracker_vis.to(device).bool()
    )
    if tracker_traj.ndim != 3 or tracker_traj.shape[-1] != 3:
        raise ValueError("tracker trajectory must have shape [T,N,3]")
    if visibility.shape != tracker_traj.shape[:2]:
        raise ValueError("tracker visibility shape mismatch")

    path = tracker_traj[start : end + 1]
    path_visibility = visibility[start : end + 1]
    anchor_uv = project_xyz(path[0], tracker_intrinsics)
    nearest, distance = _nearest_regular_grid(grid_uv, anchor_uv)
    finite = torch.isfinite(path).all(dim=(0, 2)) & torch.isfinite(anchor_uv).all(dim=-1)
    finite &= (path[..., 2] > 1e-3).all(dim=0)
    usable = finite & path_visibility[0] & path_visibility[-1] & (distance <= max_match_px)

    best: dict[int, tuple[tuple[float, float], int]] = {}
    for track_index in usable.nonzero(as_tuple=False).flatten().tolist():
        grid_index = int(nearest[track_index])
        score = (
            -float(path_visibility[:, track_index].float().mean()),
            float(distance[track_index]),
        )
        if grid_index not in best or score < best[grid_index][0]:
            best[grid_index] = (score, track_index)

    steps = end - start + 1
    particle_count = len(grid_uv)
    output_traj = current_xyz[None].repeat(steps, 1, 1)
    output_vis = torch.zeros(steps, particle_count, dtype=torch.bool, device=device)
    geom_valid = torch.zeros(particle_count, dtype=torch.bool, device=device)
    match_distance = torch.full((particle_count,), float("inf"), device=device)
    if best:
        grid_index = torch.tensor(list(best), dtype=torch.long, device=device)
        track_index = torch.tensor([best[index][1] for index in best], dtype=torch.long, device=device)
        selected_path = path[:, track_index]
        tracker_path_uv = project_xyz(selected_path, tracker_intrinsics)
        target_uv = grid_uv[grid_index][None] + tracker_path_uv - tracker_path_uv[0:1]
        depth_ratio = selected_path[..., 2] / selected_path[0, :, 2][None]
        target_depth = current_xyz[grid_index, 2][None] * depth_ratio
        target_xyz = unproject_uv(target_uv, target_depth, current_intrinsics)
        target_valid = torch.isfinite(target_xyz).all(dim=(0, 2))
        target_valid &= (target_depth > 1e-3).all(dim=0)
        grid_index = grid_index[target_valid]
        track_index = track_index[target_valid]
        target_xyz = target_xyz[:, target_valid]
        output_traj[:, grid_index] = target_xyz
        output_traj[0, grid_index] = current_xyz[grid_index]
        output_vis[:, grid_index] = path_visibility[:, track_index]
        geom_valid[grid_index] = output_vis[-1, grid_index]
        match_distance[grid_index] = distance[track_index]

    return {
        "traj": output_traj,
        "vis": output_vis,
        "geom_valid": geom_valid,
        "match_distance": match_distance,
        "matched_tracks": int(geom_valid.sum()),
    }
