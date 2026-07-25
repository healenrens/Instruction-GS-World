"""Correctness tests for arbitrary-frame causal tracker re-anchoring."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.causal_geometry import project_xyz, regular_grid_uv, unproject_uv  # noqa: E402
from igsw.latent_particle_wm.pair_targets import tracker_pair_targets_on_grid  # noqa: E402


def main() -> None:
    height, width, grid = 80, 96, 4
    intrinsics = torch.tensor(
        [[100.0, 0.0, width / 2], [0.0, 100.0, height / 2], [0.0, 0.0, 1.0]]
    )
    uv = regular_grid_uv(height, width, grid)
    depth = torch.linspace(1.0, 1.3, len(uv))
    current_xyz = unproject_uv(uv, depth, intrinsics)

    frames = 5
    tracker = []
    for frame in range(frames):
        shifted_uv = uv + uv.new_tensor([frame * 0.5, frame * -0.25])
        shifted_depth = depth * (1.0 + frame * 0.02)
        tracker.append(unproject_uv(shifted_uv, shifted_depth, intrinsics))
    tracker = torch.stack(tracker)
    visibility = torch.ones(frames, len(uv), dtype=torch.bool)
    visibility[2, 3] = False

    result = tracker_pair_targets_on_grid(
        current_xyz,
        uv,
        intrinsics,
        tracker,
        intrinsics,
        visibility,
        start=1,
        end=4,
        max_match_px=1.0,
    )
    valid = result["geom_valid"]
    assert int(valid.sum()) == len(uv)
    assert torch.equal(result["traj"][0], current_xyz)
    tracker_uv = project_xyz(tracker[1:5], intrinsics)
    expected_uv = uv[None] + tracker_uv - tracker_uv[0:1]
    actual_uv = project_xyz(result["traj"][:, valid], intrinsics)
    assert torch.allclose(actual_uv, expected_uv[:, valid], atol=1e-4)
    expected_ratio = tracker[1:5, valid, 2] / tracker[1, valid, 2][None]
    actual_ratio = result["traj"][:, valid, 2] / current_xyz[valid, 2][None]
    assert torch.allclose(actual_ratio, expected_ratio, atol=1e-5)

    second = tracker_pair_targets_on_grid(
        current_xyz,
        uv,
        intrinsics,
        tracker * tracker.new_tensor([1.0, 1.0, 1.1]),
        intrinsics,
        visibility,
        start=1,
        end=3,
        max_match_px=20.0,
    )
    assert torch.equal(second["traj"][0], current_xyz)
    assert second["geom_valid"].shape == result["geom_valid"].shape
    print("[OK] arbitrary-frame tracker targets preserve causal current geometry")


if __name__ == "__main__":
    main()
