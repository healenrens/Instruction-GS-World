"""Evaluation-only motion coherence for current-frame slot assignments."""
from __future__ import annotations

import torch


def _motion_r2(weights: torch.Tensor, displacement: torch.Tensor) -> float:
    slot_mass = weights.sum(dim=0).clamp_min(1e-6)
    slot_motion = torch.einsum(
        "pk,pd->kd",
        weights,
        displacement,
    ) / slot_mass[:, None]
    prediction = weights @ slot_motion
    global_prediction = displacement.mean(dim=0, keepdim=True)
    slot_error = (prediction - displacement).square().sum()
    global_error = (global_prediction - displacement).square().sum()
    if float(global_error) <= 1e-12:
        return 0.0
    return float(1.0 - slot_error / global_error)


def slot_motion_scores(
    output: dict,
    pair_paths: list[str],
    grid_height: int,
    grid_width: int,
) -> dict[str, list[float]]:
    """Use future trajectories only as labels for current slot assignments."""
    token_assignment = output["history_token_states"][-1].assignment.float()
    slot_assignment = output["history_slot_states"][-1].assignment.float()
    patch_to_slot = torch.einsum(
        "bmn,bmk->bnk",
        token_assignment,
        slot_assignment,
    ).cpu()
    scores = {"motion_r2": [], "dynamic_motion_r2": [], "shuffled_motion_r2": []}
    for batch_index, path in enumerate(pair_paths):
        pair = torch.load(path, map_location="cpu", weights_only=False)
        trajectory = pair["traj"].float()
        visibility = pair["vis"].bool()
        valid = (
            pair["geom_valid"].bool()
            & visibility[0]
            & visibility[-1]
            & torch.isfinite(trajectory[0]).all(dim=-1)
            & torch.isfinite(trajectory[-1]).all(dim=-1)
        )
        if int(valid.sum()) < 8:
            continue
        uv = pair["uv"][valid].float()
        x = (
            uv[:, 0] / max(float(pair["W"] - 1), 1.0) * (grid_width - 1)
        ).round().long().clamp(0, grid_width - 1)
        y = (
            uv[:, 1] / max(float(pair["H"] - 1), 1.0) * (grid_height - 1)
        ).round().long().clamp(0, grid_height - 1)
        patch_index = y * grid_width + x
        weights = patch_to_slot[batch_index, patch_index]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        displacement = trajectory[-1, valid] - trajectory[0, valid]
        scores["motion_r2"].append(_motion_r2(weights, displacement))
        magnitude = displacement.norm(dim=-1)
        dynamic = magnitude >= torch.quantile(magnitude, 0.75)
        if int(dynamic.sum()) >= 8:
            scores["dynamic_motion_r2"].append(
                _motion_r2(weights[dynamic], displacement[dynamic])
            )
        shuffle = torch.arange(len(weights) - 1, -1, -1)
        scores["shuffled_motion_r2"].append(
            _motion_r2(weights[shuffle], displacement)
        )
    return scores
