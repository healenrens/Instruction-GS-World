"""Map model slots through frozen patch trajectories instead of slot indices."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory_teacher import TrajectoryEvidence


@dataclass(frozen=True)
class SlotTrajectory:
    adjacent: torch.Tensor
    confidence: torch.Tensor
    endpoint: torch.Tensor


def build_slot_trajectory(
    assignment: torch.Tensor,
    evidence: TrajectoryEvidence,
    object_slots: int,
) -> SlotTrajectory:
    if assignment.ndim != 4 or assignment.shape[1] < 2:
        raise ValueError("slot trajectory expects [B,T,N,O] assignments")
    objects = assignment[..., :object_slots].float()
    source = objects[:, :-1]
    target = objects[:, 1:]
    mass = torch.einsum(
        "btik,btij,btjl->btkl",
        source.detach(),
        evidence.forward.float(),
        target.detach(),
    )
    row_mass = mass.sum(dim=-1)
    adjacent = mass / row_mass[..., None].clamp_min(1e-6)
    confidence = row_mass / source.detach().sum(dim=2).clamp_min(1e-6)
    batch = assignment.shape[0]
    endpoint = torch.eye(
        object_slots, device=assignment.device, dtype=adjacent.dtype
    )[None].expand(batch, -1, -1)
    for index in range(adjacent.shape[1]):
        endpoint = torch.bmm(endpoint, adjacent[:, index])
    if not bool(torch.isfinite(endpoint).all()):
        raise RuntimeError("slot trajectory endpoint is non-finite")
    return SlotTrajectory(
        adjacent=adjacent,
        confidence=confidence.clamp(0.0, 1.0),
        endpoint=endpoint,
    )


def align_endpoint_state(
    state: dict[str, torch.Tensor],
    endpoint: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = {}
    for name in (
        "identity",
        "dynamic",
        "center",
        "log_scale",
        "presence",
        "visibility",
    ):
        value = state[name][:, -1]
        result[name] = torch.einsum("bkl,bl...->bk...", endpoint, value)
    return result

