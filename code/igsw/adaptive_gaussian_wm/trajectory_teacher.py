"""Frozen-DINO trajectory evidence derived only from raw video frames."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .video_correspondence import build_video_correspondence


@dataclass(frozen=True)
class TrajectoryEvidence:
    forward: torch.Tensor
    backward: torch.Tensor
    confidence: torch.Tensor
    residual_flow: torch.Tensor
    motion_salience: torch.Tensor


@torch.no_grad()
def build_trajectory_evidence(
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    spatial_sigma: float,
    confidence_floor: float,
) -> TrajectoryEvidence:
    correspondence = build_video_correspondence(
        patches,
        coordinates,
        valid,
        temperature,
        spatial_sigma,
    )
    transported = torch.einsum(
        "btij,btjd->btid",
        correspondence.forward.float(),
        coordinates[:, 1:].float(),
    )
    flow = transported - coordinates[:, :-1].float()
    source_valid = valid[:, :-1].float()
    mean_flow = (flow * source_valid[..., None]).sum(dim=2, keepdim=True)
    mean_flow = mean_flow / source_valid.sum(dim=2, keepdim=True).clamp_min(1.0)[..., None]
    residual_flow = flow - mean_flow
    residual_motion = residual_flow.norm(dim=-1)
    scale = torch.quantile(
        residual_motion.masked_fill(~valid[:, :-1], 0.0),
        0.75,
        dim=2,
        keepdim=True,
    ).clamp_min(0.02)
    motion_salience = (residual_motion / scale).clamp(0.0, 1.0) * source_valid
    cycle_confidence = 1.0 - correspondence.cycle_error.float().clamp(0.0, 1.0)
    peak = correspondence.forward.float().amax(dim=-1)
    confidence = cycle_confidence * peak * source_valid
    confidence = torch.where(
        confidence >= confidence_floor,
        confidence,
        torch.zeros_like(confidence),
    )
    tensors = (
        correspondence.forward,
        correspondence.backward,
        confidence,
        residual_flow,
        motion_salience,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("trajectory teacher produced non-finite evidence")
    return TrajectoryEvidence(
        forward=correspondence.forward.detach(),
        backward=correspondence.backward.detach(),
        confidence=confidence.detach(),
        residual_flow=residual_flow.detach(),
        motion_salience=motion_salience.detach(),
    )


def transported_assignment(
    transport: torch.Tensor,
    target_assignment: torch.Tensor,
) -> torch.Tensor:
    aligned = torch.einsum(
        "btij,btjo->btio",
        transport.float(),
        target_assignment.float(),
    )
    return F.normalize(aligned.clamp_min(0.0), p=1, dim=-1, eps=1e-6)
