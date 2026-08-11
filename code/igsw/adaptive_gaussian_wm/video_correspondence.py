"""Frozen-feature patch transport used as pure-video self-supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoCorrespondence:
    forward: torch.Tensor
    backward: torch.Tensor
    cycle_error: torch.Tensor
    residual_motion: torch.Tensor


def _transport(
    source: torch.Tensor,
    target: torch.Tensor,
    source_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
    target_valid: torch.Tensor,
    temperature: float,
    spatial_sigma: float,
) -> torch.Tensor:
    source = F.normalize(source.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("bnd,bmd->bnm", source, target) / temperature
    distance = (
        (source_coordinates[:, :, None].float() - target_coordinates[:, None].float())
        .square()
        .sum(dim=-1)
    )
    logits = similarity - distance / (2.0 * spatial_sigma**2)
    logits = logits.masked_fill(~target_valid[:, None], -torch.finfo(logits.dtype).max)
    return logits.softmax(dim=-1)


def build_video_correspondence(
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    spatial_sigma: float,
) -> VideoCorrespondence:
    if patches.ndim != 4 or coordinates.shape != (*patches.shape[:3], 2):
        raise ValueError(
            "correspondence expects [B,T,N,D] patches and [B,T,N,2] coordinates"
        )
    if valid.shape != patches.shape[:3] or patches.shape[1] < 2:
        raise ValueError("correspondence validity or temporal length differs")
    if temperature <= 0.0 or spatial_sigma <= 0.0:
        raise ValueError(
            "correspondence temperature and spatial sigma must be positive"
        )
    forward, backward, cycle_error, residual_motion = [], [], [], []
    for time in range(patches.shape[1] - 1):
        fwd = _transport(
            patches[:, time],
            patches[:, time + 1],
            coordinates[:, time],
            coordinates[:, time + 1],
            valid[:, time + 1],
            temperature,
            spatial_sigma,
        )
        bwd = _transport(
            patches[:, time + 1],
            patches[:, time],
            coordinates[:, time + 1],
            coordinates[:, time],
            valid[:, time],
            temperature,
            spatial_sigma,
        )
        cycle_probability = torch.einsum("bij,bji->bi", fwd, bwd).clamp(0.0, 1.0)
        transported_coordinates = torch.einsum(
            "bij,bjd->bid", fwd, coordinates[:, time + 1].float()
        )
        flow = transported_coordinates - coordinates[:, time].float()
        weight = valid[:, time].float()
        global_flow = (flow * weight[..., None]).sum(dim=1, keepdim=True) / (
            weight.sum(dim=1, keepdim=True).clamp_min(1.0)[..., None]
        )
        forward.append(fwd)
        backward.append(bwd)
        cycle_error.append((1.0 - cycle_probability) * weight)
        residual_motion.append((flow - global_flow).norm(dim=-1) * weight)
    return VideoCorrespondence(
        forward=torch.stack(forward, dim=1),
        backward=torch.stack(backward, dim=1),
        cycle_error=torch.stack(cycle_error, dim=1),
        residual_motion=torch.stack(residual_motion, dim=1),
    )
