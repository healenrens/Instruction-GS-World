"""Bounded action embeddings that preserve action magnitude."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gate_canonical_center(
    actions: torch.Tensor,
    gate: float,
) -> torch.Tensor:
    """Attenuate uncertain center transport while preserving semantic channels."""
    if actions.shape[-1] < 3:
        raise ValueError("canonical center gating requires at least 3 dimensions")
    return torch.cat((gate * actions[..., :3], actions[..., 3:]), dim=-1)


def residual_action_dropout(
    actions: torch.Tensor,
    probability: float,
    training: bool,
    canonical_dim: int = 6,
) -> torch.Tensor:
    """Remove complete per-object residuals without rescaling kept values."""
    if not training or probability == 0.0:
        return actions
    if actions.shape[-1] <= canonical_dim:
        raise ValueError("residual dropout requires residual action dimensions")
    residual = actions[..., canonical_dim:]
    keep = torch.rand(
        (*residual.shape[:-1], 1),
        device=residual.device,
    ) >= probability
    return torch.cat((actions[..., :canonical_dim], residual * keep), dim=-1)


def effect_supervision_actions(
    actions: torch.Tensor,
    canonical_dim: int,
) -> torch.Tensor:
    """Match the action layout seen by the auxiliary effect heads."""
    if not 0 <= canonical_dim <= actions.shape[-1]:
        raise ValueError("canonical_dim must fit the action dimension")
    if canonical_dim in (0, actions.shape[-1]):
        return actions
    return torch.cat(
        (
            actions[..., :canonical_dim],
            torch.zeros_like(actions[..., canonical_dim:]),
        ),
        dim=-1,
    )


def bounded_action_embedding(
    projection,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Project direction while preventing learned gain from bypassing a gate."""
    projected = projection(actions)
    magnitude = (
        torch.linalg.vector_norm(actions.float(), dim=-1, keepdim=True)
        / math.sqrt(actions.shape[-1])
    ).clamp(max=1.0)
    direction = F.normalize(projected.float(), dim=-1).to(projected.dtype)
    return direction * math.sqrt(projected.shape[-1]) * magnitude.to(
        projected.dtype
    )
