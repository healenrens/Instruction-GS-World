"""Continuous change targets for calibrated object transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .query_transition_target_v59 import (
    QueryTransitionTarget,
    build_query_transition_target_v59,
)


@dataclass(frozen=True)
class GatedQueryTransitionTarget(QueryTransitionTarget):
    change_distance: torch.Tensor
    change_strength: torch.Tensor


def _teacher_change_distance(target: QueryTransitionTarget, config) -> torch.Tensor:
    semantic = 1.0 - torch.einsum(
        "bd,bkd->bk", target.source_semantic.float(), target.future_semantic.float()
    )
    geometry_scale = target.future_geometry.new_tensor((2.0, 2.0, 8.0, 8.0, 8.0))
    source_geometry = target.source_geometry[:, None].expand_as(target.future_geometry)
    geometry = F.smooth_l1_loss(
        source_geometry.float() * geometry_scale,
        target.future_geometry.float() * geometry_scale,
        reduction="none",
    ).mean(dim=-1)
    lifecycle = (
        target.future_visibility.float() - target.source_visibility[:, None].float()
    ).abs()
    return (
        config.semantic_loss_weight * semantic
        + config.geometry_loss_weight * geometry
        + config.lifecycle_loss_weight * lifecycle
    )


def build_query_transition_target_v60(
    evidence,
    relation,
    binding,
    frame_times: torch.Tensor,
    observed_frames: int,
    config,
) -> GatedQueryTransitionTarget:
    target = build_query_transition_target_v59(
        evidence,
        relation,
        binding,
        frame_times,
        observed_frames,
        config,
    )
    distance = _teacher_change_distance(target, config)
    excess = F.relu(distance - config.change_noise_floor)
    strength = 1.0 - torch.exp(-excess / config.change_scale)
    strength = strength * target.pair_valid.float()
    return GatedQueryTransitionTarget(
        **target.__dict__,
        change_distance=distance.detach(),
        change_strength=strength.detach(),
    )
