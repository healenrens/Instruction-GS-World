"""Training-only lifecycle targets derived from point-track observation histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch


LIFECYCLE_VISIBLE = 0
LIFECYCLE_OCCLUDED = 1
LIFECYCLE_UNKNOWN = 2
LIFECYCLE_ABSENT = 3


@dataclass(frozen=True)
class ComponentLifecycle:
    visibility: torch.Tensor
    presence: torch.Tensor
    known: torch.Tensor
    state: torch.Tensor
    track_state: torch.Tensor


def _last_visible_age(visibility: torch.Tensor) -> torch.Tensor:
    frames = visibility.shape[1]
    time = torch.arange(frames, device=visibility.device).view(1, frames, 1)
    observed = torch.where(visibility, time, torch.full_like(time, -frames))
    last = observed.cummax(dim=1).values
    return time - last


def _next_visible_distance(visibility: torch.Tensor) -> torch.Tensor:
    frames = visibility.shape[1]
    time = torch.arange(frames, device=visibility.device).view(1, frames, 1)
    observed = torch.where(visibility, time, torch.full_like(time, 2 * frames))
    next_time = observed.flip(1).cummin(dim=1).values.flip(1)
    return next_time - time


def track_lifecycle_states(visibility: torch.Tensor, config) -> torch.Tensor:
    """Label visible, internal occlusion, unknown gaps and sustained absence."""
    if visibility.ndim != 3 or visibility.dtype != torch.bool:
        raise ValueError("v51 track visibility must be bool [B,T,P]")
    seen_before = visibility.cumsum(dim=1) > 0
    seen_after = visibility.flip(1).cumsum(dim=1).flip(1) > 0
    internal_gap = ~visibility & seen_before & seen_after
    leading_gap = ~visibility & ~seen_before & seen_after
    trailing_gap = ~visibility & seen_before & ~seen_after
    last_age = _last_visible_age(visibility)
    next_distance = _next_visible_distance(visibility)
    state = torch.full_like(visibility, LIFECYCLE_UNKNOWN, dtype=torch.long)
    state = torch.where(visibility, LIFECYCLE_VISIBLE, state)
    state = torch.where(internal_gap, LIFECYCLE_OCCLUDED, state)
    recent_trailing = trailing_gap & (
        last_age <= config.lifecycle_occlusion_grace_frames
    )
    state = torch.where(recent_trailing, LIFECYCLE_OCCLUDED, state)
    absent_trailing = trailing_gap & (
        last_age >= config.lifecycle_absent_gap_frames
    )
    absent_leading = leading_gap & (
        next_distance >= config.lifecycle_absent_gap_frames
    )
    state = torch.where(absent_trailing | absent_leading, LIFECYCLE_ABSENT, state)
    return state


def component_lifecycle_targets(
    visibility: torch.Tensor,
    object_owner: torch.Tensor,
    component_valid: torch.Tensor,
    config,
) -> ComponentLifecycle:
    if object_owner.ndim != 3 or visibility.shape[0] != object_owner.shape[0]:
        raise ValueError("v51 component lifecycle owner shapes differ")
    track_state = track_lifecycle_states(visibility, config)
    component_size = object_owner.sum(dim=1).clamp_min(1.0)
    visible_mass = torch.einsum("btp,bpk->btk", visibility.float(), object_owner)
    visible_fraction = visible_mass / component_size[:, None]
    visible = visible_fraction >= config.lifecycle_visible_track_fraction
    absent_vote = torch.einsum(
        "btp,bpk->btk",
        (track_state == LIFECYCLE_ABSENT).float(),
        object_owner,
    ) / component_size[:, None]
    occluded_vote = torch.einsum(
        "btp,bpk->btk",
        (track_state == LIFECYCLE_OCCLUDED).float(),
        object_owner,
    ) / component_size[:, None]
    valid = component_valid[:, None]
    absent = valid & ~visible & (absent_vote >= config.lifecycle_absent_track_fraction)
    occluded = valid & ~visible & ~absent & (occluded_vote > 0.0)
    state = torch.full_like(visible, LIFECYCLE_UNKNOWN, dtype=torch.long)
    state = torch.where(valid & visible, LIFECYCLE_VISIBLE, state)
    state = torch.where(occluded, LIFECYCLE_OCCLUDED, state)
    state = torch.where(absent, LIFECYCLE_ABSENT, state)
    known = state != LIFECYCLE_UNKNOWN
    presence = ((state == LIFECYCLE_VISIBLE) | (state == LIFECYCLE_OCCLUDED)).float()
    return ComponentLifecycle(
        visibility=visible_fraction,
        presence=presence,
        known=known,
        state=state,
        track_state=track_state,
    )
