"""Scale-relative motion features and transport objectives for object states."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .jepa_losses import weighted_mean


MOTION_FEATURE_DIM = 5


def _validate_history(
    center: torch.Tensor,
    scale: torch.Tensor,
    time: torch.Tensor,
    existence: torch.Tensor,
    visibility: torch.Tensor,
) -> None:
    if center.ndim != 4 or center.shape[-1] != 2:
        raise ValueError("history center must have shape [B,T,K,2]")
    if scale.shape != center.shape[:-1]:
        raise ValueError("history scale must have shape [B,T,K]")
    if time.shape != center.shape[:2]:
        raise ValueError("history time must have shape [B,T]")
    if existence.shape != scale.shape or visibility.shape != scale.shape:
        raise ValueError("history lifecycle tensors must have shape [B,T,K]")


def temporal_motion_features(
    center: torch.Tensor,
    scale: torch.Tensor,
    time: torch.Tensor,
    existence: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Return object-relative and shared scene velocity in support units."""
    _validate_history(center, scale, time, existence, visibility)
    features = center.new_zeros((*center.shape[:-1], MOTION_FEATURE_DIM))
    if center.shape[1] == 1:
        return features
    support = torch.sqrt(
        scale[:, 1:].clamp_min(1e-6) * scale[:, :-1].clamp_min(1e-6)
    )
    delta_time = (time[:, 1:] - time[:, :-1]).abs().clamp_min(1e-3)
    velocity = (center[:, 1:] - center[:, :-1]) / support[..., None]
    velocity = velocity / delta_time[..., None, None]
    valid = (
        torch.minimum(existence[:, 1:], existence[:, :-1])
        * torch.minimum(visibility[:, 1:], visibility[:, :-1])
    ).clamp(0.0, 1.0)
    denominator = valid.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    scene_velocity = (velocity * valid[..., None]).sum(dim=-2) / denominator
    scene_velocity = scene_velocity[:, :, None].expand_as(velocity)
    relative_velocity = velocity - scene_velocity
    motion = torch.cat(
        (relative_velocity, scene_velocity, valid[..., None]),
        dim=-1,
    )
    features[:, 1:] = motion * valid[..., None]
    return features


def support_normalized_displacement(
    source_center: torch.Tensor,
    target_center: torch.Tensor,
    source_scale: torch.Tensor,
) -> torch.Tensor:
    if source_center.shape[-1] != 2 or target_center.shape[-1] != 2:
        raise ValueError("transport centers must end with dimension two")
    return (target_center - source_center) / source_scale.clamp_min(1e-6)[..., None]


def centers_from_support_transport(
    current_center: torch.Tensor,
    current_scale: torch.Tensor,
    transport: torch.Tensor,
) -> torch.Tensor:
    if current_center.shape[-1] != 2 or transport.shape[-1] != 2:
        raise ValueError("transport centers and vectors must end with dimension two")
    return current_center + current_scale[..., None] * transport


def _pairwise_relative_change(
    source: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    source_relation = source.unsqueeze(-2) - source.unsqueeze(-3)
    target_relation = target.unsqueeze(-2) - target.unsqueeze(-3)
    pair_scale = torch.sqrt(
        scale.unsqueeze(-1).clamp_min(1e-6)
        * scale.unsqueeze(-2).clamp_min(1e-6)
    )
    return (target_relation - source_relation) / pair_scale[..., None]


def relative_transport_loss(output: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output["predicted_future_centers"]
    target = output["target_future_centers"].detach()
    online_center = output["online_history_centers"][:, -1, None]
    target_center = output["target_history_centers"][:, -1, None].detach()
    online_scale = output["online_history_relative_scale"][:, -1, None]
    target_scale = output["target_history_relative_scale"][:, -1, None].detach()
    current_visibility = output["target_history_visibility"][:, -1, None].detach()
    future_visibility = output["target_future_visibility"].detach()
    horizon_valid = output["future_horizon_valid"].detach()[..., None]
    weight = current_visibility * future_visibility * horizon_valid

    predicted_transport = support_normalized_displacement(
        online_center,
        prediction,
        online_scale,
    )
    target_transport = support_normalized_displacement(
        target_center,
        target,
        target_scale,
    )
    local = weighted_mean(
        F.smooth_l1_loss(
            predicted_transport,
            target_transport,
            beta=0.1,
            reduction="none",
        ),
        weight,
    )
    predicted_pair = _pairwise_relative_change(
        online_center,
        prediction,
        online_scale,
    )
    target_pair = _pairwise_relative_change(
        target_center,
        target,
        target_scale,
    )
    pair_weight = weight.unsqueeze(-1) * weight.unsqueeze(-2)
    pairwise = weighted_mean(
        F.smooth_l1_loss(
            predicted_pair,
            target_pair,
            beta=0.1,
            reduction="none",
        ),
        pair_weight,
    )
    persistence_local = weighted_mean(
        F.smooth_l1_loss(
            torch.zeros_like(target_transport),
            target_transport,
            beta=0.1,
            reduction="none",
        ),
        weight,
    )
    total = local + 0.5 * pairwise
    return total, {
        "transport_support_normalized": local,
        "transport_pairwise_relative": pairwise,
        "transport_persistence_support_normalized": persistence_local,
        "transport_gain_over_persistence": persistence_local - local,
        "transport_predicted_magnitude": predicted_transport.square().sum(-1).sqrt().mean(),
        "transport_target_magnitude": target_transport.square().sum(-1).sqrt().mean(),
    }
