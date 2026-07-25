"""Object-level JEPA alignment objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    expanded = weight.to(value.dtype)
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    denominator = expanded.expand_as(value).sum().clamp_min(1.0)
    return (value * expanded).sum() / denominator


def object_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return weighted_mean(
        (prediction - target).square(),
        activity.detach(),
    )


def object_change_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current_prediction: torch.Tensor,
    current_target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    predicted_change = prediction - current_prediction[:, None]
    target_change = target.detach() - current_target.detach()[:, None]
    error = F.smooth_l1_loss(
        predicted_change,
        target_change,
        reduction="none",
        beta=0.05,
    )
    return weighted_mean(error, activity.detach())


def masked_history_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history_mask: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = history_mask.to(activity.dtype) * activity.detach()
    if not bool(history_mask.any()):
        return prediction.sum() * 0.0
    return object_latent_loss(prediction, target, weight)
