"""Scale-stable feature changes and change-balanced RGB supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .jepa_losses import weighted_mean


def dense_feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    mse = (prediction - target).square().mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    weight = valid_mask.to(prediction.dtype) * (coverage.detach() > 1e-4)
    return weighted_mean(mse + 0.1 * cosine, weight)


def scale_invariant_object_change_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current_prediction: torch.Tensor,
    current_target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    """Compare directional feature changes without rewarding feature-norm growth."""
    predicted_change = F.normalize(prediction.float(), dim=-1) - F.normalize(
        current_prediction[:, None].float(), dim=-1
    )
    target_change = F.normalize(target.detach().float(), dim=-1) - F.normalize(
        current_target.detach()[:, None].float(), dim=-1
    )
    error = F.smooth_l1_loss(
        predicted_change,
        target_change,
        reduction="none",
        beta=0.05,
    )
    return weighted_mean(error, activity.detach())


def _active_weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weight.sum()
    active = (denominator > 0).to(value.dtype)
    return (value * weight).sum() / denominator.clamp_min(1.0), active


def _weighted_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    square = value.square().mean(dim=2)
    mean_square, active = _active_weighted_mean(square, weight)
    return torch.sqrt(mean_square.clamp_min(1e-12)) * active


def change_balanced_rgb_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    future_valid: torch.Tensor,
    current_valid: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balance true temporal-change and static RGB delta errors."""
    if prediction.shape != target.shape:
        raise ValueError("RGB delta prediction and target shapes differ")
    expected_valid = (
        prediction.shape[0],
        prediction.shape[1],
        *prediction.shape[-2:],
    )
    if future_valid.shape != expected_valid:
        raise ValueError("future RGB valid mask shape mismatch")
    if (
        current.shape[:2] != (prediction.shape[0], 1)
        or current.shape[2:] != prediction.shape[2:]
    ):
        raise ValueError("current RGB must contain the final observed frame")
    if current_valid.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("current RGB valid mask shape mismatch")
    if threshold <= 0.0:
        raise ValueError("RGB change threshold must be positive")

    target_float = target.float() / 255.0
    current_float = current.float().expand_as(target) / 255.0
    predicted_delta = prediction.float() - current_float
    target_delta = target_float - current_float
    valid = future_valid & current_valid.expand_as(future_valid)
    valid_weight = valid.float()

    change_score = target_delta.abs().mean(dim=2)
    change_strength = (change_score / threshold).clamp(0.0, 1.0)
    change_weight = change_strength.square() * valid_weight
    static_weight = (1.0 - change_strength).square() * valid_weight
    error_map = torch.sqrt(
        (predicted_delta - target_delta).square() + 1e-6
    ).mean(dim=2)
    change_error, change_active = _active_weighted_mean(error_map, change_weight)
    static_error, static_active = _active_weighted_mean(error_map, static_weight)
    active_regions = change_active + static_active
    total = (
        change_error * change_active + static_error * static_active
    ) / active_regions.clamp_min(1.0)

    copy_error_map = torch.sqrt(target_delta.square() + 1e-6).mean(dim=2)
    copy_change_error, _ = _active_weighted_mean(copy_error_map, change_weight)
    predicted_change_rms = _weighted_rms(predicted_delta, change_weight)
    target_change_rms = _weighted_rms(target_delta, change_weight)
    return total, {
        "loss": total,
        "change_charbonnier": change_error,
        "static_charbonnier": static_error,
        "change_weight_fraction": (
            change_weight.sum() / valid_weight.sum().clamp_min(1.0)
        ),
        "copy_change_charbonnier": copy_change_error,
        "change_gain_over_copy": (copy_change_error - change_error).detach(),
        "predicted_change_rms": predicted_change_rms.detach(),
        "target_change_rms": target_change_rms.detach(),
        "change_energy_ratio": (
            predicted_change_rms / target_change_rms.clamp_min(1e-6)
        ).detach(),
    }
