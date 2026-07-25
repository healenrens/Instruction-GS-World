"""Metrics and causal checks for the DINO+RGB matched flat baseline."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .goal_eval_statistics import clustered_paired_comparison
from .matched_flat_world_model import MatchedFlatLatentWorldModel
from .temporal_region_evaluation import (
    TemporalRegionConfig,
    regional_rgb_error_frames,
    temporal_region_masks,
)


VARIANTS = ("object_posterior", "flat_posterior", "flat_history", "copy")
COMPARISONS = (
    ("flat_history_over_copy", "flat_history", "copy"),
    ("flat_posterior_over_history", "flat_posterior", "flat_history"),
    ("flat_posterior_over_copy", "flat_posterior", "copy"),
    ("object_over_flat_posterior", "object_posterior", "flat_posterior"),
    ("object_over_flat_history", "object_posterior", "flat_history"),
    ("object_over_copy", "object_posterior", "copy"),
)
RGB_REGIONS = ("change", "static")


def _masked_mean_by_query(
    value: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    weight = valid.to(value.dtype)
    return (value * weight).sum(dim=(-2, -1)) / weight.sum(
        dim=(-2, -1)
    ).clamp_min(1.0)


def feature_mse_by_query(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    weight = valid.to(error.dtype)
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def change_feature_mse_by_query(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = batch["future_features"].float()
    current = batch["history_features"][:, -1:].float()
    error = (prediction.float() - target).square().mean(dim=-1)
    change = (target - current).square().mean(dim=-1).sqrt()
    weight = change * batch["future_valid"].to(change.dtype)
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-6)


def rgb_distance_by_query(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("RGB evaluation prediction and target shapes differ")
    if valid.shape != (prediction.shape[0], prediction.shape[1], *prediction.shape[-2:]):
        raise ValueError("RGB evaluation valid mask shape differs")
    target_float = target.float() / 255.0
    error = prediction.float() - target_float
    charbonnier = _masked_mean_by_query(
        torch.sqrt(error.square() + 1e-6).mean(dim=2),
        valid,
    )
    flat_prediction = prediction.flatten(0, 1).float()
    flat_target = target_float.flatten(0, 1)
    mu_prediction = F.avg_pool2d(flat_prediction, 3, 1, 1)
    mu_target = F.avg_pool2d(flat_target, 3, 1, 1)
    variance_prediction = F.avg_pool2d(flat_prediction.square(), 3, 1, 1)
    variance_prediction = variance_prediction - mu_prediction.square()
    variance_target = F.avg_pool2d(flat_target.square(), 3, 1, 1)
    variance_target = variance_target - mu_target.square()
    covariance = F.avg_pool2d(flat_prediction * flat_target, 3, 1, 1)
    covariance = covariance - mu_prediction * mu_target
    ssim = (
        (2.0 * mu_prediction * mu_target + 0.01**2)
        * (2.0 * covariance + 0.03**2)
        / (
            (mu_prediction.square() + mu_target.square() + 0.01**2)
            * (variance_prediction + variance_target + 0.03**2)
        ).clamp_min(1e-6)
    ).mean(dim=1).clamp(-1.0, 1.0).reshape_as(valid)
    flat_valid = valid.flatten(0, 1).float()[:, None]
    ssim_valid = F.avg_pool2d(flat_valid, 3, 1, 1)
    ssim_valid = (ssim_valid[:, 0] >= 1.0 - 1e-6).reshape_as(valid)
    ssim_loss = _masked_mean_by_query(1.0 - ssim, ssim_valid)
    return charbonnier + ssim_weight * ssim_loss


def change_rgb_distance_by_query(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = batch["future_rgb"].float() / 255.0
    current = batch["history_rgb"][:, -1:].float() / 255.0
    current = current.expand_as(target)
    error = torch.sqrt((prediction.float() - target).square() + 1e-6).mean(dim=2)
    change = (target - current).abs().mean(dim=2)
    valid = batch["future_rgb_valid"] & batch["history_rgb_valid"][:, -1:].expand_as(
        batch["future_rgb_valid"]
    )
    weight = change * valid.to(change.dtype)
    return (error * weight).sum(dim=(-2, -1)) / weight.sum(
        dim=(-2, -1)
    ).clamp_min(1e-6)


def rgb_region_errors_by_query(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: TemporalRegionConfig,
) -> dict[str, dict[str, torch.Tensor]]:
    """Measure variants on robust GT-only temporal change/static masks."""
    regions = temporal_region_masks(
        batch["history_rgb"],
        batch["future_rgb"],
        batch["history_rgb_valid"],
        batch["future_rgb_valid"],
        config,
    )
    valid_pixels = regions["valid"].flatten(2).sum(dim=-1).clamp_min(1)
    result = {}
    for region in RGB_REGIONS:
        mask = regions[region]
        nonempty = mask.flatten(2).any(dim=-1)
        errors = {
            name: regional_rgb_error_frames(
                prediction,
                batch["future_rgb"],
                mask,
            )["charbonnier"]
            for name, prediction in predictions.items()
        }
        result[region] = {
            "nonempty": nonempty,
            "coverage": mask.flatten(2).sum(dim=-1).float()
            / valid_pixels.float(),
            **errors,
        }
    return result


def clustered_comparisons(
    variants: dict[str, torch.Tensor],
    clusters: torch.Tensor,
) -> dict[str, dict]:
    return {
        name: clustered_paired_comparison(
            variants[prediction],
            variants[reference],
            clusters,
        )
        for name, prediction, reference in COMPARISONS
    }


def causal_probe(
    flat: MatchedFlatLatentWorldModel,
    batch: dict[str, torch.Tensor],
    history_rgb: torch.Tensor,
    future_rgb: torch.Tensor,
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
) -> dict[str, float | bool]:
    if len(batch["future_features"]) < 2:
        raise ValueError("flat causal probe requires a batch of at least two")
    order = torch.arange(
        len(batch["future_features"]),
        device=future_scale.device,
    ).roll(1)
    arguments = (
        batch["history_features"],
        batch["history_coordinates"],
        history_scale,
        history_rgb,
    )
    base = flat(
        *arguments,
        batch["future_features"],
        batch["future_coordinates"],
        future_scale,
        future_rgb,
    )
    swapped = flat(
        *arguments,
        batch["future_features"][order],
        batch["future_coordinates"],
        future_scale,
        future_rgb[order],
    )
    fixed_feature, fixed_rgb = flat.dynamics_readout(
        base["history_state"],
        batch["history_features"][:, -1],
        history_rgb[:, -1],
        batch["future_coordinates"],
        future_scale,
        base["posterior_actions"],
    )
    action_difference = (
        base["posterior_actions"] - swapped["posterior_actions"]
    ).float().square().mean().sqrt()
    history_feature_difference = (
        base["history_feature_prediction"]
        - swapped["history_feature_prediction"]
    ).float().abs().max()
    history_rgb_difference = (
        base["history_rgb_grid_prediction"]
        - swapped["history_rgb_grid_prediction"]
    ).float().abs().max()
    fixed_feature_difference = (
        base["posterior_feature_prediction"] - fixed_feature
    ).float().abs().max()
    fixed_rgb_difference = (
        base["posterior_rgb_grid_prediction"] - fixed_rgb
    ).float().abs().max()
    direct_future_difference = torch.maximum(
        fixed_feature_difference,
        fixed_rgb_difference,
    )
    history_difference = torch.maximum(
        history_feature_difference,
        history_rgb_difference,
    )
    return {
        "future_swap_action_rms_difference": float(action_difference),
        "history_future_swap_max_difference": float(history_difference),
        "fixed_action_prediction_max_difference": float(direct_future_difference),
        "fixed_action_feature_max_difference": float(fixed_feature_difference),
        "fixed_action_rgb_grid_max_difference": float(fixed_rgb_difference),
        "posterior_responds_to_future": bool(action_difference > 1e-6),
        "history_has_no_future_input": bool(history_difference <= 1e-6),
        "dynamics_has_no_direct_future_input": bool(direct_future_difference <= 1e-6),
    }
