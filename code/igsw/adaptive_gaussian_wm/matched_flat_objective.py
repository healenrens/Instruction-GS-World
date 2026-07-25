"""Training objective for the DINO+RGB matched flat baseline."""
from __future__ import annotations

import torch

from .change_objectives import change_balanced_rgb_delta_loss
from .matched_flat_rgb import feature_grid_shape, render_rgb_grid
from .metrics import masked_feature_mse
from .rgb_supervision import rgb_reconstruction_loss


FLAT_TRAIN_METRICS = (
    "posterior_feature_mse",
    "posterior_change_feature_mse",
    "history_feature_mse",
    "history_change_feature_mse",
    "copy_feature_mse",
    "posterior_rgb_distance",
    "posterior_change_rgb",
    "posterior_rgb_change_gain_over_copy",
    "posterior_rgb_change_energy_ratio",
    "history_rgb_distance",
    "history_change_rgb",
    "history_rgb_change_gain_over_copy",
    "history_rgb_change_energy_ratio",
    "copy_rgb_distance",
    "copy_change_rgb",
    "posterior_action_rms",
    "loss",
)


def change_weighted_feature_mse(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = batch["future_features"].float()
    current = batch["history_features"][:, -1:].float()
    error = (prediction.float() - target).square().mean(dim=-1)
    change = (target - current).square().mean(dim=-1).sqrt()
    weight = change * batch["future_valid"].to(change.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)


def _feature_terms(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        masked_feature_mse(
            prediction.float(),
            batch["future_features"].float(),
            batch["future_valid"],
        ),
        change_weighted_feature_mse(prediction, batch),
    )


def _rgb_terms(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    ssim_weight: float,
    change_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    distance, _ = rgb_reconstruction_loss(
        prediction,
        batch["future_rgb"],
        batch["future_rgb_valid"],
        ssim_weight,
    )
    change, details = change_balanced_rgb_delta_loss(
        prediction,
        batch["future_rgb"],
        batch["history_rgb"][:, -1:],
        batch["future_rgb_valid"],
        batch["history_rgb_valid"][:, -1:],
        change_threshold,
    )
    return distance, change, details


def flat_training_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise both posterior and zero-action paths in DINO and RGB space."""
    grid_height, grid_width = feature_grid_shape(batch)
    posterior_rgb = render_rgb_grid(
        output["posterior_rgb_grid_prediction"],
        batch["future_rgb_valid"],
        grid_height,
        grid_width,
    )
    history_rgb = render_rgb_grid(
        output["history_rgb_grid_prediction"],
        batch["future_rgb_valid"],
        grid_height,
        grid_width,
    )
    copy_feature_prediction = batch["history_features"][:, -1:].expand_as(
        output["posterior_feature_prediction"]
    )
    copy_rgb_prediction = (
        batch["history_rgb"][:, -1:].float() / 255.0
    ).expand_as(posterior_rgb)

    posterior_feature, posterior_feature_change = _feature_terms(
        output["posterior_feature_prediction"],
        batch,
    )
    history_feature, history_feature_change = _feature_terms(
        output["history_feature_prediction"],
        batch,
    )
    copy_feature, _ = _feature_terms(copy_feature_prediction, batch)
    posterior_rgb_distance, posterior_rgb_change, posterior_rgb_details = _rgb_terms(
        posterior_rgb,
        batch,
        args.rgb_ssim_weight,
        args.rgb_change_threshold,
    )
    history_rgb_distance, history_rgb_change, history_rgb_details = _rgb_terms(
        history_rgb,
        batch,
        args.rgb_ssim_weight,
        args.rgb_change_threshold,
    )
    copy_rgb_distance, copy_rgb_change, _ = _rgb_terms(
        copy_rgb_prediction,
        batch,
        args.rgb_ssim_weight,
        args.rgb_change_threshold,
    )

    posterior_feature_loss = (
        posterior_feature + args.change_loss_weight * posterior_feature_change
    )
    history_feature_loss = (
        history_feature + args.change_loss_weight * history_feature_change
    )
    posterior_rgb_loss = (
        posterior_rgb_distance
        + args.rgb_change_loss_weight * posterior_rgb_change
    )
    history_rgb_loss = (
        history_rgb_distance + args.rgb_change_loss_weight * history_rgb_change
    )
    loss = (
        posterior_feature_loss
        + args.history_loss_weight * history_feature_loss
        + args.rgb_loss_weight
        * (posterior_rgb_loss + args.history_loss_weight * history_rgb_loss)
    )
    metrics = {
        "posterior_feature_mse": posterior_feature,
        "posterior_change_feature_mse": posterior_feature_change,
        "history_feature_mse": history_feature,
        "history_change_feature_mse": history_feature_change,
        "copy_feature_mse": copy_feature,
        "posterior_rgb_distance": posterior_rgb_distance,
        "posterior_change_rgb": posterior_rgb_change,
        "posterior_rgb_change_gain_over_copy": posterior_rgb_details[
            "change_gain_over_copy"
        ],
        "posterior_rgb_change_energy_ratio": posterior_rgb_details[
            "change_energy_ratio"
        ],
        "history_rgb_distance": history_rgb_distance,
        "history_change_rgb": history_rgb_change,
        "history_rgb_change_gain_over_copy": history_rgb_details[
            "change_gain_over_copy"
        ],
        "history_rgb_change_energy_ratio": history_rgb_details[
            "change_energy_ratio"
        ],
        "copy_rgb_distance": copy_rgb_distance,
        "copy_change_rgb": copy_rgb_change,
        "posterior_action_rms": output["posterior_actions"].float().square().mean().sqrt(),
        "loss": loss,
    }
    if tuple(metrics) != FLAT_TRAIN_METRICS:
        raise RuntimeError("matched flat training metric contract differs")
    return loss, metrics
