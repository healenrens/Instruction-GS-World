"""Shared prediction-error helpers for image-goal evaluations."""
from __future__ import annotations

import torch

from .rgb_supervision import rgb_reconstruction_loss


def history_from_output(output: dict) -> dict[str, torch.Tensor]:
    states = output["history_slot_states"]
    return {
        "slots": output["online_history_slots"],
        "center": output["online_history_centers"],
        "activity": torch.stack([state.activity for state in states], dim=1),
        "token_states": output["history_token_states"],
        "slot_states": states,
    }


def _masked_feature_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    weight = valid.to(error.dtype)
    return (error * weight).flatten(1).sum(dim=1) / (
        weight.flatten(1).sum(dim=1).clamp_min(1.0)
    )


def _weighted_latent_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    error = (
        torch.nn.functional.normalize(prediction, dim=-1)
        - torch.nn.functional.normalize(target, dim=-1)
    ).square().mean(dim=-1)
    weight = activity.to(error.dtype)
    return (error * weight).flatten(1).sum(dim=1) / (
        weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    )


def _rgb_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    return torch.stack(
        [
            rgb_reconstruction_loss(
                prediction[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                ssim_weight,
            )[0]
            for index in range(len(prediction))
        ]
    )


def prediction_errors(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    model,
) -> dict[str, torch.Tensor]:
    return {
        "feature_mse": _masked_feature_mse(
            prediction["feature"].float(),
            batch["future_features"].float(),
            batch["future_valid"],
        ),
        "latent_mse": _weighted_latent_mse(
            prediction["latent"].float(),
            output["target_future_slots"].float(),
            output["target_future_activity"],
        ),
        "rgb_distance": _rgb_distance(
            prediction["rgb"].float(),
            batch["future_rgb"],
            batch["future_rgb_valid"],
            model.config.rgb_ssim_weight,
        ),
    }
