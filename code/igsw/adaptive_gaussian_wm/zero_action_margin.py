"""Direct observed-space margins between posterior and zero-action predictions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .readout_runtime import decode_gaussian_readout, residual_future_features
from .rgb_supervision import (
    render_future_rgb,
    residual_future_rgb,
    rgb_reconstruction_loss,
)


def relative_margin_ranking(
    matched: torch.Tensor,
    reference: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Push matched errors below a detached reference by a relative margin."""
    if matched.shape != reference.shape:
        raise ValueError("matched and reference errors must align")
    if margin <= 0.0:
        raise ValueError("relative margin must be positive")
    detached = reference.detach()
    relative_delta = (matched - detached) / detached.clamp_min(1e-6)
    return F.relu(relative_delta + margin)


def _weighted_per_sample(
    error: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    expanded = weight.to(error.dtype)
    while expanded.ndim < error.ndim:
        expanded = expanded.unsqueeze(-1)
    numerator = (error * expanded).flatten(1).sum(dim=1)
    denominator = expanded.expand_as(error).flatten(1).sum(dim=1)
    return numerator / denominator.clamp_min(1e-6)


def _feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return _weighted_per_sample(
        (prediction.float() - target.float()).square().mean(dim=-1),
        valid,
    )


def _latent_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    return _weighted_per_sample(
        (
            F.normalize(prediction.float(), dim=-1)
            - F.normalize(target.float(), dim=-1)
        ).square(),
        activity,
    )


def _rgb_error(
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


def _render_zero_action(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        readout, _ = decode_gaussian_readout(
            model,
            batch,
            output["history_token_states"][-1],
            output["history_slot_states"][-1],
            output["zero_action_future_slots"],
            output["zero_action_future_centers"],
        )
        direct_feature = model.gaussian_readout.splat_features(
            readout,
            batch["future_coordinates"],
        )[0]
        feature = residual_future_features(
            direct_feature,
            output["residual_reference_features"],
            batch,
        )
        direct_rgb = render_future_rgb(
            readout,
            batch,
            model.config.rgb_render_chunk,
        )[0]
        if output["residual_reference_rgb"] is None:
            raise ValueError("zero-action RGB margin requires an RGB reference")
        rgb = residual_future_rgb(
            direct_rgb,
            output["residual_reference_rgb"],
            batch,
        )
    return feature, rgb


def observed_zero_action_margin_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank posterior above zero action in latent, dense feature, and RGB space."""
    zero = output["predicted_future_slots"].sum() * 0.0
    if model.config.zero_action_margin_weight == 0.0:
        return zero, {}
    if output["rendered_future_rgb"] is None:
        raise ValueError("observed zero-action margin requires RGB supervision")
    zero_feature, zero_rgb = _render_zero_action(model, batch, output)
    matched = {
        "feature": _feature_error(
            output["rendered_future_features"],
            batch["future_features"],
            batch["future_valid"],
        ),
        "latent": _latent_error(
            output["predicted_future_slots"],
            output["target_future_slots"].detach(),
            output["target_future_activity"].detach(),
        ),
        "rgb": _rgb_error(
            output["rendered_future_rgb"],
            batch["future_rgb"],
            batch["future_rgb_valid"],
            model.config.rgb_ssim_weight,
        ),
    }
    reference = {
        "feature": _feature_error(
            zero_feature,
            batch["future_features"],
            batch["future_valid"],
        ),
        "latent": _latent_error(
            output["zero_action_future_slots"],
            output["target_future_slots"].detach(),
            output["target_future_activity"].detach(),
        ),
        "rgb": _rgb_error(
            zero_rgb,
            batch["future_rgb"],
            batch["future_rgb_valid"],
            model.config.rgb_ssim_weight,
        ),
    }
    rankings = {
        name: relative_margin_ranking(
            matched[name],
            reference[name],
            model.config.zero_action_relative_margin,
        )
        for name in matched
    }
    total = torch.stack([value.mean() for value in rankings.values()]).mean()
    parts = {}
    for name in matched:
        parts[f"zero_margin_{name}"] = rankings[name].mean()
        parts[f"zero_improvement_{name}"] = (
            reference[name] - matched[name]
        ).detach().mean()
    return total, parts
