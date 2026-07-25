"""Legacy RGB objective kept outside the feature-only v28 core."""
from __future__ import annotations

import torch

from .change_objectives import change_balanced_rgb_delta_loss
from .observed_action import future_object_rgb_loss
from .rgb_supervision import rgb_reconstruction_loss


def rgb_loss_bundle(model, batch: dict, output: dict, reference: torch.Tensor):
    rgb = reference * 0.0
    rgb_delta = reference * 0.0
    parts: dict[str, torch.Tensor] = {}
    if model.config.rgb_supervision:
        if output["rendered_future_rgb"] is None:
            raise ValueError("RGB supervision requires a future render")
        future, future_parts = rgb_reconstruction_loss(
            output["rendered_future_rgb"],
            batch["future_rgb"],
            batch["future_rgb_valid"],
            model.config.rgb_ssim_weight,
        )
        if output["rendered_current_rgb"] is None:
            rgb = future
            parts = {f"future_{name}": value for name, value in future_parts.items()}
        else:
            current, current_parts = rgb_reconstruction_loss(
                output["rendered_current_rgb"],
                batch["history_rgb"][:, -1:],
                batch["history_rgb_valid"][:, -1:],
                model.config.rgb_ssim_weight,
            )
            rgb = 0.5 * (current + future)
            parts = {
                f"current_{name}": value for name, value in current_parts.items()
            }
            parts.update(
                {f"future_{name}": value for name, value in future_parts.items()}
            )
        rgb_delta, delta_parts = change_balanced_rgb_delta_loss(
            output["rendered_future_rgb"],
            batch["future_rgb"],
            batch["history_rgb"][:, -1:],
            batch["future_rgb_valid"],
            batch["history_rgb_valid"][:, -1:],
            model.config.rgb_change_threshold,
        )
        rgb = rgb + model.config.rgb_change_loss_weight * rgb_delta
        parts.update({f"delta_{name}": value for name, value in delta_parts.items()})
    object_rgb = future_object_rgb_loss(model, output)
    return rgb + object_rgb, rgb_delta, object_rgb, parts
