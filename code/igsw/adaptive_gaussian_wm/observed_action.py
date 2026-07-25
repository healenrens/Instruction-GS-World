"""Observable object-level anchors for the continuous latent action."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .readout_runtime import object_rgb_from_micro
from .rgb_supervision import micro_rgb_from_assignment


RGB_LOGIT_ACTION_SCALE = 1.0


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float().clamp(1e-4, 1.0 - 1e-4))


def rgb_logit_action(
    current_object_rgb: torch.Tensor,
    future_object_rgb: torch.Tensor,
) -> torch.Tensor:
    """Encode bounded per-object RGB-logit change as three action channels."""
    if current_object_rgb.ndim != 3 or current_object_rgb.shape[-1] != 3:
        raise ValueError("current object RGB must have shape [B,K,3]")
    if future_object_rgb.ndim != 4 or future_object_rgb.shape[-1] != 3:
        raise ValueError("future object RGB must have shape [B,Q,K,3]")
    if future_object_rgb.shape[:1] + future_object_rgb.shape[2:3] != (
        current_object_rgb.shape[0],
        current_object_rgb.shape[1],
    ):
        raise ValueError("current and future object RGB do not align")
    delta = (
        _stable_logit(future_object_rgb)
        - _stable_logit(current_object_rgb)[:, None]
    )
    return torch.tanh(delta / RGB_LOGIT_ACTION_SCALE)


def observed_object_rgb_targets(
    batch: dict[str, torch.Tensor],
    history: dict,
    target_future: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool observed RGB into aligned online-current and EMA-future objects."""
    required = (
        "history_rgb",
        "history_rgb_valid",
        "future_rgb",
        "future_rgb_valid",
        "feature_grid_hw",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"RGB action anchor requires fields {missing}")
    grid_hw = batch["feature_grid_hw"]
    if grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
        raise ValueError("feature_grid_hw must have shape [B,2]")
    if not bool((grid_hw == grid_hw[:1]).all()):
        raise ValueError("feature grid dimensions differ within the batch")
    grid_height = int(grid_hw[0, 0])
    grid_width = int(grid_hw[0, 1])

    current_tokens = history["token_states"][-1]
    current_slots = history["slot_states"][-1]
    current_micro = micro_rgb_from_assignment(
        current_tokens,
        batch["history_rgb"][:, -1],
        batch["history_rgb_valid"][:, -1],
        grid_height,
        grid_width,
    )
    current_object = object_rgb_from_micro(
        current_micro,
        current_slots.assignment,
        current_tokens.activation,
    )

    future_objects = []
    states = zip(
        target_future["token_states"],
        target_future["slot_states"],
        strict=True,
    )
    for index, (tokens, slots) in enumerate(states):
        micro = micro_rgb_from_assignment(
            tokens,
            batch["future_rgb"][:, index],
            batch["future_rgb_valid"][:, index],
            grid_height,
            grid_width,
        )
        future_objects.append(
            object_rgb_from_micro(
                micro,
                slots.assignment,
                tokens.activation,
            )
        )
    return current_object.detach(), torch.stack(future_objects, dim=1).detach()


def posterior_from_targets(
    model,
    batch: dict[str, torch.Tensor],
    history: dict,
    target_future: dict,
    future_scale: torch.Tensor,
    condition: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Build the teacher action while keeping observed future RGB posterior-only."""
    action_rgb = (
        observed_object_rgb_targets(batch, history, target_future)
        if model.config.rgb_semantic_action
        else (None, None)
    )
    posterior = model.latent_actions.posterior(
        history["slots"],
        history["activity"],
        target_future["slots"],
        target_future["activity"],
        future_scale,
        history["center"],
        target_future["center"],
        condition,
        *action_rgb,
    )
    return posterior, action_rgb


def future_object_rgb_loss(model, output: dict) -> torch.Tensor:
    """Anchor predicted slots to the future object's observed mean RGB."""
    target = output.get("target_future_object_rgb")
    if target is None:
        return output["predicted_future_slots"].sum() * 0.0
    prediction = torch.sigmoid(
        model.object_aggregator.decode_rgb_logits(
            output["predicted_future_slots"]
        )
    )
    error = F.smooth_l1_loss(
        prediction,
        target.detach(),
        reduction="none",
        beta=0.05,
    ).mean(dim=-1)
    weight = output["target_future_activity"].detach().to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)
