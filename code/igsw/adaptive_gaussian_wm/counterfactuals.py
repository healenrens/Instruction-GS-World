"""Shared zero-action and shuffled-action counterfactual paths."""
from __future__ import annotations

import torch

from .readout_runtime import decode_gaussian_readout, residual_future_features
from .rgb_supervision import residual_future_rgb, render_future_rgb
from .scale import signed_gap_scale


def render_state(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    slots: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    tokens = output["history_token_states"][-1]
    current_slots = output["history_slot_states"][-1]
    readout, _ = decode_gaussian_readout(
        model,
        batch,
        tokens,
        current_slots,
        slots,
        centers,
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
    direct_rgb = (
        render_future_rgb(readout, batch, model.config.rgb_render_chunk)[0]
        if model.config.rgb_supervision
        else None
    )
    rgb = (
        residual_future_rgb(
            direct_rgb,
            output["residual_reference_rgb"],
            batch,
        )
        if direct_rgb is not None
        else None
    )
    return feature, rgb


def predict_shuffled_action(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    shuffled_actions: torch.Tensor | None = None,
    condition_override: torch.Tensor | None = None,
    use_dynamics_condition: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = (
        output["posterior_actions"].roll(1, dims=0)
        if shuffled_actions is None
        else shuffled_actions
    )
    if actions.shape != output["posterior_actions"].shape:
        raise ValueError("shuffled actions must match posterior action shape")
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    )
    if use_dynamics_condition:
        condition = (
            output.get("language_condition")
            if condition_override is None
            else condition_override
        )
    else:
        condition = None
    prediction = model.dynamics(
        output["online_history_slots"],
        history_activity,
        signed_gap_scale(batch["history_times"], model.config.gap_reference),
        signed_gap_scale(batch["future_times"], model.config.gap_reference),
        actions,
        output["history_mask"],
        output["online_history_centers"],
        condition,
    )
    centers = (
        prediction.future_centers
        if prediction.future_centers is not None
        else model.object_aggregator.decode_center(prediction.future_slots)
    )
    return prediction.future_slots, centers
