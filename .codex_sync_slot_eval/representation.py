"""Shared current-frame reconstruction path for representation training and evaluation."""
from __future__ import annotations

import torch

from .rgb_supervision import current_micro_rgb, render_current_rgb


def reconstruct_current(
    model,
    batch: dict[str, torch.Tensor],
    history: dict | None = None,
    slots_override: torch.Tensor | None = None,
) -> dict:
    """Decode the latest causal history frame without reading future fields."""
    encoded = model.encode_history(batch) if history is None else history
    current_tokens = encoded["token_states"][-1]
    current_slots = encoded["slot_states"][-1]
    slot_values = (
        current_slots.slots
        if slots_override is None
        else slots_override
    )
    if slot_values.shape != current_slots.slots.shape:
        raise ValueError("slots_override must match current object slots")
    micro_rgb = (
        current_micro_rgb(current_tokens, batch)
        if model.config.rgb_supervision
        else None
    )
    readout = model.gaussian_readout(
        slot_values[:, None],
        current_tokens,
        current_slots.assignment,
        current_slots.center[:, None],
        current_slots.center,
        micro_rgb,
    )
    feature, feature_coverage = model.gaussian_readout.splat_features(
        readout,
        batch["history_coordinates"][:, -1, None],
    )
    rgb = None
    rgb_coverage = None
    if model.config.rgb_supervision:
        rgb, rgb_coverage = render_current_rgb(
            readout,
            batch,
            model.config.rgb_render_chunk,
        )
    return {
        "history": encoded,
        "tokens": current_tokens,
        "slots": current_slots,
        "readout": readout,
        "feature": feature,
        "feature_coverage": feature_coverage,
        "rgb": rgb,
        "rgb_coverage": rgb_coverage,
    }
