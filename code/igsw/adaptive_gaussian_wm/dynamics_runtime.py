"""Dispatch legacy and factorized Dynamics without widening old call sites."""
from __future__ import annotations

import torch


def run_object_dynamics(
    model,
    history_slots: torch.Tensor,
    history_activity: torch.Tensor,
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    actions: torch.Tensor,
    history_mask: torch.Tensor,
    history_centers: torch.Tensor,
    condition: torch.Tensor | None,
    *,
    history_relative_scale: torch.Tensor | None = None,
    history_relative_disparity: torch.Tensor | None = None,
    history_relations: torch.Tensor | None = None,
    history_existence: torch.Tensor | None = None,
):
    arguments = (
        history_slots,
        history_activity,
        history_scale,
        future_scale,
        actions,
        history_mask,
        history_centers,
        condition,
    )
    if not model.config.factorized_dynamics:
        return model.dynamics(*arguments)
    return model.dynamics(
        *arguments,
        history_relative_scale=history_relative_scale,
        history_relative_disparity=history_relative_disparity,
        history_relations=history_relations,
        history_existence=history_existence,
    )


def factorized_result_fields(future_output, history_output, target_future) -> dict:
    """Expose optional v28 outputs while keeping legacy results unchanged."""
    names = (
        "future_relative_scale",
        "future_relative_disparity",
        "future_visibility",
        "future_existence",
        "future_relations",
        "base_future_slots",
        "action_slot_residual",
    )
    result = {
        f"predicted_{name}": getattr(future_output, name)
        for name in names
        if hasattr(future_output, name)
    }
    result["target_future_token_states"] = target_future["token_states"]
    result["target_future_slot_states"] = target_future["slot_states"]
    if hasattr(history_output, "future_relative_scale"):
        result.update(
            {
                "zero_action_future_relative_scale": (
                    history_output.future_relative_scale
                ),
                "zero_action_future_relative_disparity": (
                    history_output.future_relative_disparity
                ),
                "zero_action_future_visibility": (
                    history_output.future_visibility
                ),
                "zero_action_future_existence": (
                    history_output.future_existence
                ),
            }
        )
    return result
