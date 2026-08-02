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


def factorized_result_fields(
    future_output,
    history_output,
    target_future,
    target_history=None,
) -> dict:
    """Expose optional v28 outputs while keeping legacy results unchanged."""
    names = (
        "future_relative_scale",
        "future_relative_disparity",
        "future_visibility_logits",
        "future_existence_logits",
        "future_visibility",
        "future_existence",
        "future_relations",
        "base_future_slots",
        "action_slot_residual",
        "future_transport_units",
        "history_motion_features",
        "future_survival_logits",
        "future_birth_logits",
        "future_observability_logits",
        "future_survival",
        "future_birth",
        "future_observability",
        "future_in_frame",
    )
    result = {
        f"predicted_{name}": getattr(future_output, name)
        for name in names
        if hasattr(future_output, name) and getattr(future_output, name) is not None
    }
    result["target_future_token_states"] = target_future["token_states"]
    result["target_future_slot_states"] = target_future["slot_states"]
    if target_history is not None:
        result["target_history_token_states"] = target_history["token_states"]
        result["target_history_slot_states"] = target_history["slot_states"]
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
        for name in (
            "future_survival",
            "future_birth",
            "future_observability",
            "future_transport_units",
        ):
            value = getattr(history_output, name, None)
            if value is not None:
                result[f"zero_action_{name}"] = value
    return result
