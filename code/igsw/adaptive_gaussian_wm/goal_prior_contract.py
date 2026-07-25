"""Causal contract checks for history, physical time, and explicit image goals."""
from __future__ import annotations

import torch

from .goal_conditioning import (
    build_goal_prior_context,
    encode_explicit_goal,
    goal_scale_from_batch,
)
from .observed_action import posterior_from_targets
from .scale import signed_gap_scale


def goal_prior_context(
    model,
    conditioner,
    batch: dict[str, torch.Tensor],
    history: dict[str, torch.Tensor],
    goal: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    return build_goal_prior_context(
        model,
        conditioner,
        history,
        goal,
        history_scale,
        future_scale,
        goal_scale_from_batch(model, batch),
    )


def _roll_fields(
    batch: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    for name in names:
        if name in result:
            result[name] = result[name].roll(1, dims=0)
    return result


@torch.no_grad()
def causal_goal_prior_contract(
    model,
    conditioner,
    batch: dict[str, torch.Tensor],
) -> dict:
    """Prove future supervision is isolated while goal and time remain visible."""
    history = model.encode_history(batch)
    _, target = model.encode_targets(batch)
    goal = encode_explicit_goal(model, batch, history)
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    posterior = posterior_from_targets(
        model,
        batch,
        history,
        target,
        future_scale,
        condition=None,
    )[0]
    context, _ = goal_prior_context(
        model,
        conditioner,
        batch,
        history,
        goal,
    )

    future_swap = _roll_fields(
        batch,
        (
            "future_features",
            "future_coordinates",
            "future_valid",
            "future_rgb",
            "future_rgb_valid",
        ),
    )
    swapped_history = model.encode_history(future_swap)
    _, swapped_target = model.encode_targets(future_swap)
    swapped_goal = encode_explicit_goal(
        model,
        future_swap,
        swapped_history,
    )
    swapped_posterior = posterior_from_targets(
        model,
        future_swap,
        swapped_history,
        swapped_target,
        future_scale,
        condition=None,
    )[0]
    swapped_context, _ = goal_prior_context(
        model,
        conditioner,
        future_swap,
        swapped_history,
        swapped_goal,
    )

    goal_swap = _roll_fields(
        batch,
        (
            "goal_features",
            "goal_coordinates",
            "goal_valid",
            "goal_rgb",
            "goal_rgb_valid",
        ),
    )
    wrong_goal = encode_explicit_goal(model, goal_swap, history)
    goal_swap_context, _ = goal_prior_context(
        model,
        conditioner,
        goal_swap,
        history,
        wrong_goal,
    )
    query_time = dict(batch)
    query_time["future_times"] = batch["future_times"] + 0.5
    query_time_context, _ = goal_prior_context(
        model,
        conditioner,
        query_time,
        history,
        goal,
    )
    goal_time = dict(batch)
    goal_time["goal_time"] = batch["goal_time"] + 0.5
    goal_time_context, _ = goal_prior_context(
        model,
        conditioner,
        goal_time,
        history,
        goal,
    )

    def maximum(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.float() - right.float()).abs().max())

    def rms(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(
            (left.float() - right.float()).square().mean().sqrt()
        )

    values = {
        "future_swap_history_max_difference": maximum(
            history["slots"],
            swapped_history["slots"],
        ),
        "future_swap_context_max_difference": maximum(
            context,
            swapped_context,
        ),
        "future_swap_posterior_rms_difference": rms(
            posterior,
            swapped_posterior,
        ),
        "goal_swap_context_rms_difference": rms(
            context,
            goal_swap_context,
        ),
        "query_time_context_rms_difference": rms(
            context,
            query_time_context,
        ),
        "goal_time_context_rms_difference": rms(
            context,
            goal_time_context,
        ),
    }
    values["gate"] = {
        "future_supervision_isolated": (
            values["future_swap_history_max_difference"] < 1e-6
            and values["future_swap_context_max_difference"] < 1e-6
            and values["future_swap_posterior_rms_difference"] > 1e-6
        ),
        "goal_changes_prior_context": (
            values["goal_swap_context_rms_difference"] > 1e-6
        ),
        "query_time_changes_prior_context": (
            values["query_time_context_rms_difference"] > 1e-6
        ),
        "goal_time_changes_prior_context": (
            values["goal_time_context_rms_difference"] > 1e-6
        ),
    }
    values["gate"]["all_passed"] = all(values["gate"].values())
    return values
