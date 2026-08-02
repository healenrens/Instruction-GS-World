"""Runtime wiring for fixed-short and terminal latent-effect predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .action_embedding import residual_action_dropout
from .dynamics_runtime import run_object_dynamics
from .feature_readout_runtime import decode_inference_features
from .model_phases import select_dynamics_actions
from .observed_action import posterior_from_targets
from .scale import signed_gap_scale


@dataclass
class TransitionEffects:
    posterior: torch.Tensor
    selected: torch.Tensor
    dynamics: torch.Tensor
    prior_context: torch.Tensor
    action_rgb: tuple[torch.Tensor | None, torch.Tensor | None]
    effect_scale: torch.Tensor


def transition_effect_scales(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    future_scale = signed_gap_scale(
        batch["future_times"], model.config.gap_reference
    )
    if not model.config.dual_horizon_dynamics:
        return future_scale
    if future_scale.shape[1] != 2:
        raise ValueError("dual-horizon inference requires short and goal queries")
    if not bool((batch["future_times"][:, 1] > batch["future_times"][:, 0]).all()):
        raise ValueError("goal query scale must follow the short query")
    return future_scale


def compose_dynamics_effects(model, effects: torch.Tensor) -> torch.Tensor:
    if not model.config.dual_horizon_dynamics:
        return effects
    if effects.shape[1] != 2 or model.effect_composer is None:
        raise ValueError("dual-horizon effects require [short,tail] and a composer")
    total = model.effect_composer(effects[:, 0], effects[:, 1])
    return torch.stack((effects[:, 0], total), dim=1)


def _posterior_transition(
    model,
    history: dict,
    future: dict,
    scale: torch.Tensor,
    condition: torch.Tensor | None,
) -> torch.Tensor:
    return model.latent_actions.posterior(
        history["slots"],
        history["activity"],
        future["slots"],
        future["activity"],
        scale,
        history["center"],
        future["center"],
        condition,
    )


def _state_slice(state: dict, start: int, end: int) -> dict:
    return {
        name: state[name][:, start:end]
        for name in ("slots", "activity", "center")
    }


def prepare_transition_effects(
    model,
    batch: dict[str, torch.Tensor],
    history: dict,
    target_future: dict,
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    condition: torch.Tensor | None,
    use_posterior: bool,
    actions_override: torch.Tensor | None,
    action_free: bool,
) -> TransitionEffects:
    if not model.config.dual_horizon_dynamics:
        posterior, action_rgb = posterior_from_targets(
            model, batch, history, target_future, future_scale, condition
        )
        context = model.prior_context(
            history,
            future_scale,
            history_scale,
            condition,
            batch.get("condition_tokens"),
            batch.get("condition_token_valid"),
        )
        selected = select_dynamics_actions(
            model,
            posterior,
            context,
            use_posterior,
            actions_override,
            action_free,
        )
        selected = residual_action_dropout(
            selected,
            model.config.action_residual_dropout,
            model.training,
            model.config.canonical_action_dim,
        )
        return TransitionEffects(
            posterior, selected, selected, context, action_rgb, future_scale
        )

    if target_future["slots"].shape[1] != 2:
        raise ValueError("dual-horizon effects require short and goal targets")
    effect_scale = transition_effect_scales(model, batch)
    short_scale = effect_scale[:, :1]
    tail_scale = effect_scale[:, 1:2]
    short = _posterior_transition(
        model,
        history,
        _state_slice(target_future, 0, 1),
        short_scale,
        condition,
    )
    tail_history = _state_slice(target_future, 0, 1)
    tail = _posterior_transition(
        model,
        tail_history,
        _state_slice(target_future, 1, 2),
        tail_scale,
        condition,
    )
    posterior = torch.cat((short, tail), dim=1)
    context = model.prior_context(
        history,
        effect_scale,
        history_scale,
        condition,
        batch.get("condition_tokens"),
        batch.get("condition_token_valid"),
    )
    selected = select_dynamics_actions(
        model,
        posterior,
        context,
        use_posterior,
        actions_override,
        action_free,
    )
    selected = residual_action_dropout(
        selected,
        model.config.action_residual_dropout,
        model.training,
        model.config.canonical_action_dim,
    )
    dynamics = compose_dynamics_effects(model, selected)
    return TransitionEffects(
        posterior,
        selected,
        dynamics,
        context,
        (None, None),
        effect_scale,
    )


def supervised_horizon_mask(
    batch: dict[str, torch.Tensor],
    dual_horizon: bool,
    action_free: bool,
) -> torch.Tensor:
    mask = batch.get("future_horizon_valid")
    if mask is None:
        mask = torch.ones_like(batch["future_valid"][..., 0], dtype=torch.bool)
    if mask.shape != batch["future_valid"].shape[:2]:
        raise ValueError("future_horizon_valid must have shape [B,Q]")
    mask = mask.clone()
    if dual_horizon:
        if mask.shape[1] != 2:
            raise ValueError("dual-horizon supervision requires exactly two targets")
        if action_free:
            mask[:, 1] = False
    return mask


def mask_horizon_supervision(
    batch: dict[str, torch.Tensor],
    output: dict,
    action_free: bool,
) -> dict[str, torch.Tensor]:
    mask = supervised_horizon_mask(
        batch, output.get("dual_horizon", False), action_free
    )
    output["future_horizon_valid"] = mask
    output["target_future_activity"] = (
        output["target_future_activity"] * mask[..., None].to(
            output["target_future_activity"].dtype
        )
    )
    loss_batch = dict(batch)
    loss_batch["future_horizon_valid"] = mask
    loss_batch["future_valid"] = batch["future_valid"] & mask[..., None]
    return loss_batch


def _goal_query_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    for name in ("future_features", "future_coordinates", "future_valid", "future_times"):
        result[name] = batch[name][:, 1:2]
    return result


def rollout_goal_prediction(
    model,
    batch: dict[str, torch.Tensor],
    history: dict,
    current_tokens,
    current_slots,
    direct_output,
    effects: TransitionEffects,
    condition: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    short_slots = direct_output.future_slots[:, :1]
    short_activity = direct_output.future_visibility[:, :1]
    if not model.config.factorized_lifecycle:
        short_activity = short_activity * direct_output.future_existence[:, :1]
    history_scale = torch.zeros_like(effects.effect_scale[:, :1])
    mask = torch.zeros_like(short_activity, dtype=torch.bool)
    rollout = run_object_dynamics(
        model,
        short_slots,
        short_activity,
        history_scale,
        effects.effect_scale[:, 1:2],
        effects.selected[:, 1:2],
        mask,
        direct_output.future_centers[:, :1],
        condition,
        history_relative_scale=direct_output.future_relative_scale[:, :1],
        history_relative_disparity=(
            direct_output.future_relative_disparity[:, :1]
        ),
        history_relations=direct_output.future_relations[:, :1],
        history_existence=direct_output.future_existence[:, :1],
    )
    rendered = decode_inference_features(
        model,
        _goal_query_batch(batch),
        current_tokens,
        current_slots,
        rollout,
    )
    return {
        "rollout_goal_slots": rollout.future_slots,
        "rollout_goal_centers": rollout.future_centers,
        "rollout_goal_object_features": model.object_aggregator.decode_feature(
            rollout.future_slots
        ),
        "rollout_goal_rendered_features": rendered,
        "rollout_goal_visibility": rollout.future_visibility,
        "rollout_goal_existence": rollout.future_existence,
        "rollout_goal_relative_scale": rollout.future_relative_scale,
        "rollout_goal_relative_disparity": rollout.future_relative_disparity,
        "rollout_goal_relations": rollout.future_relations,
    }
